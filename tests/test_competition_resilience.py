from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from track2_captioner.caption_pipeline import CaptionPipeline, FALLBACK_CAPTIONS
from track2_captioner.config import Settings
from track2_captioner.fireworks_client import FireworksClient, RetryableHTTPError
from track2_captioner.harness import DEFAULT_STYLES, run_harness


class SelectorClient:
    def __init__(self, response: str) -> None:
        self.response = response

    def chat(self, *_args, **_kwargs) -> str:
        return self.response


def make_pipeline(directory: Path, client: object) -> CaptionPipeline:
    pipeline = object.__new__(CaptionPipeline)
    pipeline.settings = Settings(
        api_key="",
        model="accounts/fireworks/models/kimi-k2p6",
        caption_model="accounts/fireworks/models/kimi-k2p6",
        judge_model="unused",
    )
    pipeline.work_dir = directory / "frames"
    pipeline.dry_run = False
    pipeline.max_frames = 3
    pipeline.run_checks = False
    pipeline.client = client
    return pipeline


def make_frames(directory: Path, count: int) -> list[Path]:
    frames: list[Path] = []
    for index in range(count):
        path = directory / f"frame_{index:03d}.jpg"
        Image.new("RGB", (16, 9), (index * 5 % 255, 80, 120)).save(path)
        frames.append(path)
    return frames


class CompetitionResilienceTests(unittest.TestCase):
    def test_one_failed_kimi_style_retries_then_falls_back_only_that_style(self) -> None:
        class RetryingClient(FireworksClient):
            def __init__(self) -> None:
                super().__init__("", "https://example.invalid", proxy_url="https://proxy.invalid")
                self.attempts: dict[str, int] = {}

            def chat(self, _model, messages, **_kwargs):
                system = messages[0]["content"].lower()
                markers = {
                    "formal": "write a clear formal",
                    "sarcastic": "write a sarcastic",
                    "humorous_tech": "write a humorous-tech",
                    "humorous_non_tech": "write a humorous non-tech",
                }
                style = next(style for style, marker in markers.items() if marker in system)

                def call() -> str:
                    self.attempts[style] = self.attempts.get(style, 0) + 1
                    if style == "sarcastic":
                        raise RetryableHTTPError(503, "temporary outage")
                    return f"A grounded {style} caption."

                return self._with_retry(call, None)

        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            client = RetryingClient()
            pipeline = make_pipeline(directory, client)
            captions = pipeline._captions(DEFAULT_STYLES, make_frames(directory, 3))

        self.assertEqual(client.attempts["sarcastic"], 2)
        self.assertEqual(captions["sarcastic"], FALLBACK_CAPTIONS["sarcastic"])
        for style in set(DEFAULT_STYLES) - {"sarcastic"}:
            self.assertEqual(client.attempts[style], 1)
            self.assertNotEqual(captions[style], FALLBACK_CAPTIONS[style])

    def test_caption_cleaner_removes_reasoning_and_rejects_analysis(self) -> None:
        text = "<think>private reasoning</think> Final answer: Caption: A cat watches the garden."
        self.assertEqual(CaptionPipeline._clean_caption(text), "A cat watches the garden.")
        self.assertEqual(
            CaptionPipeline._clean_caption("The prompt asks me to analyze the frames first."),
            "",
        )
        self.assertEqual(CaptionPipeline._clean_caption("word " * 71), "")

    def test_non_retryable_model_error_is_not_retried(self) -> None:
        client = FireworksClient("", "https://example.invalid", proxy_url="https://proxy.invalid")
        attempts = 0

        def call() -> str:
            nonlocal attempts
            attempts += 1
            raise RuntimeError("HTTP 401: authentication failed")

        with self.assertRaises(RuntimeError):
            client._with_retry(call, None)
        self.assertEqual(attempts, 1)

    def test_twelve_task_timing_simulation_and_output_order(self) -> None:
        state = {"active": 0, "peak": 0}
        lock = threading.Lock()

        class TimedPipeline:
            def __init__(self, **_kwargs) -> None:
                pass

            def process(self, asset, styles, deadline):
                with lock:
                    state["active"] += 1
                    state["peak"] = max(state["peak"], state["active"])
                time.sleep(0.03 if int(asset.video_id[1:]) % 2 == 0 else 0.005)
                with lock:
                    state["active"] -= 1
                return {"captions": {style: f"{asset.video_id}-{style}" for style in styles}}

        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            input_path = directory / "input" / "tasks.json"
            output_path = directory / "output" / "results.json"
            input_path.parent.mkdir()
            tasks = [
                {"task_id": f"v{index}", "video_url": "unused", "styles": DEFAULT_STYLES}
                for index in range(12)
            ]
            input_path.write_text(json.dumps(tasks), encoding="utf-8")
            settings = Settings("", "model", "model", "judge", task_workers=2)
            started = time.monotonic()
            with (
                patch("track2_captioner.harness.load_settings", return_value=settings),
                patch("track2_captioner.harness.CaptionPipeline", TimedPipeline),
                patch("track2_captioner.harness.download_video", return_value=Path("video.mp4")),
            ):
                run_harness(input_path, output_path)
            elapsed = time.monotonic() - started
            output = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(state["peak"], 2)
        self.assertLess(elapsed, 0.20)
        self.assertEqual([item["task_id"] for item in output], [f"v{i}" for i in range(12)])

    def test_expired_global_deadline_skips_work_and_writes_all_results(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            input_path = directory / "tasks.json"
            output_path = directory / "results.json"
            tasks = [{"task_id": "v1", "video_url": "unused", "styles": DEFAULT_STYLES}]
            input_path.write_text(json.dumps(tasks), encoding="utf-8")
            settings = Settings("", "model", "model", "judge", runtime_budget_seconds=0)
            with (
                patch("track2_captioner.harness.load_settings", return_value=settings),
                patch("track2_captioner.harness.download_video") as download_mock,
            ):
                run_harness(input_path, output_path)
            output = json.loads(output_path.read_text(encoding="utf-8"))

        download_mock.assert_not_called()
        self.assertEqual(output[0]["captions"], FALLBACK_CAPTIONS)

    def test_clean_working_directory_evaluator_mount_smoke(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as name:
            mount = Path(name)
            input_dir = mount / "input"
            output_dir = mount / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            dummy_video = mount / "clip.mp4"
            dummy_video.write_bytes(b"dry-run placeholder")
            (input_dir / "tasks.json").write_text(
                json.dumps([{
                    "task_id": "v1",
                    "video_url": str(dummy_video),
                    "styles": DEFAULT_STYLES,
                }]),
                encoding="utf-8",
            )
            env = os.environ.copy()
            env.update({
                "PYTHONPATH": str(repo),
                "TRACK2_DRY_RUN": "1",
                "TRACK2_INPUT": str(input_dir / "tasks.json"),
                "TRACK2_OUTPUT": str(output_dir / "results.json"),
            })
            completed = subprocess.run(
                [sys.executable, "-m", "track2_captioner.harness"],
                cwd=mount,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            output = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(output[0]["task_id"], "v1")
        self.assertEqual(set(output[0]["captions"]), set(DEFAULT_STYLES))

    @unittest.skipUnless(
        shutil.which("docker") and os.getenv("TRACK2_RUN_DOCKER_TEST") == "1",
        "set TRACK2_RUN_DOCKER_TEST=1 to build and run the evaluator container",
    )
    def test_clean_container_evaluator_mount(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        image = "amd-track2-captioner:integration-test"
        with tempfile.TemporaryDirectory() as name:
            mount = Path(name)
            input_dir = mount / "input"
            output_dir = mount / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            (input_dir / "clip.mp4").write_bytes(b"dry-run placeholder")
            (input_dir / "tasks.json").write_text(
                json.dumps([{
                    "task_id": "v1",
                    "video_url": "/input/clip.mp4",
                    "styles": DEFAULT_STYLES,
                }]),
                encoding="utf-8",
            )
            subprocess.run(
                ["docker", "build", "-t", image, "."],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
            subprocess.run(
                [
                    "docker", "run", "--rm",
                    "-e", "TRACK2_DRY_RUN=1",
                    "-v", f"{input_dir}:/input:ro",
                    "-v", f"{output_dir}:/output",
                    image,
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            output = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(output[0]["task_id"], "v1")
        self.assertEqual(set(output[0]["captions"]), set(DEFAULT_STYLES))


if __name__ == "__main__":
    unittest.main()
