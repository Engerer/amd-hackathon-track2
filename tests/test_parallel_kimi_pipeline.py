from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from threading import Lock
from unittest.mock import patch

from PIL import Image

from track2_captioner.caption_pipeline import (
    CaptionPipeline,
    FRAME_POSITIONS,
    QWEN_SELECTOR_SYSTEM,
    SHARED_VISUAL_SYSTEM,
)
from track2_captioner.config import Settings, load_settings
from track2_captioner.video_ingest import (
    VideoAsset,
    _extract_model_selected_frames,
    compute_dynamic_frame_count,
)


STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.lock = Lock()

    def chat(self, model, messages, **kwargs):
        with self.lock:
            self.calls.append({"model": model, "messages": messages, "kwargs": kwargs})
        system = messages[0]["content"].lower()
        if "keyframe selection expert" in system:
            return '{"selected_indices":[4,12,22]}'
        if "humorous non-tech" in system:
            return "A person types at a computer, giving the keyboard its daily workout."
        if "humorous-tech" in system:
            return "A person types at a computer while the keyboard processes another manual data upload."
        if "sarcastic" in system:
            return "A person types at a computer, because apparently the keyboard needs constant supervision."
        return "A person types at a computer in an office."


class ParallelKimiPipelineTests(unittest.TestCase):
    def make_frames(self, directory: Path, count: int = 3) -> list[Path]:
        frames = []
        for index in range(count):
            path = directory / f"frame_{index + 1:03d}.jpg"
            Image.new("RGB", (32, 18), (50 + index, 90, 130)).save(path)
            frames.append(path)
        return frames

    def make_pipeline(self, directory: Path) -> CaptionPipeline:
        pipeline = object.__new__(CaptionPipeline)
        pipeline.settings = Settings(
            api_key="",
            model="accounts/fireworks/models/kimi-k2p6",
            caption_model="accounts/fireworks/models/kimi-k2p6",
            judge_model="unused",
            selector_model="accounts/fireworks/models/qwen3p7-plus",
        )
        pipeline.work_dir = directory / "work"
        pipeline.dry_run = False
        pipeline.max_frames = 3
        pipeline.run_checks = False
        pipeline.client = FakeClient()
        return pipeline

    def test_frame_budget_is_three(self) -> None:
        for duration in (30.0, 60.0, 120.0, 240.0):
            self.assertEqual(compute_dynamic_frame_count(duration, 99), 3)

    def test_shared_prompt_allows_reference_style_figurative_humor(self) -> None:
        self.assertIn("representative moment", SHARED_VISUAL_SYSTEM)
        self.assertIn("figurative personification", SHARED_VISUAL_SYSTEM)
        self.assertIn("at least two concrete visual anchors", SHARED_VISUAL_SYSTEM)

    def test_qwen_selects_one_frame_from_each_temporal_third(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            pipeline = self.make_pipeline(directory)
            candidates = self.make_frames(directory, 25)
            selected = pipeline._select_frame_indices(candidates)

        self.assertEqual(selected, [3, 11, 21])
        self.assertEqual(len(pipeline.client.calls), 1)
        call = pipeline.client.calls[0]
        self.assertEqual(call["model"], "accounts/fireworks/models/qwen3p7-plus")
        self.assertEqual(call["messages"][0]["content"], QWEN_SELECTOR_SYSTEM)
        content = call["messages"][1]["content"]
        self.assertEqual(sum(part.get("type") == "image_url" for part in content), 25)
        self.assertEqual(call["kwargs"]["reasoning_effort"], "medium")
        self.assertIn("json_schema", call["kwargs"])

    def test_qwen_failure_uses_local_quality_selection(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            candidates = self.make_frames(directory, 25)
            quality_selection = [candidates[2], candidates[12], candidates[22]]

            def failing_selector(_candidates):
                raise RuntimeError("selector unavailable")

            with (
                patch(
                    "track2_captioner.video_ingest._extract_candidate_frames",
                    return_value=candidates,
                ),
                patch(
                    "track2_captioner.video_ingest.select_quality_frames",
                    return_value=quality_selection,
                ) as quality_mock,
            ):
                selected = _extract_model_selected_frames(
                    Path("video.mp4"),
                    directory / "selection",
                    3,
                    896,
                    failing_selector,
                )

        self.assertEqual(len(selected), 3)
        quality_mock.assert_called_once_with(candidates, 3)

    def test_each_style_gets_same_three_frames_and_unique_system_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            pipeline = self.make_pipeline(directory)
            frames = self.make_frames(directory)
            captions = pipeline._captions(STYLES, frames)

        self.assertEqual(set(captions), set(STYLES))
        self.assertEqual(len(pipeline.client.calls), 4)
        systems = set()
        for call in pipeline.client.calls:
            self.assertEqual(call["model"], "accounts/fireworks/models/kimi-k2p6")
            self.assertEqual(call["kwargs"]["reasoning_effort"], "medium")
            systems.add(call["messages"][0]["content"])
            content = call["messages"][1]["content"]
            self.assertEqual(sum(part.get("type") == "image_url" for part in content), 3)
            labels = [part["text"] for part in content if part.get("type") == "text"]
            for position in FRAME_POSITIONS:
                self.assertTrue(any(position in label for label in labels))
        self.assertEqual(len(systems), 4)

    def test_process_recombines_track2_caption_map(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            pipeline = self.make_pipeline(directory)
            frames = self.make_frames(directory)
            with patch("track2_captioner.caption_pipeline.extract_frames", return_value=frames):
                result = pipeline.process(VideoAsset("sample", Path("video.mp4")), STYLES)

        self.assertEqual(result["frame_count"], 3)
        self.assertEqual(set(result["captions"]), set(STYLES))
        self.assertEqual(len(pipeline.client.calls), 4)
        self.assertEqual(result["observations"], {})
        self.assertEqual(result["checks"], {})

    def test_default_models_are_kimi(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            settings = load_settings()
        self.assertEqual(settings.model, "accounts/fireworks/models/kimi-k2p6")
        self.assertEqual(settings.caption_model, settings.model)
        self.assertEqual(settings.selector_model, "accounts/fireworks/models/qwen3p7-plus")
        self.assertEqual(settings.reasoning_effort, "medium")


if __name__ == "__main__":
    unittest.main()
