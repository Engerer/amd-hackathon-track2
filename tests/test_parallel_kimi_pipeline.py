from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from threading import Lock
from unittest.mock import patch

from PIL import Image

from track2_captioner.caption_pipeline import CaptionPipeline, FRAME_POSITIONS
from track2_captioner.config import Settings, load_settings
from track2_captioner.video_ingest import VideoAsset, compute_dynamic_frame_count


STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.lock = Lock()

    def chat(self, model, messages, **kwargs):
        with self.lock:
            self.calls.append({"model": model, "messages": messages, "kwargs": kwargs})
        system = messages[0]["content"].lower()
        if "humorous non-tech" in system:
            return "A person types at a computer, giving the keyboard its daily workout."
        if "humorous-tech" in system:
            return "A person types at a computer while the keyboard processes another manual data upload."
        if "sarcastic" in system:
            return "A person types at a computer, because apparently the keyboard needs constant supervision."
        return "A person types at a computer in an office."


class ParallelKimiPipelineTests(unittest.TestCase):
    def make_frames(self, directory: Path) -> list[Path]:
        frames = []
        for index in range(5):
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
        )
        pipeline.work_dir = directory / "work"
        pipeline.dry_run = False
        pipeline.max_frames = 5
        pipeline.run_checks = False
        pipeline.client = FakeClient()
        return pipeline

    def test_frame_budget_is_five(self) -> None:
        for duration in (30.0, 60.0, 120.0, 240.0):
            self.assertEqual(compute_dynamic_frame_count(duration, 99), 5)

    def test_each_style_gets_same_five_frames_and_unique_system_prompt(self) -> None:
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
            systems.add(call["messages"][0]["content"])
            content = call["messages"][1]["content"]
            self.assertEqual(sum(part.get("type") == "image_url" for part in content), 5)
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

        self.assertEqual(result["frame_count"], 5)
        self.assertEqual(set(result["captions"]), set(STYLES))
        self.assertEqual(len(pipeline.client.calls), 4)
        self.assertEqual(result["observations"], {})
        self.assertEqual(result["checks"], {})

    def test_default_models_are_kimi(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            settings = load_settings()
        self.assertEqual(settings.model, "accounts/fireworks/models/kimi-k2p6")
        self.assertEqual(settings.caption_model, settings.model)


if __name__ == "__main__":
    unittest.main()
