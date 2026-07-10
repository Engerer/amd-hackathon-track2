from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from track2_captioner.caption_pipeline import CaptionPipeline
from track2_captioner.config import Settings
from track2_captioner.harness import DEFAULT_STYLES, fallback_captions, task_styles
from track2_captioner.video_ingest import (
    FrameCandidate,
    VideoAsset,
    _select_adaptive_candidates,
    compute_dynamic_frame_count,
    extract_frames,
)


class AccuracyPipelineTests(unittest.TestCase):
    def test_fast_profile_preserves_all_timeline_frames(self) -> None:
        expected = [Path(f"frame_{index:03d}.jpg") for index in range(1, 6)]
        with (
            patch("track2_captioner.video_ingest.probe_duration_seconds", return_value=60.0),
            patch("track2_captioner.video_ingest._extract_timestamp_frames_opencv", return_value=expected),
        ):
            actual = extract_frames(Path("video.mp4"), Path("frames"), 15, frame_profile="fast")
        self.assertEqual(actual, expected)

    def test_hybrid_budget_is_five_for_every_challenge_duration(self) -> None:
        for duration in (30.0, 60.0, 90.0, 120.0):
            with self.subTest(duration=duration):
                self.assertEqual(compute_dynamic_frame_count(duration, 20, "hybrid"), 5)

    def test_hybrid_selector_returns_full_budget_with_timeline_coverage(self) -> None:
        candidates = [
            FrameCandidate(
                timestamp=float(index * 4),
                score=(index % 7) / 7,
                sharpness=100.0 + index,
                brightness=120.0,
                motion=float(index % 5),
                hash_value=index,
            )
            for index in range(30)
        ]
        selected = _select_adaptive_candidates(candidates, duration=116.0, final_count=5)
        self.assertEqual(len(selected), 5)
        self.assertLessEqual(selected[0].timestamp, 4.0)
        self.assertGreaterEqual(selected[-1].timestamp, 112.0)

    def test_model_pack_contains_five_timestamped_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            frames: list[Path] = []
            for index in range(5):
                timestamp = 0.5 + (index * 29.75)
                frame_path = temp_dir / f"frame_{index:03d}_t{timestamp:09.3f}.jpg"
                image = Image.new("RGB", (896, 504), (40 + index * 8, 80, 120))
                draw = ImageDraw.Draw(image)
                draw.rectangle((50 + index, 50, 300, 300), outline="white", width=4)
                image.save(frame_path, quality=90)
                frames.append(frame_path)

            pipeline = object.__new__(CaptionPipeline)
            model_images = pipeline._prepare_model_images(frames, temp_dir, 120.0)

            self.assertEqual(len(model_images), 5)
            self.assertTrue(all("_t" in image.name for image in model_images))
            with Image.open(model_images[0]) as prepared:
                self.assertEqual(prepared.size, (896, 504))
                self.assertTrue(all(channel < 45 for channel in prepared.getpixel((2, 2))))

            content = pipeline._direct_caption_content(
                VideoAsset("sample", Path("video.mp4")),
                model_images,
                5,
                120.0,
            )
            request = json.loads(content[0]["text"])
            self.assertEqual(request["video_duration_seconds"], 120.0)
            self.assertEqual(len(request["frame_timestamps_seconds"]), 5)
            self.assertEqual(request["frame_timestamps_seconds"][0], 0.5)
            self.assertIn("factual ground-truth observations", request["task"])
            self.assertNotIn("requested_styles", request)
            self.assertNotIn("optional_transcript", request)
            self.assertNotIn("optional_audio_context", request)
            self.assertEqual(sum(item["type"] == "image_url" for item in content), 5)
            self.assertEqual(sum(item["type"] == "text" for item in content), 6)
            self.assertIn("Frame 1/5 at 00:00.5 of total 02:00.0", content[1]["text"])
            self.assertEqual(content[2]["type"], "image_url")

            style_request = pipeline._style_caption_request(
                {
                    "summary": "A cyclist rides along a path.",
                    "setting": "tree-lined path",
                    "subjects": ["cyclist"],
                    "actions": ["rides along the path"],
                },
                DEFAULT_STYLES,
            )
            self.assertEqual(style_request["generation_order"], DEFAULT_STYLES)
            self.assertEqual(set(style_request["style_instructions"]), set(DEFAULT_STYLES))
            self.assertTrue(
                all("Few-shot style reference" in prompt for prompt in style_request["style_instructions"].values())
            )
            self.assertTrue(
                any("Do not reuse the sentence structures or opening phrases" in rule for rule in style_request["rules"])
            )

    def test_caption_pipeline_uses_one_visual_call_then_one_text_only_call(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.calls: list[dict] = []

            def chat(self, model, messages, **kwargs):
                self.calls.append({"model": model, "messages": messages, "kwargs": kwargs})
                if len(self.calls) == 1:
                    return json.dumps({
                        "summary": "A cyclist rides along a paved path.",
                        "setting": "tree-lined paved path",
                        "subjects": ["cyclist"],
                        "key_objects": ["bicycle", "trees"],
                        "actions": ["rides along the path"],
                        "timeline": ["beginning: cyclist enters", "end: cyclist continues"],
                        "visible_text": [],
                        "uncertainties": [],
                    })
                return json.dumps({
                    "captions": {
                        "formal": "A cyclist rides steadily along a tree-lined paved path.",
                        "sarcastic": "Along the tree-lined path goes a cyclist, because apparently coasting was too ordinary.",
                        "humorous_tech": "With low latency, a cyclist processes the paved route like a well-tuned scheduler.",
                        "humorous_non_tech": "Steady as a Monday coffee run, a cyclist rolls along the path.",
                    }
                })

        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            model_images: list[Path] = []
            for index in range(5):
                frame_path = temp_dir / f"frame_{index:03d}_t{index * 10:09.3f}.jpg"
                Image.new("RGB", (32, 18), (40 + index, 80, 120)).save(frame_path)
                model_images.append(frame_path)

            client = FakeClient()
            pipeline = object.__new__(CaptionPipeline)
            pipeline.dry_run = False
            pipeline.client = client
            pipeline.settings = Settings(
                api_key="",
                model="vision-model",
                caption_model="style-model",
                judge_model="judge-model",
            )

            captions, observations = pipeline._direct_captions(
                VideoAsset("sample", Path("video.mp4")),
                model_images,
                5,
                120.0,
                DEFAULT_STYLES,
                False,
            )

            self.assertEqual(len(client.calls), 2)
            self.assertEqual(client.calls[0]["model"], "vision-model")
            self.assertEqual(client.calls[1]["model"], "style-model")
            visual_content = client.calls[0]["messages"][1]["content"]
            self.assertEqual(sum(item["type"] == "image_url" for item in visual_content), 5)
            self.assertIsInstance(client.calls[1]["messages"][1]["content"], str)
            self.assertNotIn("image_url", client.calls[1]["messages"][1]["content"])
            self.assertEqual(set(captions), set(DEFAULT_STYLES))
            self.assertEqual(observations["setting"], "tree-lined paved path")

    def test_fallbacks_cover_every_style_with_distinct_tones(self) -> None:
        captions = fallback_captions(DEFAULT_STYLES)
        self.assertEqual(set(captions), set(DEFAULT_STYLES))
        self.assertEqual(len(set(captions.values())), len(DEFAULT_STYLES))
        self.assertEqual(task_styles({"styles": ["formal"]}), DEFAULT_STYLES)


if __name__ == "__main__":
    unittest.main()
