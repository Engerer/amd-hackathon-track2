from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from track2_captioner.caption_pipeline import CaptionPipeline
from track2_captioner.harness import DEFAULT_STYLES, fallback_captions
from track2_captioner.video_ingest import (
    FrameCandidate,
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

    def test_model_pack_contains_only_one_five_frame_storyboard(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            frames: list[Path] = []
            for index in range(5):
                frame_path = temp_dir / f"frame_{index:03d}.jpg"
                image = Image.new("RGB", (768, 432), (40 + index * 8, 80, 120))
                draw = ImageDraw.Draw(image)
                draw.rectangle((50 + index, 50, 300, 300), outline="white", width=4)
                image.save(frame_path, quality=90)
                frames.append(frame_path)

            pipeline = object.__new__(CaptionPipeline)
            model_images = pipeline._prepare_model_images(frames, temp_dir)

            self.assertEqual(len(model_images), 1)
            self.assertEqual(model_images[0].name, "storyboard.jpg")

    def test_fallbacks_cover_every_style_with_distinct_tones(self) -> None:
        captions = fallback_captions(DEFAULT_STYLES)
        self.assertEqual(set(captions), set(DEFAULT_STYLES))
        self.assertEqual(len(set(captions.values())), len(DEFAULT_STYLES))


if __name__ == "__main__":
    unittest.main()
