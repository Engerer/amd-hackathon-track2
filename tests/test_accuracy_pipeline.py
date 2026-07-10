from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from track2_captioner.caption_pipeline import CaptionPipeline, EMPTY_EVIDENCE
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
        self.assertLessEqual(selected[0].timestamp, 10.0)

    def test_evidence_ledger_structure(self) -> None:
        """Test that evidence ledger has the required typed fields."""
        required_fields = [
            "summary", "setting", "subjects", "objects", "actions",
            "claims", "uncertainties",
        ]
        for field in required_fields:
            self.assertIn(field, EMPTY_EVIDENCE)
        # Claims should be a list
        self.assertIsInstance(EMPTY_EVIDENCE["claims"], list)
        # Contradictions field should exist
        self.assertIn("contradictions", EMPTY_EVIDENCE)

    def test_parallel_candidate_generation_dry_run(self) -> None:
        """Test that dry run produces captions for all styles."""
        settings = Settings(
            api_key="", model="m", caption_model="m", judge_model="m",
        )
        pipeline = CaptionPipeline(
            settings=settings,
            work_dir=Path("/tmp/test"),
            dry_run=True,
        )
        captions = pipeline._generate_and_select_captions(
            dict(EMPTY_EVIDENCE), [], DEFAULT_STYLES, enable_retry=False,
            clip_deadline=time.monotonic() + 10.0
        )
        self.assertEqual(set(captions.keys()), set(DEFAULT_STYLES))
        for style in DEFAULT_STYLES:
            self.assertTrue(len(captions[style]) > 0)

    def test_caption_pipeline_perception_call(self) -> None:
        """Test that the pipeline calls the multimodal perception model."""
        class FakeClient:
            def __init__(self) -> None:
                self.calls: list[dict] = []

            def chat(self, model, messages, **kwargs):
                self.calls.append({"model": model, "messages": messages, "kwargs": kwargs})
                return json.dumps({
                    "summary": "A person walks down a city street.",
                    "setting": "urban street",
                    "subjects": ["person"],
                    "subject_counts": {"person": "1"},
                    "objects": ["buildings", "traffic light"],
                    "actions": ["walking"],
                    "ocr": [],
                    "camera_motion": "static",
                    "claims": [
                        {
                            "type": "action",
                            "text": "a person walks along the sidewalk",
                            "confidence": 0.9,
                            "timestamps": ["00:02"],
                        }
                    ],
                    "uncertainties": [],
                    "contradictions": [],
                })

        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            frames: list[Path] = []
            for index in range(5):
                frame_path = temp_dir / f"frame_{index:03d}_t{index * 10:09.3f}.jpg"
                Image.new("RGB", (32, 18), (40 + index, 80, 120)).save(frame_path)
                frames.append(frame_path)

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
            pipeline.enable_style_retry = False

            evidence = pipeline._extract_evidence(
                VideoAsset("sample", Path("video.mp4")),
                keyframes=frames,
                video_duration=120.0,
                clip_deadline=time.monotonic() + 10.0,
            )

            self.assertIn("claims", evidence)
            self.assertIn("subjects", evidence)
            self.assertEqual(evidence["setting"], "urban street")

    def test_fallbacks_cover_every_style_with_distinct_tones(self) -> None:
        captions = fallback_captions(DEFAULT_STYLES)
        self.assertEqual(set(captions), set(DEFAULT_STYLES))
        self.assertEqual(len(set(captions.values())), len(DEFAULT_STYLES))
        self.assertEqual(task_styles({"styles": ["formal"]}), ["formal"])

    def test_grounded_check_returns_continuous_scores(self) -> None:
        """Test that the check function returns continuous scores, not just pass/fail."""
        settings = Settings(
            api_key="", model="m", caption_model="m", judge_model="m",
        )
        pipeline = CaptionPipeline(
            settings=settings,
            work_dir=Path("/tmp/test"),
            dry_run=True,
        )
        result = pipeline._check(
            "formal",
            dict(EMPTY_EVIDENCE),
            "A sample caption.",
        )
        # Should have continuous scores
        self.assertIn("factual_accuracy", result)
        self.assertIn("style_strength", result)
        self.assertIn("overall_score", result)
        # Should also have backward-compatible pass/fail
        self.assertIn("accuracy", result)
        self.assertIn("tone", result)

    def test_caption_issues_detects_missing_and_weak(self) -> None:
        """Test style issue detection."""
        issues = CaptionPipeline._caption_issues(
            ["formal", "sarcastic", "humorous_tech"],
            {
                "formal": "A person walks.",
                "sarcastic": "A person walks.",  # No sarcasm markers
                # humorous_tech missing
            },
        )
        self.assertTrue(any("sarcastic" in i and "weak" in i for i in issues))
        self.assertTrue(any("humorous_tech" in i and "missing" in i for i in issues))


if __name__ == "__main__":
    unittest.main()
