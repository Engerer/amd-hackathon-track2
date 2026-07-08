from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from track2_captioner.harness import (
    DEFAULT_STYLES,
    adaptive_frame_budget,
    fallback_captions,
    read_tasks,
    run_harness,
)


class HarnessContractTests(unittest.TestCase):
    def test_read_tasks_requires_array(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            path = Path(temp_dir_name) / "tasks.json"
            path.write_text('{"task_id": "not-an-array"}', encoding="utf-8")

            with self.assertRaises(ValueError):
                read_tasks(path)

    def test_dry_run_writes_requested_styles_without_download(self) -> None:
        task = {
            "task_id": "remote_clip",
            "video_url": "https://example.com/clip.mp4",
            "styles": DEFAULT_STYLES,
        }

        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            input_path = temp_dir / "tasks.json"
            output_path = temp_dir / "results.json"
            input_path.write_text(json.dumps([task]), encoding="utf-8")

            env = {
                "TRACK2_DRY_RUN": "true",
                "TRACK2_RUNTIME_BUDGET_SECONDS": "60",
            }
            with patch.dict(os.environ, env, clear=False):
                with patch("track2_captioner.harness.download_video") as download_video:
                    run_harness(input_path, output_path)
                    download_video.assert_not_called()

            results = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual([item["task_id"] for item in results], ["remote_clip"])
            captions = results[0]["captions"]
            self.assertEqual(set(captions), set(DEFAULT_STYLES))
            self.assertTrue(all(isinstance(value, str) and value for value in captions.values()))

    def test_fallback_captions_use_observations_and_style(self) -> None:
        observations = {
            "summary": "A kitten walks through green foliage.",
            "setting": "garden",
            "subjects": ["orange kitten"],
            "key_objects": ["green leaves"],
            "actions": ["walking"],
        }

        captions = fallback_captions(DEFAULT_STYLES, observations)

        self.assertIn("kitten", captions["formal"].lower())
        self.assertIn("clearly", captions["sarcastic"].lower())
        self.assertIn("rollback", captions["humorous_tech"].lower())
        self.assertIn("everyday", captions["humorous_non_tech"].lower())

    def test_adaptive_frame_budget_reduces_when_time_is_tight(self) -> None:
        self.assertEqual(adaptive_frame_budget(32, remaining_seconds=60, tasks_left=1), 8)
        self.assertEqual(adaptive_frame_budget(32, remaining_seconds=120, tasks_left=1), 12)
        self.assertEqual(adaptive_frame_budget(32, remaining_seconds=250, tasks_left=2), 18)
        self.assertEqual(adaptive_frame_budget(32, remaining_seconds=500, tasks_left=2), 32)


if __name__ == "__main__":
    unittest.main()
