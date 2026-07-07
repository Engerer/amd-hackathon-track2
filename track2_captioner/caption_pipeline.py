from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from track2_captioner.config import Settings
from track2_captioner.fireworks_client import FireworksClient, image_to_data_url
from track2_captioner.json_tools import parse_json_object
from track2_captioner.prompts import STYLE_PROMPTS, load_prompt
from track2_captioner.video_ingest import VideoAsset, extract_frames


EMPTY_OBSERVATIONS = {
    "setting": "Dry run placeholder setting.",
    "subjects": ["sample subject"],
    "actions": ["sample action"],
    "sequence": ["sample beginning", "sample middle", "sample ending"],
    "visible_text": [],
    "audio_or_speech": [],
    "uncertainties": ["Dry run did not inspect the video."],
}


class CaptionPipeline:
    def __init__(
        self,
        settings: Settings,
        work_dir: Path,
        dry_run: bool = False,
        max_frames: int = 12,
        run_checks: bool = True,
    ) -> None:
        self.settings = settings
        self.work_dir = work_dir
        self.dry_run = dry_run
        self.max_frames = max_frames
        self.run_checks = run_checks
        self.client = None if dry_run else FireworksClient(
            settings.api_key,
            settings.base_url,
            settings.proxy_url,
            settings.proxy_token,
        )

    def process(self, asset: VideoAsset, styles: list[str] | None = None) -> dict[str, Any]:
        selected_styles = styles or list(STYLE_PROMPTS)
        frame_dir = self.work_dir / asset.video_id
        frames = [] if self.dry_run else extract_frames(asset.path, frame_dir, self.max_frames)
        transcript = self._read_transcript(asset)
        observations = self._observe(asset, frames, transcript)
        captions = {
            style: self._caption(style, observations)
            for style in selected_styles
            if style in STYLE_PROMPTS
        }
        checks = {}
        if self.run_checks:
            checks = {
                style: self._check(style, observations, caption)
                for style, caption in captions.items()
            }

        return {
            "video_id": asset.video_id,
            "source_path": str(asset.path),
            "observations": observations,
            "captions": captions,
            "checks": checks,
        }

    def _observe(self, asset: VideoAsset, frames: list[Path], transcript: str) -> dict[str, Any]:
        if self.dry_run:
            return dict(EMPTY_OBSERVATIONS)

        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"Video id: {asset.video_id}\n"
                    f"Optional transcript:\n{transcript or '[none provided]'}\n\n"
                    "Analyze these sampled frames in chronological order."
                ),
            }
        ]
        for frame in frames:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(frame)},
                }
            )

        assert self.client is not None
        response = self.client.chat(
            self.settings.model,
            [
                {"role": "system", "content": load_prompt("perception_system.txt")},
                {"role": "user", "content": content},
            ],
        )
        return self._parse_or_repair_json(response, "perception observations")

    def _caption(self, style: str, observations: dict[str, Any]) -> str:
        if self.dry_run:
            examples = {
                "formal": "A sample subject performs a sample action in a simple sequence.",
                "sarcastic": "A sample subject performs a sample action, because apparently the plot needed momentum.",
                "humorous_tech": "A sample subject executes the action pipeline with no visible rollback plan.",
                "humorous_non_tech": "A sample subject gets things moving, and honestly, that is more than some Mondays manage.",
            }
            return examples[style]

        assert self.client is not None
        prompt = load_prompt(STYLE_PROMPTS[style])
        response = self.client.chat(
            self.settings.model,
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(observations, indent=2)},
            ],
            max_tokens=220,
        )
        return response.strip().strip('"')

    def _check(self, style: str, observations: dict[str, Any], caption: str) -> dict[str, str]:
        if self.dry_run:
            return {"accuracy": "unknown", "tone": "unknown", "notes": "Dry run skipped model judging."}

        assert self.client is not None
        response = self.client.chat(
            self.settings.judge_model,
            [
                {"role": "system", "content": load_prompt("judge.txt")},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "target_style": style,
                            "observations": observations,
                            "caption": caption,
                        },
                        indent=2,
                    ),
                },
            ],
            max_tokens=160,
        )
        parsed = parse_json_object(response)
        return {
            "accuracy": str(parsed.get("accuracy", "fail")),
            "tone": str(parsed.get("tone", "fail")),
            "notes": str(parsed.get("notes", "")),
        }

    def _parse_or_repair_json(self, response: str, label: str) -> dict[str, Any]:
        try:
            return parse_json_object(response)
        except Exception:
            assert self.client is not None
            repaired = self.client.chat(
                self.settings.model,
                [
                    {
                        "role": "system",
                        "content": (
                            "Repair the user's text into strict valid JSON only. "
                            "Do not add commentary, markdown, or new facts."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Repair this {label} JSON:\n\n{response}",
                    },
                ],
                max_tokens=700,
            )
            return parse_json_object(repaired)

    @staticmethod
    def _read_transcript(asset: VideoAsset) -> str:
        if asset.transcript_path is None:
            return ""
        return asset.transcript_path.read_text(encoding="utf-8").strip()
