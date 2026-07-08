from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from track2_captioner.config import Settings
from track2_captioner.fireworks_client import FireworksClient, image_to_data_url
from track2_captioner.json_tools import parse_json_object
from track2_captioner.prompts import STYLE_PROMPTS, load_prompt
from track2_captioner.video_ingest import VideoAsset, extract_frames


OBSERVATION_SCHEMA = json.dumps({
    "setting": "one concise sentence",
    "subjects": ["visible subject or object"],
    "actions": ["important visible action"],
    "sequence": ["what happens first", "what happens next", "what happens last"],
    "visible_text": ["text visible in frames, or empty array"],
    "audio_or_speech": ["relevant transcript or audio cue, or empty array"],
    "uncertainties": ["anything unclear or ambiguous, or empty array"],
}, indent=2)

CHECK_SCHEMA = json.dumps({
    "accuracy": "pass or fail",
    "tone": "pass or fail",
    "notes": "brief explanation",
}, indent=2)

CREATIVE_STYLES = {"sarcastic", "humorous_tech", "humorous_non_tech"}


EMPTY_OBSERVATIONS = {
    "setting": "Dry run placeholder setting.",
    "subjects": ["sample subject"],
    "actions": ["sample action"],
    "sequence": ["sample beginning", "sample middle", "sample ending"],
    "visible_text": [],
    "audio_or_speech": [],
    "uncertainties": ["Dry run did not inspect the video."],
}

DRY_RUN_CAPTIONS = {
    "formal": "A sample subject performs a sample action in a simple sequence.",
    "sarcastic": "A sample subject performs a sample action, because apparently the plot needed momentum.",
    "humorous_tech": "A sample subject executes the action pipeline with no visible rollback plan.",
    "humorous_non_tech": "A sample subject gets things moving, and honestly, that is more than some Mondays manage.",
}

STYLE_DESCRIPTIONS = {
    "formal": "Professional, objective, factual tone. No jokes, slang, sarcasm, or embellishment.",
    "sarcastic": "Dry, ironic, lightly mocking tone while staying true to the observed video.",
    "humorous_tech": "Funny with technology or programming references, but still grounded in the observations.",
    "humorous_non_tech": "Funny everyday humor for a general audience, with no technical jargon.",
}


class CaptionPipeline:
    def __init__(
        self,
        settings: Settings,
        work_dir: Path,
        dry_run: bool = False,
        max_frames: int = 10,
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
            max_retries=settings.max_retries,
        )

    def process(self, asset: VideoAsset, styles: list[str] | None = None) -> dict[str, Any]:
        selected_styles = [style for style in (styles or list(STYLE_PROMPTS)) if style in STYLE_PROMPTS]
        frame_dir = self.work_dir / asset.video_id
        frames = [] if self.dry_run else extract_frames(asset.path, frame_dir, self.max_frames)
        transcript = self._read_transcript(asset)
        observations = self._observe(asset, frames, transcript)
        captions = self._captions(selected_styles, observations)
        checks = {}
        if self.run_checks:
            checks = self._run_checks_concurrent(selected_styles, observations, captions)

        return {
            "video_id": asset.video_id,
            "source_path": str(asset.path),
            "frames": [str(frame) for frame in frames],
            "frame_count": len(frames),
            "sampling_strategy": frames[0].parent.name if frames else "none",
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
                    f"Analyze these {len(frames)} sampled frames in chronological order."
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
            max_tokens=self.settings.max_tokens,
            temperature=self.settings.temperature,
            reasoning_effort=self.settings.reasoning_effort,
            json_mode=True,
        )
        return self._parse_or_repair_json(response, "perception observations", OBSERVATION_SCHEMA)

    def _captions(self, styles: list[str], observations: dict[str, Any]) -> dict[str, str]:
        if self.dry_run:
            return {style: DRY_RUN_CAPTIONS[style] for style in styles}

        requested = {style: STYLE_DESCRIPTIONS[style] for style in styles}
        prompt_payload = {
            "task": "Write one video caption or summary for each requested style.",
            "rules": [
                "Use only facts supported by the observations.",
                "Do not add events, objects, speech, motives, identities, or hidden context.",
                "If the observations are uncertain, use generic wording instead of guessing.",
                "Each caption must be in English and 1-2 sentences.",
                "Return strict JSON only, with exactly the requested style keys.",
            ],
            "requested_styles": requested,
            "observations": observations,
            "output_shape": {style: "caption text" for style in styles},
        }

        assert self.client is not None
        caption_schema = json.dumps({style: "caption text" for style in styles}, indent=2)
        response = self.client.chat(
            self.settings.model,
            [
                {
                    "role": "system",
                    "content": (
                        "You write concise, faithful video captions in multiple styles. "
                        "Return only a valid JSON object."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt_payload, indent=2)},
            ],
            max_tokens=self.settings.max_tokens,
            temperature=self.settings.temperature,
            reasoning_effort=self.settings.reasoning_effort,
            json_mode=True,
        )
        parsed = self._parse_or_repair_json(response, "caption JSON", caption_schema)
        captions: dict[str, str] = {}
        missing: list[str] = []
        for style in styles:
            caption = parsed.get(style)
            if isinstance(caption, str) and caption.strip():
                captions[style] = caption.strip().strip('"')
            else:
                missing.append(style)

        if missing:
            raise ValueError(f"Model response missed requested styles: {', '.join(missing)}")
        return captions

    def _caption(self, style: str, observations: dict[str, Any]) -> str:
        if self.dry_run:
            return DRY_RUN_CAPTIONS[style]

        assert self.client is not None
        prompt = load_prompt(STYLE_PROMPTS[style])
        temp = (
            self.settings.creative_temperature
            if style in CREATIVE_STYLES
            else self.settings.temperature
        )
        response = self.client.chat(
            self.settings.model,
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(observations, indent=2)},
            ],
            max_tokens=self.settings.caption_max_tokens,
            temperature=temp,
            reasoning_effort=self.settings.reasoning_effort,
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
            max_tokens=self.settings.check_max_tokens,
            temperature=self.settings.temperature,
            reasoning_effort=self.settings.reasoning_effort,
            json_mode=True,
        )
        parsed = parse_json_object(response)
        return {
            "accuracy": str(parsed.get("accuracy", "fail")),
            "tone": str(parsed.get("tone", "fail")),
            "notes": str(parsed.get("notes", "")),
        }

    def _run_checks_concurrent(
        self,
        styles: list[str],
        observations: dict[str, Any],
        captions: dict[str, str],
    ) -> dict[str, dict[str, str]]:
        """Run quality checks for all styles concurrently using a thread pool."""
        if self.dry_run:
            return {
                style: {"accuracy": "unknown", "tone": "unknown", "notes": "Dry run skipped model judging."}
                for style in styles
            }

        checks: dict[str, dict[str, str]] = {}
        workers = min(len(captions), 4)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_style = {
                pool.submit(self._check, style, observations, caption): style
                for style, caption in captions.items()
            }
            for future in as_completed(future_to_style):
                style = future_to_style[future]
                checks[style] = future.result()
        return checks

    def _parse_or_repair_json(
        self, response: str, label: str, expected_schema: str | None = None,
    ) -> dict[str, Any]:
        try:
            return parse_json_object(response)
        except Exception:
            assert self.client is not None
            schema_hint = ""
            if expected_schema:
                schema_hint = f"\n\nExpected JSON schema:\n{expected_schema}"
            repaired = self.client.chat(
                self.settings.model,
                [
                    {
                        "role": "system",
                        "content": (
                            "Repair the user's text into strict valid JSON only. "
                            "Do not add commentary, markdown, or new facts."
                            + schema_hint
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Repair this {label} JSON:\n\n{response}",
                    },
                ],
                max_tokens=self.settings.max_tokens,
                temperature=0.0,
                json_mode=True,
            )
            return parse_json_object(repaired)

    @staticmethod
    def _read_transcript(asset: VideoAsset) -> str:
        if asset.transcript_path is None:
            return ""
        return asset.transcript_path.read_text(encoding="utf-8").strip()
