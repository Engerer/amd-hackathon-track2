from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from track2_captioner.config import Settings
from track2_captioner.fireworks_client import FireworksClient, image_to_data_url
from track2_captioner.json_tools import parse_json_object
from track2_captioner.prompts import STYLE_PROMPTS, load_prompt
from track2_captioner.video_ingest import DEFAULT_MAX_FRAMES, FrameSample, VideoAsset, extract_frame_samples


logger = logging.getLogger(__name__)

OBSERVATION_SCHEMA = json.dumps({
    "summary": "one factual overview sentence",
    "setting": "where the video appears to take place",
    "subjects": ["visible subject or object"],
    "key_objects": ["important visible objects, colors, signs, or environmental details"],
    "actions": ["important visible action"],
    "timeline": ["beginning: ...", "middle: ...", "end: ..."],
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
TECH_STYLE_WORDS = {
    "api",
    "bug",
    "cache",
    "commit",
    "debug",
    "deploy",
    "latency",
    "log",
    "pipeline",
    "queue",
    "rollback",
    "runtime",
    "scheduler",
}
SARCASM_STYLE_MARKERS = {
    "apparently",
    "because",
    "clearly",
    "naturally",
    "of course",
    "obviously",
    "serious",
    "thrilling",
}


EMPTY_OBSERVATIONS = {
    "summary": "Dry run placeholder summary.",
    "setting": "Dry run placeholder setting.",
    "subjects": ["sample subject"],
    "key_objects": ["sample object"],
    "actions": ["sample action"],
    "timeline": ["beginning: sample start", "middle: sample middle", "end: sample ending"],
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
        max_frames: int = DEFAULT_MAX_FRAMES,
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
        frame_samples = [] if self.dry_run else extract_frame_samples(asset.path, frame_dir, self.max_frames)
        transcript = self._read_transcript(asset)
        observations = self._observe(asset, frame_samples, transcript)
        captions = self._captions(selected_styles, observations)
        checks = {}
        if self.run_checks:
            checks = self._run_checks_concurrent(selected_styles, observations, captions)

        frames = [sample.path for sample in frame_samples]
        sources = sorted({sample.source for sample in frame_samples})
        return {
            "video_id": asset.video_id,
            "source_path": str(asset.path),
            "frames": [str(frame) for frame in frames],
            "frame_count": len(frames),
            "sampling_strategy": "+".join(sources) if sources else "none",
            "frame_samples": [
                {
                    "path": str(sample.path),
                    "timestamp_seconds": sample.timestamp_seconds,
                    "source": sample.source,
                }
                for sample in frame_samples
            ],
            "observations": observations,
            "captions": captions,
            "checks": checks,
        }

    def _observe(self, asset: VideoAsset, frames: list[FrameSample], transcript: str) -> dict[str, Any]:
        if self.dry_run:
            return dict(EMPTY_OBSERVATIONS)

        frame_manifest = "\n".join(
            f"{index}. {self._format_frame_time(frame.timestamp_seconds)} ({frame.source})"
            for index, frame in enumerate(frames, start=1)
        )
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"Video id: {asset.video_id}\n"
                    f"Optional transcript:\n{transcript or '[none provided]'}\n\n"
                    f"Analyze these {len(frames)} sampled frames in chronological order. "
                    f"Frame metadata:\n{frame_manifest or '[none]'}\n\n"
                    "Capture exact visible facts that would help a judge compare captions: "
                    "setting, subjects, colors, countable objects, actions, scene changes, "
                    "visible text, camera movement, and transcript-backed speech. "
                    "Use the frame metadata only to describe temporal order or approximate timing."
                ),
            }
        ]
        for frame in frames:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(frame.path)},
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
        observations = self._parse_or_repair_json(response, "perception observations", OBSERVATION_SCHEMA)
        return self._sanitize_observations(observations)

    def _captions(self, styles: list[str], observations: dict[str, Any]) -> dict[str, str]:
        if self.dry_run:
            return {style: DRY_RUN_CAPTIONS[style] for style in styles}

        captions: dict[str, str] = {}
        workers = min(len(styles), 4)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_style = {
                pool.submit(self._caption, style, observations): style
                for style in styles
            }
            for future in as_completed(future_to_style):
                style = future_to_style[future]
                try:
                    captions[style] = future.result()
                except Exception:
                    logger.warning("Caption generation failed for style %s.", style, exc_info=True)
                    captions[style] = self._fallback_caption(style, observations)
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
        caption_request = {
            "target_style": style,
            "style_requirement": STYLE_DESCRIPTIONS.get(style, ""),
            "strict_grounding_rules": [
                "Use only summary, setting, subjects, key_objects, actions, timeline, visible_text, and audio_or_speech as factual evidence.",
                "Never turn anything in uncertainties into a fact.",
                "If the exact location, identity, motive, or text is uncertain, use generic wording instead of guessing.",
                "Mention the main subject, setting, and primary action when supported.",
                "Return only the final caption text.",
            ],
            "observations": observations,
        }
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(caption_request, indent=2)},
        ]
        response = self._caption_chat(messages, temp)
        caption = response.strip().strip('"')
        if self._needs_style_retry(style, caption):
            retry_messages = messages + [
                {
                    "role": "user",
                    "content": (
                        "Rewrite the caption. It was too plain for the requested style. "
                        "Keep the same observed facts, but make the target style obvious. "
                        "Return only the rewritten caption text."
                    ),
                }
            ]
            response = self._caption_chat(retry_messages, temp)
            caption = response.strip().strip('"')
        return caption

    def _caption_chat(self, messages: list[dict[str, Any]], temperature: float) -> str:
        assert self.client is not None
        try:
            return self.client.chat(
                self.settings.caption_model,
                messages,
                max_tokens=self.settings.caption_max_tokens,
                temperature=temperature,
                reasoning_effort=self.settings.reasoning_effort,
            )
        except Exception:
            if self.settings.caption_model == self.settings.model:
                raise
            logger.warning(
                "Caption model %s failed; falling back to %s.",
                self.settings.caption_model,
                self.settings.model,
                exc_info=True,
            )
            return self.client.chat(
                self.settings.model,
                messages,
                max_tokens=self.settings.caption_max_tokens,
                temperature=temperature,
                reasoning_effort=self.settings.reasoning_effort,
            )

    @staticmethod
    def _needs_style_retry(style: str, caption: str) -> bool:
        normalized = caption.lower()
        if style == "humorous_tech":
            return not any(word in normalized for word in TECH_STYLE_WORDS)
        if style == "sarcastic":
            return not any(marker in normalized for marker in SARCASM_STYLE_MARKERS)
        return False

    @staticmethod
    def _sanitize_observations(observations: dict[str, Any]) -> dict[str, Any]:
        cleaned = dict(observations)
        uncertainties = cleaned.get("uncertainties")
        if isinstance(uncertainties, list):
            cleaned["uncertainties"] = [
                CaptionPipeline._sanitize_uncertainty(str(item))
                for item in uncertainties
            ]
        return cleaned

    @staticmethod
    def _sanitize_uncertainty(text: str) -> str:
        lowered = text.lower()
        if "exact city" in lowered or "exact location" in lowered:
            if any(marker in lowered for marker in ("suggest", "may be", "might be", "probably", "looks like")):
                return re.split(r"\bthough\b|\bbut\b|;|,", text, maxsplit=1, flags=re.IGNORECASE)[0].strip() + "."
        return text

    def _fallback_caption(self, style: str, observations: dict[str, Any]) -> str:
        summary = str(observations.get("summary") or "").strip()
        setting = str(observations.get("setting") or "").strip()
        subjects = ", ".join(str(item) for item in observations.get("subjects", [])[:2])
        actions = ", ".join(str(item) for item in observations.get("actions", [])[:2])
        details = ", ".join(str(item) for item in observations.get("key_objects", [])[:2])
        base = summary or f"The video shows {subjects or 'visible subjects'} with {actions or 'visible activity'}."
        if setting and setting.lower() not in base.lower():
            base = f"{base.rstrip('.')} in {setting}."
        if details and details.lower() not in base.lower():
            base = f"{base.rstrip('.')} with {details}."
        if style == "formal":
            return base
        if style == "sarcastic":
            return f"{base} Clearly, ordinary visual evidence has never worked harder."
        if style == "humorous_tech":
            return f"{base} The scene ships its visual update with no rollback needed."
        return f"{base} It is doing its best to make everyday motion look eventful."

    @staticmethod
    def _format_frame_time(timestamp_seconds: float | None) -> str:
        if timestamp_seconds is None:
            return "time unknown"
        minutes = int(timestamp_seconds // 60)
        seconds = timestamp_seconds - (minutes * 60)
        if minutes:
            return f"about {minutes}m {seconds:.1f}s"
        return f"about {seconds:.1f}s"

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
