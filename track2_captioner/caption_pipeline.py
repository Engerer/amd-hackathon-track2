from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from track2_captioner.config import Settings
from track2_captioner.fireworks_client import FireworksClient, image_to_data_url
from track2_captioner.json_tools import parse_json_object
from track2_captioner.prompts import STYLE_PROMPTS, load_prompt
from track2_captioner.video_ingest import DEFAULT_FRAME_PROFILE, DEFAULT_MAX_FRAMES, VideoAsset, extract_frames


logger = logging.getLogger(__name__)

CAPTION_SCHEMA = json.dumps({
    "captions": {
        "formal": "formal caption when requested",
        "sarcastic": "sarcastic caption when requested",
        "humorous_tech": "humorous technology caption when requested",
        "humorous_non_tech": "humorous non-technical caption when requested",
    },
    "visual_facts": ["brief factual details used for grounding"],
}, indent=2)

CHECK_SCHEMA = json.dumps({
    "accuracy": "pass or fail",
    "tone": "pass or fail",
    "notes": "brief explanation",
}, indent=2)

DIRECT_CAPTION_SYSTEM = (
    "You are a direct multimodal video captioning agent. "
    "Inspect the sampled video frames in chronological order and generate final captions directly. "
    "Do not write an intermediate observation report. Return strict JSON only."
)

HUMOR_NON_TECH_WORDS = {
    "actually",
    "apparently",
    "basically",
    "classic",
    "except",
    "feels",
    "guess",
    "honestly",
    "like",
    "manage",
    "meanwhile",
    "seems",
    "somehow",
    "standard",
    "trying",
    "typical",
}
NON_TECH_FORBIDDEN_WORDS = {
    "api",
    "bug",
    "cache",
    "code",
    "debug",
    "deploy",
    "latency",
    "log",
    "pipeline",
    "prompt",
    "queue",
    "rollback",
    "runtime",
    "scheduler",
    "server",
}
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
    "summary": "Direct multimodal captioning did not run.",
    "setting": "unknown",
    "subjects": [],
    "key_objects": [],
    "actions": [],
    "timeline": [],
    "visible_text": [],
    "audio_or_speech": [],
    "uncertainties": ["Dry run or fallback did not inspect the video."],
}

DRY_RUN_CAPTIONS = {
    "formal": "A sample subject performs a sample action in a simple sequence.",
    "sarcastic": "A sample subject performs a sample action, because apparently the plot needed momentum.",
    "humorous_tech": "A sample subject executes the action pipeline with no visible rollback plan.",
    "humorous_non_tech": "A sample subject gets things moving, and honestly, that is more than some Mondays manage.",
}

STYLE_DESCRIPTIONS = {
    "formal": "Professional, objective, factual tone. No jokes, slang, sarcasm, or embellishment.",
    "sarcastic": "Dry, ironic, lightly mocking tone while staying true to the visible video.",
    "humorous_tech": "Funny with technology or programming references, while staying grounded in the frames.",
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
        enable_style_retry: bool = True,
    ) -> None:
        self.settings = settings
        self.work_dir = work_dir
        self.dry_run = dry_run
        self.max_frames = max_frames
        self.run_checks = run_checks
        self.enable_style_retry = enable_style_retry
        self.client = None if dry_run else FireworksClient(
            settings.api_key,
            settings.base_url,
            settings.proxy_url,
            settings.proxy_token,
            max_retries=settings.max_retries,
        )

    def process(
        self,
        asset: VideoAsset,
        styles: list[str] | None = None,
        max_frames: int | None = None,
        frame_profile: str = DEFAULT_FRAME_PROFILE,
        enable_style_retry: bool | None = None,
        force_fallback_captions: bool = False,
        caption_fallback_deadline: float | None = None,
    ) -> dict[str, Any]:
        started_at = time.monotonic()
        timings: dict[str, float] = {}
        selected_styles = [style for style in (styles or list(STYLE_PROMPTS)) if style in STYLE_PROMPTS]

        frame_dir = self.work_dir / asset.video_id
        frame_started_at = time.monotonic()
        frames = [] if self.dry_run else extract_frames(
            asset.path,
            frame_dir,
            max_frames or self.max_frames,
            frame_profile=frame_profile,
        )
        timings["frame_extraction_sec"] = time.monotonic() - frame_started_at

        transcript = self._read_transcript(asset)
        caption_started_at = time.monotonic()
        if force_fallback_captions or (
            caption_fallback_deadline is not None
            and time.monotonic() >= caption_fallback_deadline
        ):
            observations = dict(EMPTY_OBSERVATIONS)
            captions = {
                style: self._fallback_caption(style, observations)
                for style in selected_styles
            }
        else:
            captions, observations = self._direct_captions(
                asset=asset,
                frames=frames,
                transcript=transcript,
                styles=selected_styles,
                enable_style_retry=(
                    self.enable_style_retry
                    if enable_style_retry is None
                    else enable_style_retry
                ),
            )
        timings["vision_observation_sec"] = 0.0
        timings["caption_generation_sec"] = time.monotonic() - caption_started_at

        checks = {}
        if self.run_checks:
            check_started_at = time.monotonic()
            checks = self._run_checks_concurrent(selected_styles, observations, captions)
            timings["quality_check_sec"] = time.monotonic() - check_started_at
        timings["total_process_sec"] = time.monotonic() - started_at

        return {
            "video_id": asset.video_id,
            "source_path": str(asset.path),
            "frames": [str(frame) for frame in frames],
            "frame_count": len(frames),
            "sampling_strategy": frames[0].parent.name if frames else "none",
            "observations": observations,
            "captions": captions,
            "checks": checks,
            "timings": timings,
        }

    def _direct_captions(
        self,
        asset: VideoAsset,
        frames: list[Path],
        transcript: str,
        styles: list[str],
        enable_style_retry: bool,
    ) -> tuple[dict[str, str], dict[str, Any]]:
        if self.dry_run:
            return {style: DRY_RUN_CAPTIONS[style] for style in styles}, dict(EMPTY_OBSERVATIONS)

        assert self.client is not None
        content = self._direct_caption_content(asset, frames, transcript, styles)
        response = self.client.chat(
            self.settings.model,
            [
                {"role": "system", "content": DIRECT_CAPTION_SYSTEM},
                {"role": "user", "content": content},
            ],
            max_tokens=self.settings.caption_max_tokens,
            temperature=self.settings.creative_temperature,
            reasoning_effort=self.settings.reasoning_effort,
            json_mode=True,
        )
        parsed = self._parse_or_repair_json(response, "direct multimodal captions", CAPTION_SCHEMA)
        observations = self._observations_from_direct_response(parsed)
        raw_captions = self._extract_caption_map(styles, parsed)
        captions = self._sanitize_caption_map(styles, raw_captions, observations)

        if enable_style_retry:
            issues = self._caption_issues(styles, raw_captions)
            if issues:
                captions = self._retry_direct_captions(asset, frames, transcript, styles, captions, issues)
                observations = self._observations_from_direct_response({
                    "captions": captions,
                    "visual_facts": observations.get("key_objects", []),
                })

        return captions, observations

    def _direct_caption_content(
        self,
        asset: VideoAsset,
        frames: list[Path],
        transcript: str,
        styles: list[str],
    ) -> list[dict[str, Any]]:
        style_requirements = {
            style: STYLE_DESCRIPTIONS[style]
            for style in styles
            if style in STYLE_DESCRIPTIONS
        }
        request = {
            "video_id": asset.video_id,
            "task": "Generate final captions directly from the provided frames.",
            "requested_styles": style_requirements,
            "optional_transcript": transcript or "[none provided]",
            "rules": [
                "Use the frames as the primary source of truth.",
                "Write every caption in English.",
                "Treat frames as chronological samples from one video.",
                "Mention the main subject, setting, and primary action when visible.",
                "Use only visible or transcript-backed facts; do not invent motives, identities, locations, speech, or hidden context.",
                "Never turn uncertainty into a concrete claim.",
                "Each caption should be one concise sentence, ideally 12 to 30 words.",
                "For humorous_tech, include a clear tech reference such as API, bug, debug, deploy, latency, log, cache, pipeline, queue, rollback, runtime, or scheduler.",
                "For humorous_non_tech, avoid all tech, programming, AI, prompt, model, server, and software jargon.",
                "Return captions only for the requested styles.",
                "Return strict JSON matching the schema.",
            ],
            "response_schema": CAPTION_SCHEMA,
        }
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": json.dumps(request, indent=2),
            }
        ]
        for frame in frames:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(frame)},
                }
            )
        return content

    def _retry_direct_captions(
        self,
        asset: VideoAsset,
        frames: list[Path],
        transcript: str,
        styles: list[str],
        current_captions: dict[str, str],
        issues: list[str],
    ) -> dict[str, str]:
        assert self.client is not None
        repair_request = {
            "video_id": asset.video_id,
            "task": "Repair the direct video captions using the frames again.",
            "issues": issues,
            "current_captions": current_captions,
            "requested_styles": {
                style: STYLE_DESCRIPTIONS[style]
                for style in styles
                if style in STYLE_DESCRIPTIONS
            },
            "optional_transcript": transcript or "[none provided]",
            "rules": [
                "Keep only visible or transcript-backed facts.",
                "Write every caption in English.",
                "Fix only missing, weakly styled, overlong, or jargon-violating captions.",
                "Return one concise caption for every requested style.",
                "Return strict JSON matching the schema.",
            ],
            "response_schema": CAPTION_SCHEMA,
        }
        content: list[dict[str, Any]] = [{"type": "text", "text": json.dumps(repair_request, indent=2)}]
        for frame in frames:
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(frame)}})

        response = self.client.chat(
            self.settings.model,
            [
                {"role": "system", "content": DIRECT_CAPTION_SYSTEM},
                {"role": "user", "content": content},
            ],
            max_tokens=self.settings.caption_max_tokens,
            temperature=max(0.2, self.settings.creative_temperature - 0.2),
            reasoning_effort=self.settings.reasoning_effort,
            json_mode=True,
        )
        parsed = self._parse_or_repair_json(response, "direct caption repair", CAPTION_SCHEMA)
        repaired = self._extract_caption_map(styles, parsed)
        merged = dict(current_captions)
        merged.update({style: caption for style, caption in repaired.items() if caption})
        return self._sanitize_caption_map(styles, merged, self._observations_from_direct_response(parsed))

    @staticmethod
    def _extract_caption_map(styles: list[str], parsed: dict[str, Any]) -> dict[str, str]:
        payload = parsed.get("captions", parsed)
        if not isinstance(payload, dict):
            return {}

        captions: dict[str, str] = {}
        for style in styles:
            value = payload.get(style)
            if isinstance(value, dict):
                value = value.get("caption") or value.get("text")
            caption = CaptionPipeline._clean_caption(str(value or ""))
            if caption:
                captions[style] = caption
        return captions

    def _sanitize_caption_map(
        self,
        styles: list[str],
        captions: dict[str, str],
        observations: dict[str, Any],
    ) -> dict[str, str]:
        return {
            style: self._clean_caption(captions.get(style, "")) or self._fallback_caption(style, observations)
            for style in styles
        }

    @staticmethod
    def _observations_from_direct_response(parsed: dict[str, Any]) -> dict[str, Any]:
        facts = parsed.get("visual_facts") or parsed.get("facts") or []
        if isinstance(facts, str):
            facts = [facts]
        if not isinstance(facts, list):
            facts = []
        summary = str(parsed.get("summary") or "Captions were generated directly from sampled frames.").strip()
        return {
            "summary": summary,
            "setting": str(parsed.get("setting") or "inferred from sampled frames").strip(),
            "subjects": [str(item) for item in parsed.get("subjects", [])] if isinstance(parsed.get("subjects"), list) else [],
            "key_objects": [str(item) for item in facts[:8]],
            "actions": [str(item) for item in parsed.get("actions", [])] if isinstance(parsed.get("actions"), list) else [],
            "timeline": [str(item) for item in parsed.get("timeline", [])] if isinstance(parsed.get("timeline"), list) else [],
            "visible_text": [str(item) for item in parsed.get("visible_text", [])] if isinstance(parsed.get("visible_text"), list) else [],
            "audio_or_speech": [str(item) for item in parsed.get("audio_or_speech", [])] if isinstance(parsed.get("audio_or_speech"), list) else [],
            "uncertainties": ["Direct mode uses one multimodal caption call instead of a separate observation pass."],
        }

    @staticmethod
    def _clean_caption(response: str) -> str:
        cleaned = response.strip().strip('"').strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r"\s+([,.!?])", r"\1", cleaned)
        return cleaned

    @staticmethod
    def _caption_issues(styles: list[str], captions: dict[str, str]) -> list[str]:
        issues: list[str] = []
        for style in styles:
            caption = captions.get(style, "")
            if not caption:
                issues.append(f"{style}: missing caption")
                continue
            if CaptionPipeline._needs_style_retry(style, caption):
                issues.append(f"{style}: style too weak")
            if CaptionPipeline._needs_length_retry(caption):
                issues.append(f"{style}: too long")
            if style == "humorous_non_tech" and CaptionPipeline._contains_non_tech_jargon(caption):
                issues.append(f"{style}: contains tech jargon")
        return issues

    @staticmethod
    def _needs_style_retry(style: str, caption: str) -> bool:
        normalized = caption.lower()
        if style == "humorous_tech":
            return not any(word in normalized for word in TECH_STYLE_WORDS)
        if style == "sarcastic":
            return not any(marker in normalized for marker in SARCASM_STYLE_MARKERS)
        if style == "humorous_non_tech":
            return not any(word in normalized for word in HUMOR_NON_TECH_WORDS)
        return False

    @staticmethod
    def _contains_non_tech_jargon(caption: str) -> bool:
        normalized = caption.lower()
        return any(re.search(rf"\b{re.escape(word)}\b", normalized) for word in NON_TECH_FORBIDDEN_WORDS)

    @staticmethod
    def _needs_length_retry(caption: str) -> bool:
        words = re.findall(r"\b[\w'-]+\b", caption)
        sentence_count = len(re.findall(r"[.!?]+", caption))
        return len(words) > 36 or sentence_count > 2

    def _fallback_caption(self, style: str, observations: dict[str, Any]) -> str:
        summary = str(observations.get("summary") or observations.get("setting") or "").strip()
        subjects = ", ".join(str(item) for item in observations.get("subjects", [])[:2])
        actions = ", ".join(str(item) for item in observations.get("actions", [])[:2])
        base = summary or f"The video shows {subjects or 'visible subjects'} with {actions or 'visible activity'}."
        if style == "formal":
            return base
        if style == "sarcastic":
            return f"{base} Clearly, ordinary visual evidence has never worked harder."
        if style == "humorous_tech":
            return f"{base} The scene ships its visual update with no rollback needed."
        return f"{base} It is doing its best to make everyday motion look eventful."

    def _check(self, style: str, observations: dict[str, Any], caption: str) -> dict[str, str]:
        if self.dry_run:
            return {"accuracy": "unknown", "tone": "unknown", "notes": "Dry run skipped model judging."}

        assert self.client is not None
        response = self.client.chat(
            self.settings.model,
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
                max_tokens=self.settings.caption_max_tokens,
                temperature=0.0,
                json_mode=True,
            )
            return parse_json_object(repaired)

    @staticmethod
    def _read_transcript(asset: VideoAsset) -> str:
        if asset.transcript_path is None:
            return ""
        return asset.transcript_path.read_text(encoding="utf-8").strip()
