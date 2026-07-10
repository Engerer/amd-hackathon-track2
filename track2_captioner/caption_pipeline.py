from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from track2_captioner.config import Settings
from track2_captioner.fireworks_client import FireworksClient, image_to_data_url
from track2_captioner.json_tools import parse_json_object
from track2_captioner.prompts import STYLE_PROMPTS, load_prompt
from track2_captioner.video_ingest import (
    DEFAULT_FRAME_PROFILE,
    DEFAULT_MAX_FRAMES,
    VideoAsset,
    extract_frames,
    probe_duration_seconds,
)


logger = logging.getLogger(__name__)

CAPTION_SCHEMA = json.dumps({
    "core_facts": {
        "subject": "main visible subject, using generic wording if uncertain",
        "action": "main visible action or state",
        "setting": "visible setting without guessing a specific location",
        "important_objects": ["objects that matter to the caption"],
    },
    "description": "1-2 sentence neutral description of the sampled video evidence",
    "visible_text": ["readable text visible in the frames, or []"],
    "actions": ["specific visible actions, or []"],
    "objects": ["important visible objects, or []"],
    "timeline": ["brief chronological notes from the sampled frames"],
    "visual_facts": ["brief factual details used for grounding"],
    "uncertainties": ["details that should not become concrete caption claims"],
    "captions": {
        "formal": "formal caption when requested",
        "sarcastic": "sarcastic caption when requested",
        "humorous_tech": "humorous technology caption when requested",
        "humorous_non_tech": "humorous non-technical caption when requested",
    },
}, indent=2)

CHECK_SCHEMA = json.dumps({
    "accuracy": "pass or fail",
    "tone": "pass or fail",
    "notes": "brief explanation",
}, indent=2)

DIRECT_CAPTION_SYSTEM = (
    "You are a direct multimodal video captioning agent. "
    "Inspect the sampled video frames in chronological order and generate final captions directly. "
    "Establish one shared factual core before writing any styled captions. Return strict JSON only."
)
TIMESTAMP_BANNER_HEIGHT = 64
TIMESTAMP_FONT_SIZE = 32

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
    "humorous_tech": (
        "Funny through one consistent technology metaphor, such as APIs, debugging, game engines, "
        "networking, robotics, queues, deploys, or runtime behavior. Do not mix unrelated metaphors."
    ),
    "humorous_non_tech": "Funny everyday humor for a general audience, with no technical jargon.",
}

STYLE_TEMPLATES = {
    "formal": "State the shared factual core plainly and objectively.",
    "sarcastic": "State the shared factual core, then add a short dry ironic aside about only the visible action.",
    "humorous_tech": "State the shared factual core through one concise technology metaphor without adding events.",
    "humorous_non_tech": "State the shared factual core with one everyday joke or comparison and no technical language.",
}

CAPTION_STYLE_ALIASES = {
    "formal": {"formal", "professional", "objective"},
    "sarcastic": {"sarcastic", "sarcasm", "dryhumor", "dryhumour", "ironic"},
    "humorous_tech": {
        "humoroustech",
        "humoroustechnical",
        "humoroustechnology",
        "techhumor",
        "techhumour",
        "technicalhumor",
        "technicalhumour",
        "technologyhumor",
        "technologyhumour",
        "funnytech",
        "tech",
    },
    "humorous_non_tech": {
        "humorousnontech",
        "humorousnontechnical",
        "humorousnontechnology",
        "nontechhumor",
        "nontechhumour",
        "nontechnicalhumor",
        "nontechnicalhumour",
        "everydayhumor",
        "everydayhumour",
        "generalhumor",
        "generalhumour",
        "funny",
    },
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
            request_timeout_seconds=settings.request_timeout_seconds,
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
        audio_context: str = "",
    ) -> dict[str, Any]:
        started_at = time.monotonic()
        timings: dict[str, float] = {}
        selected_styles = list(STYLE_PROMPTS)

        frame_dir = self.work_dir / asset.video_id
        should_fallback = force_fallback_captions or (
            caption_fallback_deadline is not None
            and time.monotonic() >= caption_fallback_deadline
        )
        frame_started_at = time.monotonic()
        frames: list[Path] = []
        model_images: list[Path] = []
        video_duration = None if self.dry_run else probe_duration_seconds(asset.path)
        if not should_fallback:
            frames = [] if self.dry_run else extract_frames(
                asset.path,
                frame_dir,
                max_frames or self.max_frames,
                frame_profile=frame_profile,
            )
            model_images = self._prepare_model_images(frames, frame_dir, video_duration)
        timings["frame_extraction_sec"] = time.monotonic() - frame_started_at

        transcript = self._read_transcript(asset)
        caption_started_at = time.monotonic()
        if should_fallback:
            observations = dict(EMPTY_OBSERVATIONS)
            captions = {
                style: self._fallback_caption(style, observations)
                for style in selected_styles
            }
        else:
            captions, observations = self._direct_captions(
                asset=asset,
                model_images=model_images,
                source_frame_count=len(frames),
                video_duration=video_duration,
                transcript=transcript,
                audio_context=audio_context,
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
            "model_images": [str(image) for image in model_images],
            "model_image_count": len(model_images),
            "duration_seconds": video_duration,
            "sampling_strategy": frames[0].parent.name if frames else "none",
            "observations": observations,
            "captions": captions,
            "checks": checks,
            "timings": timings,
        }

    def _prepare_model_images(
        self,
        frames: list[Path],
        frame_dir: Path,
        video_duration: float | None,
    ) -> list[Path]:
        if not frames:
            return []
        if len(frames) != 5:
            raise ValueError(f"Expected exactly 5 frames, received {len(frames)}.")

        output_dir = frame_dir / "timestamped"
        output_dir.mkdir(parents=True, exist_ok=True)
        prepared: list[Path] = []
        for index, frame_path in enumerate(frames):
            timestamp = self._frame_timestamp(frame_path, index, len(frames), video_duration)
            output_path = output_dir / f"frame_{index + 1:03d}_t{timestamp:09.3f}.jpg"
            self._add_timestamp_banner(
                frame_path,
                output_path,
                index=index,
                frame_count=len(frames),
                timestamp=timestamp,
                video_duration=video_duration,
            )
            prepared.append(output_path)
        return prepared

    @staticmethod
    def _frame_timestamp(
        frame_path: Path,
        index: int,
        frame_count: int,
        video_duration: float | None,
    ) -> float:
        match = re.search(r"_t(\d+(?:\.\d+)?)", frame_path.stem)
        if match:
            return float(match.group(1))
        if not video_duration or video_duration <= 0:
            return float(index)
        if frame_count <= 1:
            return video_duration / 2
        start = min(0.5, max(video_duration * 0.05, 0.0))
        end = max(video_duration - 0.5, start)
        return start + ((end - start) * (index / (frame_count - 1)))

    @staticmethod
    def _format_timestamp(seconds: float | None) -> str:
        if seconds is None or seconds < 0:
            return "unknown"
        minutes = int(seconds // 60)
        remainder = seconds - (minutes * 60)
        return f"{minutes:02d}:{remainder:04.1f}"

    @staticmethod
    def _timestamp_font() -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", TIMESTAMP_FONT_SIZE)
        except OSError:
            return ImageFont.load_default()

    @classmethod
    def _add_timestamp_banner(
        cls,
        source_path: Path,
        output_path: Path,
        index: int,
        frame_count: int,
        timestamp: float,
        video_duration: float | None,
    ) -> None:
        with Image.open(source_path) as source:
            image = source.convert("RGB")
        canvas = image.copy()
        label = (
            f"Frame {index + 1}/{frame_count}  |  "
            f"{cls._format_timestamp(timestamp)} / total {cls._format_timestamp(video_duration)}"
        )
        draw = ImageDraw.Draw(canvas)
        draw.rectangle(
            (0, 0, canvas.width, min(TIMESTAMP_BANNER_HEIGHT, canvas.height)),
            fill=(20, 20, 20),
        )
        draw.text(
            (18, 12),
            label,
            fill=(255, 255, 255),
            font=cls._timestamp_font(),
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path, format="JPEG", quality=90, optimize=True)

    def _direct_captions(
        self,
        asset: VideoAsset,
        model_images: list[Path],
        source_frame_count: int,
        video_duration: float | None,
        transcript: str,
        audio_context: str,
        styles: list[str],
        enable_style_retry: bool,
    ) -> tuple[dict[str, str], dict[str, Any]]:
        if self.dry_run:
            return {style: DRY_RUN_CAPTIONS[style] for style in styles}, dict(EMPTY_OBSERVATIONS)

        assert self.client is not None
        content = self._direct_caption_content(
            asset,
            model_images,
            source_frame_count,
            video_duration,
            transcript,
            audio_context,
            styles,
        )
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
                captions = self._retry_direct_captions(
                    asset,
                    model_images,
                    source_frame_count,
                    video_duration,
                    transcript,
                    audio_context,
                    styles,
                    captions,
                    issues,
                )
                observations = self._observations_from_direct_response({
                    "captions": captions,
                    "visual_facts": observations.get("key_objects", []),
                })

        return captions, observations

    def _direct_caption_content(
        self,
        asset: VideoAsset,
        model_images: list[Path],
        source_frame_count: int,
        video_duration: float | None,
        transcript: str,
        audio_context: str,
        styles: list[str],
    ) -> list[dict[str, Any]]:
        style_requirements = {
            style: STYLE_DESCRIPTIONS[style]
            for style in styles
            if style in STYLE_DESCRIPTIONS
        }
        frame_timestamps = [
            round(self._frame_timestamp(frame, index, len(model_images), video_duration), 3)
            for index, frame in enumerate(model_images)
        ]
        visual_input = self._visual_input_description(
            model_images,
            source_frame_count,
            video_duration,
        )
        request = {
            "video_id": asset.video_id,
            "task": "Use the five timestamped frames to generate all four final caption styles in one response.",
            "visual_input": visual_input,
            "video_duration_seconds": round(video_duration, 3) if video_duration else None,
            "video_duration": self._format_timestamp(video_duration),
            "frame_timestamps_seconds": frame_timestamps,
            "requested_styles": style_requirements,
            "style_templates": {
                style: STYLE_TEMPLATES[style]
                for style in styles
                if style in STYLE_TEMPLATES
            },
            "optional_transcript": transcript or "[none provided]",
            "optional_audio_context": audio_context or "[none provided]",
            "rules": [
                "Use the frames as the primary source of truth.",
                "These are exactly five chronological frames sampled from one video, not five separate images or videos.",
                "Use each frame timestamp and the total video duration to reason about chronology and change over time.",
                "First establish one core_facts object containing the shared subject, action, setting, and important objects.",
                "Every styled caption must preserve the same core subject, action, and setting; style may change wording but not facts.",
                "Begin each caption with a recognizable factual clause before adding sarcasm or humor.",
                "A humorous or sarcastic clause must not introduce new visible nouns, actions, speech, motives, or locations.",
                "Also return description, visible_text, actions, objects, timeline, visual_facts, and uncertainties as compact grounding fields.",
                "For visible_text, list only readable text that is actually visible; use [] if no text is readable.",
                "If visible text is important, incorporate it naturally in captions as a human viewer would.",
                "Use transcript or audio context only as supporting evidence; do not invent exact speech, music, or sounds.",
                "Write every caption in English.",
                "Treat the five images as chronological samples from one video in Frame 1 through Frame 5 order.",
                "Mention the main subject, setting, and primary action when visible.",
                "Use only visible or transcript-backed facts; do not invent motives, identities, locations, speech, or hidden context.",
                "Never turn uncertainty into a concrete claim.",
                "Each caption should be one concise sentence, ideally 12 to 30 words.",
                "For humorous_tech, use one consistent technology metaphor and include a clear tech reference such as API, bug, debug, deploy, latency, log, cache, queue, rollback, runtime, or scheduler.",
                "For humorous_non_tech, avoid all tech, programming, AI, prompt, model, server, and software jargon.",
                "Before returning, verify each caption includes visible subject/action/setting when possible and matches its requested style.",
                "Never mention OCR, VLMs, AI models, prompts, frames, timestamps, storyboards, or video-analysis mechanics in the captions.",
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
        for index, frame in enumerate(model_images):
            timestamp = frame_timestamps[index]
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"Frame {index + 1}/{len(model_images)} at "
                        f"{self._format_timestamp(timestamp)} of total "
                        f"{self._format_timestamp(video_duration)}."
                    ),
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(frame)},
                }
            )
        return content

    @staticmethod
    def _visual_input_description(
        model_images: list[Path],
        source_frame_count: int,
        video_duration: float | None,
    ) -> str:
        return (
            f"Exactly {source_frame_count} images are attached simultaneously. They are five "
            f"timestamped frames sampled from one {CaptionPipeline._format_timestamp(video_duration)} "
            "video and are ordered chronologically from Frame 1 to Frame 5."
        )

    def _retry_direct_captions(
        self,
        asset: VideoAsset,
        model_images: list[Path],
        source_frame_count: int,
        video_duration: float | None,
        transcript: str,
        audio_context: str,
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
            "visual_input": self._visual_input_description(
                model_images,
                source_frame_count,
                video_duration,
            ),
            "video_duration_seconds": round(video_duration, 3) if video_duration else None,
            "requested_styles": {
                style: STYLE_DESCRIPTIONS[style]
                for style in styles
                if style in STYLE_DESCRIPTIONS
            },
            "optional_transcript": transcript or "[none provided]",
            "optional_audio_context": audio_context or "[none provided]",
            "rules": [
                "Keep only visible or transcript-backed facts.",
                "Preserve one identical factual subject, action, and setting across all styles.",
                "Put the factual clause first; style wording must not introduce new events or objects.",
                "Write every caption in English.",
                "Fix only missing, weakly styled, overlong, or jargon-violating captions.",
                "Use audio context only as supporting evidence, not as a source for invented details.",
                "If visible text is important, incorporate it naturally without saying OCR or analysis.",
                "For humorous_tech, keep one consistent technology metaphor.",
                "Never mention OCR, VLMs, AI models, prompts, frames, timestamps, storyboards, or video-analysis mechanics in the captions.",
                "Return one concise caption for every requested style.",
                "Return strict JSON matching the schema.",
            ],
            "response_schema": CAPTION_SCHEMA,
        }
        content: list[dict[str, Any]] = [{"type": "text", "text": json.dumps(repair_request, indent=2)}]
        for index, frame in enumerate(model_images):
            timestamp = self._frame_timestamp(frame, index, len(model_images), video_duration)
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"Frame {index + 1}/{len(model_images)} at "
                        f"{self._format_timestamp(timestamp)} of total "
                        f"{self._format_timestamp(video_duration)}."
                    ),
                }
            )
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

        normalized_payload: dict[str, Any] = {}
        for key, value in payload.items():
            normalized_key = CaptionPipeline._normalize_caption_key(str(key))
            if normalized_key:
                normalized_payload.setdefault(normalized_key, value)

        captions: dict[str, str] = {}
        for style in styles:
            value = payload.get(style)
            if value is None:
                for alias in CAPTION_STYLE_ALIASES.get(style, {style}):
                    value = normalized_payload.get(CaptionPipeline._normalize_caption_key(alias))
                    if value is not None:
                        break
            if isinstance(value, dict):
                value = value.get("caption") or value.get("text")
            caption = CaptionPipeline._clean_caption(str(value or ""))
            if caption:
                captions[style] = caption
        return captions

    @staticmethod
    def _normalize_caption_key(key: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", key.lower())

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
        core = parsed.get("core_facts")
        if not isinstance(core, dict):
            core = {}
        visible_text = (
            CaptionPipeline._string_list_from_keys(parsed, ["visible_text", "ocr", "text"])
            or CaptionPipeline._string_list_from_keys(core, ["visible_text", "text"])
        )
        actions = (
            CaptionPipeline._string_list_from_keys(parsed, ["actions", "visible_actions"])
            or CaptionPipeline._string_list_from_keys(core, ["action", "actions"])
        )
        subjects = (
            CaptionPipeline._string_list_from_keys(parsed, ["subjects"])
            or CaptionPipeline._string_list_from_keys(core, ["subject", "subjects"])
        )
        objects = (
            CaptionPipeline._string_list_from_keys(parsed, ["objects", "key_objects", "subjects"])
            or CaptionPipeline._string_list_from_keys(core, ["important_objects", "objects", "subject"])
        )
        facts = CaptionPipeline._string_list_from_keys(parsed, ["visual_facts", "facts"])
        if not facts:
            facts = [*objects[:3], *actions[:3], *visible_text[:2]]
        summary = str(
            parsed.get("description")
            or parsed.get("summary")
            or core.get("summary")
            or "Captions were generated directly from sampled frames."
        ).strip()
        return {
            "summary": summary,
            "setting": str(core.get("setting") or parsed.get("setting") or "inferred from sampled frames").strip(),
            "subjects": subjects,
            "key_objects": [str(item) for item in facts[:8]],
            "actions": actions,
            "timeline": CaptionPipeline._string_list_from_keys(parsed, ["timeline"]),
            "visible_text": visible_text,
            "audio_or_speech": CaptionPipeline._string_list_from_keys(parsed, ["audio_or_speech", "speech", "transcript"]),
            "uncertainties": (
                CaptionPipeline._string_list_from_keys(parsed, ["uncertainties", "uncertain"])
                or ["Direct mode uses one multimodal caption call instead of a separate observation pass."]
            ),
        }

    @staticmethod
    def _string_list_from_keys(payload: dict[str, Any], keys: list[str]) -> list[str]:
        for key in keys:
            value = payload.get(key)
            if value:
                if isinstance(value, list):
                    return [str(item).strip() for item in value if str(item).strip()]
                if isinstance(value, str):
                    cleaned = value.strip()
                    if cleaned:
                        return [cleaned]
        return []

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
