from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from pathlib import Path
from typing import Any

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

# ---------------------------------------------------------------------------
# JSON schemas for Fireworks/Kimi schema-constrained generation
# ---------------------------------------------------------------------------

EVIDENCE_LEDGER_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "setting": {"type": "string"},
        "subjects": {
            "type": "array",
            "items": {"type": "string"},
        },
        "subject_counts": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
        "objects": {
            "type": "array",
            "items": {"type": "string"},
        },
        "actions": {
            "type": "array",
            "items": {"type": "string"},
        },
        "ocr": {
            "type": "array",
            "items": {"type": "string"},
        },
        "camera_motion": {"type": "string"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "text": {"type": "string"},
                    "timestamps": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "confidence": {"type": "number"},
                    "source": {"type": "string"},
                },
                "required": ["type", "text", "confidence"],
            },
        },
        "uncertainties": {
            "type": "array",
            "items": {"type": "string"},
        },
        "contradictions": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "summary",
        "setting",
        "subjects",
        "objects",
        "actions",
        "claims",
        "uncertainties",
    ],
}

# Used to generate 2-3 candidates in one call per style
CANDIDATE_GENERATION_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["candidates"]
}

# Dynamic multi-style selector schema
SELECTOR_ALL_STYLES_SCHEMA = {
    "type": "object",
    "properties": {
        "selected_captions": {
            "type": "object",
            "properties": {
                "formal": {"type": "string"},
                "sarcastic": {"type": "string"},
                "humorous_tech": {"type": "string"},
                "humorous_non_tech": {"type": "string"},
            }
        }
    },
    "required": ["selected_captions"]
}

# Quality check evaluation schema
CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "factual_accuracy": {"type": "number"},
        "subject_action_coverage": {"type": "number"},
        "unsupported_claims": {
            "type": "array",
            "items": {"type": "string"},
        },
        "omissions": {
            "type": "array",
            "items": {"type": "string"},
        },
        "style_strength": {"type": "number"},
        "naturalness": {"type": "number"},
        "concision": {"type": "number"},
        "overall_score": {"type": "number"},
        "repair_instructions": {"type": "string"},
    },
    "required": ["factual_accuracy", "style_strength", "overall_score"],
}

# Legacy schemas kept for backward compatibility in JSON repair
OBSERVATION_SCHEMA_STR = json.dumps(EVIDENCE_LEDGER_SCHEMA, indent=2)
CAPTION_SCHEMA_STR = json.dumps(CANDIDATE_GENERATION_SCHEMA, indent=2)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

CANDIDATE_BATCH_SYSTEM = (
    "You generate caption candidates for one or more requested video-caption styles.\n"
    "Treat the verified evidence ledger as the sole source of truth.\n"
    "Return strict JSON containing candidates_by_style.\n"
    "Figurative language may change framing, but cannot introduce new subjects, actions, settings, objects, or intentions."
)

STYLE_CAPTION_SYSTEM = (
    "You repair one video caption using only the verified evidence ledger. "
    "Return strict JSON with key 'caption' containing one English sentence."
)

SELECTOR_ALL_STYLES_SYSTEM = (
    "You are a grounded multimodal caption selector.\n\n"
    "You will receive:\n"
    "1. Exactly 5 pristine keyframes from a video.\n"
    "2. A factual evidence ledger describing what was observed in the video.\n"
    "3. A dictionary of candidate captions generated for multiple target styles.\n\n"
    "Your job is to look at the frames, read the evidence ledger, evaluate the candidates "
    "for each style, and select the best caption for each style.\n\n"
    "Rules:\n"
    "- Only select captions that are factually accurate according to the frames and evidence. "
    "If a candidate introduces unsupported subjects, actions, or objects, reject it.\n"
    "- Pick the candidate that has the strongest style match (formal, sarcastic, humorous_tech, humorous_non_tech).\n"
    "- Return strict JSON with a single key 'selected_captions' containing the chosen caption for each style."
)

DIRECT_CAPTION_SYSTEM = (
    "You are an accuracy-first video captioner. Infer the video only from four sparse, "
    "chronologically ordered frames and their timestamps. Write a short literal grounding "
    "clause naming the visible subject, clearest visible action, and setting. Every caption "
    "must explicitly restate that same grounding; a joke may follow it but may never replace "
    "it. Sparse frames do not prove continuous action between timestamps. Never invent or "
    "infer identity, intent, emotion, causality, dialogue, goals, exact quantities, off-screen "
    "events, or details too small to verify. Mention a change or sequence only when multiple "
    "ordered frames clearly support it. Each caption is one concise English sentence. formal "
    "is objective and professional. sarcastic states the grounded fact with a dry, lightly "
    "mocking aside about the visible situation, never a fictional motive. humorous_tech states "
    "the grounded fact and adds a natural programming or technology analogy. "
    "humorous_non_tech states the grounded fact and adds everyday humor with no technology "
    "jargon. Figurative language cannot add a new factual claim. Return only the "
    "schema-conforming JSON object."
)

# ---------------------------------------------------------------------------
# Style word lists for validation checks
# ---------------------------------------------------------------------------

HUMOR_NON_TECH_WORDS = {
    "actually", "apparently", "basically", "classic", "except",
    "feels", "guess", "honestly", "like", "manage",
    "meanwhile", "seems", "somehow", "standard", "trying", "typical",
}
NON_TECH_FORBIDDEN_WORDS = {
    "api", "bug", "cache", "code", "debug", "deploy", "latency",
    "log", "pipeline", "prompt", "queue", "rollback", "runtime",
    "scheduler", "server",
}
TECH_STYLE_WORDS = {
    "api", "bug", "cache", "commit", "debug", "deploy", "latency",
    "log", "pipeline", "queue", "rollback", "runtime", "scheduler",
}
SARCASM_STYLE_MARKERS = {
    "apparently", "because", "clearly", "naturally", "of course",
    "obviously", "serious", "thrilling",
}

EMPTY_EVIDENCE = {
    "summary": "No visual evidence was extracted.",
    "setting": "unknown",
    "subjects": [],
    "subject_counts": {},
    "objects": [],
    "actions": [],
    "ocr": [],
    "camera_motion": "unknown",
    "claims": [],
    "uncertainties": ["No perception pass was run."],
    "contradictions": [],
}

DRY_RUN_CAPTIONS = {
    "formal": "A sample subject performs a sample action in a simple sequence.",
    "sarcastic": "A sample subject performs a sample action, because apparently the plot needed momentum.",
    "humorous_tech": "A sample subject executes the action pipeline with no visible rollback plan.",
    "humorous_non_tech": "A sample subject gets things moving, and honestly, that is more than some Mondays manage.",
}


class CaptionPipeline:
    """Single-call video captioning pipeline targeting Kimi K2.6.

    Flow:
      1. Select four timeline anchors, replacing one interior anchor with a salient
         motion frame when that adds useful evidence.
      2. Make one schema-constrained multimodal call that grounds and writes all
         requested styles directly from the four pristine frames.
    """

    def __init__(
        self,
        settings: Settings,
        work_dir: Path,
        dry_run: bool = False,
        max_frames: int = 4,
        run_checks: bool = True,
        enable_style_retry: bool = True,
    ) -> None:
        self.settings = settings
        self.work_dir = work_dir
        self.dry_run = dry_run
        self.max_frames = 4
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
    ) -> dict[str, Any]:
        started_at = time.monotonic()
        timings: dict[str, float] = {}
        requested_styles = styles if styles else list(STYLE_PROMPTS)

        frame_dir = self.work_dir / asset.video_id

        # Enforce absolute clip deadline
        if caption_fallback_deadline is None:
            caption_fallback_deadline = started_at + 28.0

        should_fallback = force_fallback_captions or (time.monotonic() >= caption_fallback_deadline)

        # --- Stage 1: Four-frame visual coverage ---
        frame_started_at = time.monotonic()
        frames: list[Path] = []
        video_duration = None if self.dry_run else probe_duration_seconds(asset.path)

        if not should_fallback:
            # Four images performed best in leaderboard observations and leave
            # Kimi enough visual context without diluting attention.
            frames = [] if self.dry_run else extract_frames(
                asset.path,
                frame_dir,
                4,
                frame_profile=frame_profile,
            )
        timings["frame_extraction_sec"] = time.monotonic() - frame_started_at

        # Check deadline before starting Call 1
        remaining_time = caption_fallback_deadline - time.monotonic()
        if remaining_time < 1.5:
            should_fallback = True

        # --- Stage 2: one direct multimodal caption call ---
        caption_started_at = time.monotonic()
        if should_fallback or not frames:
            evidence = dict(EMPTY_EVIDENCE)
            captions = {
                style: self._fallback_caption(style, evidence)
                for style in requested_styles
            }
        else:
            evidence = {
                "summary": "Captions generated directly from four ordered video frames.",
                "setting": "",
                "subjects": [],
                "objects": [],
                "actions": [],
                "claims": [],
                "uncertainties": [],
            }
            try:
                captions = self._caption_four_frames_direct(
                    asset=asset,
                    keyframes=frames,
                    video_duration=video_duration,
                    styles=requested_styles,
                    clip_deadline=caption_fallback_deadline,
                )
            except Exception as exc:
                logger.error("Direct multimodal caption call failed: %s", exc)
                evidence = dict(EMPTY_EVIDENCE)
                captions = {
                    style: self._fallback_caption(style, evidence)
                    for style in requested_styles
                }
            timings["direct_caption_sec"] = time.monotonic() - caption_started_at
            timings["evidence_extraction_sec"] = 0.0
            timings["caption_generation_sec"] = timings["direct_caption_sec"]

        timings["total_caption_sec"] = time.monotonic() - caption_started_at

        # Extra judge/repair calls are deliberately disabled in single-call mode.
        checks = {}

        timings["total_process_sec"] = time.monotonic() - started_at

        return {
            "video_id": asset.video_id,
            "source_path": str(asset.path),
            "frames": [str(frame) for frame in frames],
            "frame_count": len(frames),
            "ocr_frame_count": 0,
            "crop_frame_count": 0,
            "motion_frame_count": 0,
            "duration_seconds": video_duration,
            "sampling_strategy": "direct_4_frame_single_call",
            "evidence": evidence,
            "observations": evidence,
            "captions": captions,
            "checks": checks,
            "timings": timings,
        }

    def _caption_four_frames_direct(
        self,
        asset: VideoAsset,
        keyframes: list[Path],
        video_duration: float | None,
        styles: list[str],
        clip_deadline: float,
    ) -> dict[str, str]:
        """Generate all requested styles in exactly one multimodal model call."""
        assert self.client is not None
        if len(keyframes) != 4:
            raise ValueError(f"Direct captioning requires exactly four frames, got {len(keyframes)}.")

        timestamps = [
            round(self._frame_timestamp(frame, index, 4, video_duration), 3)
            for index, frame in enumerate(keyframes)
        ]
        style_descriptions = {
            "formal": "professional, objective, factual",
            "sarcastic": "dry, ironic, lightly mocking",
            "humorous_tech": "funny with a natural technology or programming reference",
            "humorous_non_tech": "funny everyday humor with no technical jargon",
        }
        content: list[dict[str, Any]] = [{
            "type": "text",
            "text": json.dumps({
                "task": "Caption this video directly from the four chronological frames.",
                "video_duration_seconds": round(video_duration, 3) if video_duration else None,
                "frame_timestamps_seconds": timestamps,
                "requested_styles": {
                    style: style_descriptions[style]
                    for style in styles
                },
                "output_rules": [
                    "one sentence per requested style",
                    "12-30 words per caption when practical",
                    "begin every caption with a literal subject-action-setting clause",
                    "keep the same verified subject, setting, and action across styles",
                    "put any joke after the grounded visual fact",
                    "do not state exact counts, motives, emotions, or unseen outcomes",
                ],
            }, separators=(",", ":")),
        }]
        for index, frame in enumerate(keyframes):
            content.append({
                "type": "text",
                "text": f"Frame {index + 1}/4 at {self._format_timestamp(timestamps[index])}",
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": image_to_data_url(frame)},
            })

        caption_properties = {style: {"type": "string"} for style in styles}
        schema = {
            "type": "object",
            "properties": {
                "grounding": {"type": "string"},
                **caption_properties,
            },
            "required": ["grounding", *styles],
            "additionalProperties": False,
        }
        remaining = clip_deadline - time.monotonic()
        timeout = min(
            max(0.5, remaining - 0.25),
            getattr(self.settings, "stage_deadline_direct", 23.0),
        )
        response = self.client.chat(
            self.settings.model,
            [
                {"role": "system", "content": DIRECT_CAPTION_SYSTEM},
                {"role": "user", "content": content},
            ],
            max_tokens=min(self.settings.caption_max_tokens, 500),
            temperature=self.settings.temperature,
            reasoning_effort=self.settings.reasoning_effort,
            json_schema=schema,
            timeout_seconds=timeout,
        )
        parsed = parse_json_object(response)
        # Some OpenAI-compatible proxies preserve the constrained fields but
        # flatten the single object wrapper. Accept both shapes locally without
        # spending a second model call on JSON repair.
        raw_captions = (
            parsed.get("captions")
            or parsed.get("selected_captions")
            or parsed.get("captions_by_style")
        )
        if not isinstance(raw_captions, dict) and all(style in parsed for style in styles):
            raw_captions = parsed
        if not isinstance(raw_captions, dict):
            raise ValueError(
                "Direct caption response did not contain requested styles; "
                f"received keys={sorted(parsed)}."
            )
        return self._sanitize_caption_map(styles, raw_captions, evidence={})

    # ======================================================================
    # Call 1: Multimodal Evidence Extraction
    # ======================================================================

    def _extract_evidence(
        self,
        asset: VideoAsset,
        keyframes: list[Path],
        video_duration: float | None,
        clip_deadline: float,
    ) -> dict[str, Any]:
        """Extract a structured evidence ledger with frame references from 5 pristine frames."""
        assert self.client is not None

        # Describe the actual attached frames without assigning semantic roles
        # that may be invalid after the selected frames are sorted chronologically.
        motion_metadata = {
            "allocation_strategy": (
                "Three mandatory temporal anchors plus up to two visually diverse "
                "salient frames, presented below in chronological order."
            ),
            "timeline_events": [
                {
                    "frame_index": index + 1,
                    "timestamp_seconds": round(
                        self._frame_timestamp(frame, index, len(keyframes), video_duration),
                        3,
                    ),
                    "type": "chronological_selected_frame",
                }
                for index, frame in enumerate(keyframes)
            ]
        }

        content: list[dict[str, Any]] = []
        frame_timestamps = []
        for index, frame in enumerate(keyframes):
            timestamp = self._frame_timestamp(frame, index, len(keyframes), video_duration)
            frame_timestamps.append(round(timestamp, 3))

        request = {
            "task": "Extract a structured evidence ledger from these 5 frames. Group observations with explicit frame references (e.g. Frame 1, Frame 3).",
            "video_duration_seconds": round(video_duration, 3) if video_duration else None,
            "frame_timestamps_seconds": frame_timestamps,
            "motion_timeline_evidence": motion_metadata,
        }
        content.append({"type": "text", "text": json.dumps(request, indent=2)})

        for index, frame in enumerate(keyframes):
            ts = frame_timestamps[index]
            content.append({
                "type": "text",
                "text": f"Frame {index + 1}/5 at {self._format_timestamp(ts)}.",
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": image_to_data_url(frame)},
            })

        remaining_time = clip_deadline - time.monotonic()
        timeout = min(remaining_time - 2.0, getattr(self.settings, "stage_deadline_perception", 12.0))
        timeout = max(1.0, timeout)

        response = self.client.chat(
            self.settings.model,
            [
                {"role": "system", "content": load_prompt("perception_system.txt")},
                {"role": "user", "content": content},
            ],
            max_tokens=self.settings.max_tokens,
            temperature=self.settings.temperature,
            reasoning_effort=self.settings.reasoning_effort,
            json_schema=EVIDENCE_LEDGER_SCHEMA,
            timeout_seconds=timeout,
        )

        parsed = self._parse_or_repair_json(
            response,
            "evidence ledger",
            EVIDENCE_LEDGER_SCHEMA,
            timeout_seconds=max(0.5, clip_deadline - time.monotonic() - 0.25),
        )

        # Semantic consensus & threshold check: filter low-confidence claims
        if "claims" in parsed and isinstance(parsed["claims"], list):
            valid_claims = []
            for claim in parsed["claims"]:
                if not isinstance(claim, dict):
                    continue
                conf = claim.get("confidence", 1.0)
                # Filter out low-confidence unsupported claims
                if conf >= 0.70:
                    valid_claims.append(claim)
            parsed["claims"] = valid_claims

        return parsed

    # ======================================================================
    # Call 2 & 3: Parallel Candidate Generation & Multimodal Selection
    # ======================================================================

    def _generate_and_select_captions(
        self,
        evidence: dict[str, Any],
        keyframes: list[Path],
        styles: list[str],
        enable_retry: bool,
        clip_deadline: float,
    ) -> dict[str, str]:
        if self.dry_run:
            return {style: DRY_RUN_CAPTIONS.get(style, "") for style in styles}
        assert self.client is not None

        remaining_time = clip_deadline - time.monotonic()

        # Adaptive candidate count scaling based on remaining time
        if remaining_time > 15.0:
            num_candidates = 3
        elif remaining_time >= 8.0:
            num_candidates = 2
        else:
            num_candidates = 1  # Low latency mode: skip selector entirely
        num_candidates = min(
            num_candidates,
            max(1, min(getattr(self.settings, "candidate_count", 3), 3)),
        )

        # Call 2: Generate candidates for every requested style in one text call.
        cand_timeout = min(
            remaining_time - 3.0,
            getattr(self.settings, "stage_deadline_candidates", 10.0),
        )
        cand_timeout = max(1.0, cand_timeout)
        try:
            style_candidates = self._generate_candidates_all_styles(
                evidence,
                styles,
                num_candidates,
                timeout_seconds=cand_timeout,
            )
        except Exception as exc:
            logger.error("Call 2 (candidate generation) failed: %s", exc)
            style_candidates = {style: [] for style in styles}

        # Enforce fallbacks for missing candidates
        for style in styles:
            if not style_candidates[style]:
                style_candidates[style] = [self._fallback_caption(style, evidence)]

        # Call 3: Grounded Multimodal Selector
        remaining_time = clip_deadline - time.monotonic()
        if num_candidates == 1 or remaining_time < 3.0:
            # Skip selector if only 1 candidate or low on time
            captions = {style: candidates[0] for style, candidates in style_candidates.items()}
        else:
            try:
                captions = self._select_best_candidates_multimodal(
                    evidence,
                    keyframes,
                    style_candidates,
                    remaining_time,
                )
            except Exception as e:
                logger.error(f"Call 3 (Multimodal Selector) failed: {e}")
                # Fallback to the first candidate of each style
                captions = {style: candidates[0] for style, candidates in style_candidates.items()}

        # Clean/sanitize output captions
        captions = self._sanitize_caption_map(styles, captions, evidence)

        # Style-specific repair pass if verified style rules fail
        if enable_retry and (clip_deadline - time.monotonic() > 2.0):
            issues = self._caption_issues(styles, captions)
            if issues:
                captions = self._repair_captions(
                    evidence,
                    styles,
                    captions,
                    issues,
                    deadline=clip_deadline,
                )

        return captions

    def _generate_candidates_all_styles(
        self,
        evidence: dict[str, Any],
        styles: list[str],
        num_candidates: int,
        timeout_seconds: float,
    ) -> dict[str, list[str]]:
        """Call 2: generate candidates for all requested styles in one call."""
        assert self.client is not None

        style_instructions = {
            style: load_prompt(STYLE_PROMPTS[style])
            for style in styles
            if style in STYLE_PROMPTS
        }

        request = {
            "requested_styles": styles,
            "evidence_ledger": evidence,
            "style_instructions": style_instructions,
            "rules": [
                "Use ONLY facts from the evidence ledger. Do not add unsupported details.",
                f"Generate exactly {num_candidates} distinct candidate captions for each requested style.",
                "Each caption must be one concise English sentence, ideally 12 to 30 words.",
                "Figurative language may change framing but CANNOT introduce new subjects or actions.",
                "Return JSON with key 'candidates_by_style'.",
            ],
        }

        style_properties = {
            style: {"type": "array", "items": {"type": "string"}}
            for style in styles
        }
        schema = {
            "type": "object",
            "properties": {
                "candidates_by_style": {
                    "type": "object",
                    "properties": style_properties,
                    "required": styles,
                    "additionalProperties": False,
                },
            },
            "required": ["candidates_by_style"],
            "additionalProperties": False,
        }

        response = self.client.chat(
            self.settings.caption_model,
            [
                {"role": "system", "content": CANDIDATE_BATCH_SYSTEM},
                {"role": "user", "content": json.dumps(request, indent=2)},
            ],
            max_tokens=self.settings.caption_max_tokens,
            temperature=self.settings.creative_temperature,
            json_schema=schema,
            timeout_seconds=timeout_seconds,
        )
        parsed = parse_json_object(response)
        payload = parsed.get("candidates_by_style", {})
        result: dict[str, list[str]] = {}
        for style in styles:
            raw_candidates = payload.get(style, []) if isinstance(payload, dict) else []
            if not isinstance(raw_candidates, list):
                raw_candidates = []
            cleaned = [
                self._clean_caption(str(candidate))
                for candidate in raw_candidates
                if self._clean_caption(str(candidate))
            ]
            result[style] = cleaned[:num_candidates]
        return result

    # ======================================================================
    # Call 3: Grounded Multimodal Selector
    # ======================================================================

    def _select_best_candidates_multimodal(
        self,
        evidence: dict[str, Any],
        keyframes: list[Path],
        style_candidates: dict[str, list[str]],
        remaining_time: float,
    ) -> dict[str, str]:
        """Call 3: Multimodal selector evaluates all candidate styles against the 5 pristine frames in a single call."""
        assert self.client is not None

        content: list[dict[str, Any]] = []
        request = {
            "task": "Evaluate candidate captions against the visual evidence and select the best one for each style.",
            "evidence_ledger": evidence,
            "candidates_by_style": style_candidates,
        }
        content.append({"type": "text", "text": json.dumps(request, indent=2)})

        for index, frame in enumerate(keyframes):
            content.append({
                "type": "text",
                "text": f"Pristine Frame {index + 1}/5.",
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": image_to_data_url(frame)},
            })

        selector_model = getattr(self.settings, "selector_model", "") or self.settings.caption_model
        timeout = min(remaining_time - 1.5, getattr(self.settings, "stage_deadline_selection", 6.0))
        timeout = max(1.0, timeout)

        response = self.client.chat(
            selector_model,
            [
                {"role": "system", "content": SELECTOR_ALL_STYLES_SYSTEM},
                {"role": "user", "content": content},
            ],
            max_tokens=self.settings.caption_max_tokens,
            temperature=0.1,
            json_schema=SELECTOR_ALL_STYLES_SCHEMA,
            timeout_seconds=timeout,
        )

        parsed = parse_json_object(response)
        selected = parsed.get("selected_captions", {})

        # In case the selector fails to pick all requested styles
        for style in style_candidates:
            if style not in selected:
                selected[style] = style_candidates[style][0]

        return selected

    # ======================================================================
    # Quality checks & Repairs
    # ======================================================================

    def _check(
        self,
        style: str,
        evidence: dict[str, Any],
        caption: str,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Run grounded quality check for a single caption."""
        if self.dry_run:
            return {
                "factual_accuracy": 0.0,
                "style_strength": 0.0,
                "overall_score": 0.0,
                "accuracy": "unknown",
                "tone": "unknown",
                "notes": "Dry run skipped model judging.",
            }

        assert self.client is not None
        try:
            response = self.client.chat(
                self.settings.judge_model,
                [
                    {"role": "system", "content": load_prompt("judge.txt")},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "target_style": style,
                                "evidence_ledger": evidence,
                                "caption": caption,
                            },
                            indent=2,
                        ),
                    },
                ],
                max_tokens=self.settings.check_max_tokens,
                temperature=self.settings.temperature,
                reasoning_effort=self.settings.reasoning_effort,
                json_schema=CHECK_SCHEMA,
                timeout_seconds=timeout_seconds,
            )
            parsed = parse_json_object(response)

            factual = float(parsed.get("factual_accuracy", 0.5))
            style_score = float(parsed.get("style_strength", 0.5))
            overall = float(parsed.get("overall_score", (factual + style_score) / 2))

            return {
                "factual_accuracy": factual,
                "subject_action_coverage": float(parsed.get("subject_action_coverage", 0.5)),
                "unsupported_claims": parsed.get("unsupported_claims", []),
                "omissions": parsed.get("omissions", []),
                "style_strength": style_score,
                "naturalness": float(parsed.get("naturalness", 0.5)),
                "concision": float(parsed.get("concision", 0.5)),
                "overall_score": overall,
                "repair_instructions": str(parsed.get("repair_instructions", "")),
                # Backward compatibility
                "accuracy": "pass" if factual >= 0.7 else "fail",
                "tone": "pass" if style_score >= 0.7 else "fail",
                "notes": str(parsed.get("repair_instructions", "")),
            }
        except Exception as exc:
            return {
                "factual_accuracy": 0.0,
                "style_strength": 0.0,
                "overall_score": 0.0,
                "accuracy": "fail",
                "tone": "fail",
                "notes": f"Check failed: {exc}",
            }

    def _run_checks_concurrent(
        self,
        styles: list[str],
        evidence: dict[str, Any],
        captions: dict[str, str],
        deadline: float | None = None,
    ) -> dict[str, dict[str, Any]]:
        if self.dry_run:
            return {
                style: {
                    "accuracy": "unknown", "tone": "unknown",
                    "notes": "Dry run skipped model judging.",
                    "factual_accuracy": 0.0, "style_strength": 0.0, "overall_score": 0.0,
                }
                for style in styles
            }

        checks: dict[str, dict[str, Any]] = {}
        workers = min(len(captions), 4)
        timeout = None if deadline is None else max(0.5, deadline - time.monotonic() - 0.25)
        pool = ThreadPoolExecutor(max_workers=workers)
        future_to_style: dict[Any, str] = {}
        try:
            future_to_style = {
                pool.submit(self._check, style, evidence, caption, timeout): style
                for style, caption in captions.items()
            }
            try:
                for future in as_completed(future_to_style, timeout=timeout):
                    style = future_to_style[future]
                    try:
                        checks[style] = future.result()
                    except Exception as exc:
                        checks[style] = {
                            "accuracy": "fail", "tone": "fail",
                            "notes": f"Check error: {exc}",
                            "factual_accuracy": 0.0, "style_strength": 0.0, "overall_score": 0.0,
                        }
            except TimeoutError:
                logger.warning("Quality checks exceeded their deadline.")
        finally:
            for future in future_to_style:
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
        return checks

    def _repair_captions(
        self,
        evidence: dict[str, Any],
        styles: list[str],
        current_captions: dict[str, str],
        issues: list[str],
        deadline: float,
    ) -> dict[str, str]:
        """Repair captions that fail rules or style checks."""
        repaired = dict(current_captions)
        issues_by_style: dict[str, list[str]] = {}
        for issue in issues:
            style, _, instruction = issue.partition(":")
            style = style.strip()
            if style in current_captions:
                issues_by_style.setdefault(style, []).append(
                    instruction.strip() or "make caption correct and style strong"
                )

        for style, style_issues in issues_by_style.items():
            remaining = deadline - time.monotonic()
            if remaining <= 0.75:
                break
            result = self._repair_single_caption(
                evidence,
                style,
                current_captions[style],
                "; ".join(style_issues),
                timeout_seconds=max(0.5, remaining - 0.25),
            )
            if result:
                repaired[style] = result
        return self._sanitize_caption_map(styles, repaired, evidence)

    def _repair_single_caption(
        self,
        evidence: dict[str, Any],
        style: str,
        caption: str,
        repair_instructions: str,
        timeout_seconds: float | None = None,
    ) -> str:
        """Call style model to repair a single caption with rules enforcement."""
        assert self.client is not None
        request = {
            "task": "Repair this caption following the repair instructions.",
            "target_style": style,
            "evidence_ledger": evidence,
            "current_caption": caption,
            "repair_instructions": repair_instructions,
            "rules": [
                "Keep all accurate facts from the current caption.",
                "Fix only what the repair instructions specify.",
                "Use ONLY facts from the evidence ledger.",
                "Return JSON with key 'caption'.",
            ],
        }

        try:
            response = self.client.chat(
                self.settings.caption_model,
                [
                    {"role": "system", "content": STYLE_CAPTION_SYSTEM},
                    {"role": "user", "content": json.dumps(request, indent=2)},
                ],
                max_tokens=self.settings.caption_max_tokens,
                temperature=0.15,
                json_schema={"type": "object", "properties": {"caption": {"type": "string"}}, "required": ["caption"]},
                timeout_seconds=timeout_seconds,
            )
            parsed = parse_json_object(response)
            repaired = str(parsed.get("caption", "")).strip()
            return self._clean_caption(repaired) if repaired else ""
        except Exception:
            return ""

    # ======================================================================
    # Utilities
    # ======================================================================

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

    def _sanitize_caption_map(
        self,
        styles: list[str],
        captions: dict[str, str],
        evidence: dict[str, Any],
    ) -> dict[str, str]:
        return {
            style: self._clean_caption(captions.get(style, "")) or self._fallback_caption(style, evidence)
            for style in styles
        }

    def _fallback_caption(self, style: str, evidence: dict[str, Any]) -> str:
        summary = str(evidence.get("summary") or evidence.get("setting") or "").strip()
        if summary.lower().startswith("no visual evidence"):
            summary = ""
        subjects = ", ".join(str(item) for item in evidence.get("subjects", [])[:2])
        actions = ", ".join(str(item) for item in evidence.get("actions", [])[:2])
        base = summary or f"The video shows {subjects or 'visible subjects'} with {actions or 'visible activity'}."
        if style == "formal":
            return base
        if style == "sarcastic":
            return f"{base} Clearly, ordinary visual evidence has never worked harder."
        if style == "humorous_tech":
            return f"{base} The scene ships its visual update with no rollback needed."
        return f"{base} It is doing its best to make everyday motion look eventful."

    def _parse_or_repair_json(
        self,
        response: str,
        label: str,
        expected_schema: dict[str, Any] | str | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        try:
            return parse_json_object(response)
        except Exception:
            if self.client is None:
                return {}
            schema_hint = ""
            if expected_schema:
                schema_hint = f"\n\nExpected JSON schema:\n{json.dumps(expected_schema, indent=2)}"
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
                json_mode=not isinstance(expected_schema, dict),
                json_schema=expected_schema if isinstance(expected_schema, dict) else None,
                timeout_seconds=timeout_seconds,
            )
            return parse_json_object(repaired)
