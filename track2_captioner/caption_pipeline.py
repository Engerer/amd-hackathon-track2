from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future, TimeoutError
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

STYLE_CAPTION_SYSTEM = (
    "You are a single-style caption candidate generator.\n"
    "Treat the verified evidence ledger as the sole source of truth.\n"
    "Return JSON with key 'candidates' containing a list of English sentences.\n"
    "Figurative language may change framing, but cannot introduce new subjects, actions, settings, objects, or intentions."
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
    """Streamlined 3-Call Video Captioning Pipeline targeting Kimi K2.6.

    Flow:
      1. Deliberate 5-frame selection with perceptual hash deduplication.
      2. Call 1 (Multimodal): Extract factual evidence ledger with frame references from 5 pristine frames.
      3. Call 2 (Parallel): Generate 2-3 candidates per requested style in parallel.
      4. Call 3 (Multimodal Selector): Single multimodal call evaluating candidates against the 5 pristine frames and selecting best captions.
    """

    def __init__(
        self,
        settings: Settings,
        work_dir: Path,
        dry_run: bool = False,
        max_frames: int = 5,
        run_checks: bool = True,
        enable_style_retry: bool = True,
    ) -> None:
        self.settings = settings
        self.work_dir = work_dir
        self.dry_run = dry_run
        self.max_frames = 5  # Strict cap of 5 frames
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

        # --- Stage 1: Deliberate pristine frame extraction ---
        frame_started_at = time.monotonic()
        frames: list[Path] = []
        video_duration = None if self.dry_run else probe_duration_seconds(asset.path)

        if not should_fallback:
            # We strictly request 5 frames
            frames = [] if self.dry_run else extract_frames(
                asset.path,
                frame_dir,
                5,
                frame_profile=frame_profile,
            )
        timings["frame_extraction_sec"] = time.monotonic() - frame_started_at

        # Check deadline before starting Call 1
        remaining_time = caption_fallback_deadline - time.monotonic()
        if remaining_time < 1.5:
            should_fallback = True

        # --- Stage 2: 3-Call Core Architecture ---
        caption_started_at = time.monotonic()
        if should_fallback or not frames:
            evidence = dict(EMPTY_EVIDENCE)
            captions = {
                style: self._fallback_caption(style, evidence)
                for style in requested_styles
            }
        else:
            # Call 1: Multimodal Evidence Extraction
            try:
                evidence = self._extract_evidence(
                    asset=asset,
                    keyframes=frames,
                    video_duration=video_duration,
                    clip_deadline=caption_fallback_deadline,
                )
            except Exception as exc:
                logger.error(f"Call 1 (Evidence Extraction) failed: {exc}")
                evidence = dict(EMPTY_EVIDENCE)

            timings["evidence_extraction_sec"] = time.monotonic() - caption_started_at

            # Call 2 & 3: Parallel Candidate Generation & Multimodal Selection
            remaining_time = caption_fallback_deadline - time.monotonic()
            if remaining_time < 1.5:
                captions = {
                    style: self._fallback_caption(style, evidence)
                    for style in requested_styles
                }
            else:
                do_retry = self.enable_style_retry if enable_style_retry is None else enable_style_retry
                captions = self._generate_and_select_captions(
                    evidence,
                    frames,
                    requested_styles,
                    do_retry,
                    caption_fallback_deadline,
                )
            
            timings["caption_generation_sec"] = time.monotonic() - (caption_started_at + timings.get("evidence_extraction_sec", 0.0))

        timings["total_caption_sec"] = time.monotonic() - caption_started_at

        # --- Stage 4: Continuous Quality checks ---
        checks = {}
        if self.run_checks and not should_fallback:
            check_started_at = time.monotonic()
            checks = self._run_checks_concurrent(requested_styles, evidence, captions)
            timings["quality_check_sec"] = time.monotonic() - check_started_at
        
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
            "sampling_strategy": "deliberate_5_frame",
            "evidence": evidence,
            "observations": evidence,
            "captions": captions,
            "checks": checks,
            "timings": timings,
        }

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

        # Compile motion/native-video metadata locally
        motion_metadata = {
            "allocation_strategy": "Deliberate 5-Frame Allocation (beginning, strongest_scene_change, middle, strongest_action, end)",
            "timeline_events": [
                {"frame_index": 1, "type": "Beginning Timeline Anchor", "description": "Establishes baseline setting and initial state."},
                {"frame_index": 2, "type": "Strongest Scene Change Spike", "description": "Highest frame difference / transition index in video."},
                {"frame_index": 3, "type": "Middle Timeline Anchor", "description": "Tracks progression / mid-point state of action."},
                {"frame_index": 4, "type": "Strongest Action/Detail Peak", "description": "Highest motion delta / visual detail frame."},
                {"frame_index": 5, "type": "End Timeline Anchor", "description": "Tracks final state and resolution of action."}
            ]
        }

        content: list[dict[str, Any]] = []
        frame_timestamps = []
        for index, frame in enumerate(keyframes):
            timestamp = self._frame_timestamp(frame, index, len(keyframes), video_duration)
            frame_timestamps.append(round(timestamp, 3))

        # Direct video url grounding support
        is_native_video = any(x in self.settings.model.lower() for x in ["omni", "video", "gemini"])

        request = {
            "task": "Extract a structured evidence ledger from these 5 frames. Group observations with explicit frame references (e.g. Frame 1, Frame 3).",
            "video_duration_seconds": round(video_duration, 3) if video_duration else None,
            "frame_timestamps_seconds": frame_timestamps,
            "motion_timeline_evidence": motion_metadata,
            "native_video_grounding_url": str(asset.path) if is_native_video else None,
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
        )
        
        parsed = self._parse_or_repair_json(response, "evidence ledger", EVIDENCE_LEDGER_SCHEMA)
        
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

        # Call 2: Generate candidates concurrently
        style_candidates: dict[str, list[str]] = {style: [] for style in styles}
        futures: dict[Future, str] = {}

        max_workers = min(len(styles), 4)
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="candidates") as pool:
            for style in styles:
                future = pool.submit(
                    self._generate_candidates_for_style,
                    evidence, style, num_candidates,
                )
                futures[future] = style

            try:
                cand_timeout = min(remaining_time - 3.0, getattr(self.settings, "stage_deadline_candidates", 10.0))
                cand_timeout = max(1.0, cand_timeout)
                for future in as_completed(futures, timeout=cand_timeout):
                    style = futures[future]
                    try:
                        candidates = future.result()
                        if candidates:
                            style_candidates[style] = candidates
                    except Exception as e:
                        logger.error(f"Candidate generation failed for {style}: {e}")
            except TimeoutError:
                # Cancel pending futures immediately on timeout
                for f in futures:
                    f.cancel()
                logger.warning("Candidate generation timed out; cancelling pending tasks.")

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
                    evidence, keyframes, style_candidates, remaining_time,
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
                captions = self._repair_captions(evidence, styles, captions, issues)

        return captions

    def _generate_candidates_for_style(
        self,
        evidence: dict[str, Any],
        style: str,
        num_candidates: int,
    ) -> list[str]:
        """Call 2: Generate 2-3 candidates for one style in a single schema-constrained call."""
        assert self.client is not None

        style_instruction = load_prompt(STYLE_PROMPTS[style]) if style in STYLE_PROMPTS else ""
        
        request = {
            "target_style": style,
            "evidence_ledger": evidence,
            "style_instructions": style_instruction,
            "rules": [
                "Use ONLY facts from the evidence ledger. Do not add unsupported details.",
                f"Generate exactly {num_candidates} distinct candidate captions in the target style.",
                "Each caption must be one concise English sentence, ideally 12 to 30 words.",
                "Figurative language may change framing but CANNOT introduce new subjects or actions.",
                "Return JSON with key 'candidates' containing the list of captions.",
            ]
        }

        # Schema to return exactly num_candidates items
        schema = {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": num_candidates,
                    "maxItems": num_candidates
                }
            },
            "required": ["candidates"]
        }

        response = self.client.chat(
            self.settings.caption_model,
            [
                {"role": "system", "content": STYLE_CAPTION_SYSTEM},
                {"role": "user", "content": json.dumps(request, indent=2)},
            ],
            max_tokens=self.settings.caption_max_tokens,
            temperature=self.settings.creative_temperature if style != "formal" else 0.1,
            json_schema=schema,
        )
        parsed = parse_json_object(response)
        return parsed.get("candidates", [])

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

    def _check(self, style: str, evidence: dict[str, Any], caption: str) -> dict[str, Any]:
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
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_style = {
                pool.submit(self._check, style, evidence, caption): style
                for style, caption in captions.items()
            }
            for future in as_completed(future_to_style):
                style = future_to_style[future]
                try:
                    checks[style] = future.result()
                except Exception as exc:
                    checks[style] = {
                        "accuracy": "fail", "tone": "fail",
                        "notes": f"Check error: {exc}",
                        "factual_accuracy": 0.0, "style_strength": 0.0, "overall_score": 0.0,
                    }
        return checks

    def _repair_captions(
        self,
        evidence: dict[str, Any],
        styles: list[str],
        current_captions: dict[str, str],
        issues: list[str],
    ) -> dict[str, str]:
        """Repair captions that fail rules or style checks."""
        repaired = dict(current_captions)
        for issue in issues:
            parts = issue.split(":", 1)
            style = parts[0].strip()
            instruction = parts[1].strip() if len(parts) > 1 else "make caption correct and style strong"
            if style in current_captions:
                result = self._repair_single_caption(evidence, style, current_captions[style], instruction)
                if result:
                    repaired[style] = result
        return self._sanitize_caption_map(styles, repaired, evidence)

    def _repair_single_caption(
        self,
        evidence: dict[str, Any],
        style: str,
        caption: str,
        repair_instructions: str,
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
        self, response: str, label: str, expected_schema: str | None = None,
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
                json_mode=True,
            )
            return parse_json_object(repaired)
