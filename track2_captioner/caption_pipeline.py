from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
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
# JSON schemas for Fireworks schema-constrained generation
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

CAPTION_SINGLE_SCHEMA = {
    "type": "object",
    "properties": {
        "caption": {"type": "string"},
    },
    "required": ["caption"],
}

SELECTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "best_index": {"type": "integer"},
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
    "required": ["best_index", "factual_accuracy", "style_strength", "overall_score"],
}

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
CAPTION_SCHEMA_STR = json.dumps(CAPTION_SINGLE_SCHEMA, indent=2)

# ---------------------------------------------------------------------------
# Style caption system prompt
# ---------------------------------------------------------------------------

STYLE_CAPTION_SYSTEM = (
    "You are a single-style caption generator for a video captioning pipeline. "
    "You receive a verified evidence ledger describing what was observed in a video. "
    "Generate exactly one caption in the requested style. "
    "Return strict JSON with a single key 'caption' containing one English sentence. "
    "Figurative language may change framing but CANNOT introduce a new subject, action, "
    "setting, object, or intention that is not in the evidence ledger."
)

SELECTOR_SYSTEM = (
    "You are a grounded caption selector. You receive an evidence ledger describing "
    "what was observed in a video, a list of candidate captions for a specific style, "
    "and the target style name. Score each candidate and select the best one. "
    "Return strict JSON with scoring fields."
)

# ---------------------------------------------------------------------------
# Word sets for style detection (retained from original for quick checks)
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

# ---------------------------------------------------------------------------
# Empty / fallback data
# ---------------------------------------------------------------------------

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

CAPTION_STYLE_ALIASES = {
    "formal": {"formal", "professional", "objective"},
    "sarcastic": {"sarcastic", "sarcasm", "dryhumor", "dryhumour", "ironic"},
    "humorous_tech": {
        "humoroustech", "humoroustechnical", "humoroustechnology",
        "techhumor", "techhumour", "technicalhumor", "technicalhumour",
        "technologyhumor", "technologyhumour", "funnytech", "tech",
    },
    "humorous_non_tech": {
        "humorousnontech", "humorousnontechnical", "humorousnontechnology",
        "nontechhumor", "nontechhumour", "nontechnicalhumor", "nontechnicalhumour",
        "everydayhumor", "everydayhumour", "generalhumor", "generalhumour", "funny",
    },
}

STYLE_LABELS = {
    "formal": "Formal",
    "sarcastic": "Sarcastic",
    "humorous_tech": "Humorous-tech",
    "humorous_non_tech": "Humorous non-tech",
}


class CaptionPipeline:
    """Accuracy-first video captioning pipeline.

    Architecture:
      1. Parallel complementary experts extract evidence from video frames,
         crops, OCR frames, and motion clips.
      2. Evidence is fused into a structured ledger with typed claims,
         uncertainties, and contradictions.
      3. Multiple candidate captions are generated per style in parallel,
         each receiving only one target style and the verified ledger.
      4. A grounded selector scores candidates on accuracy, coverage,
         style strength, naturalness, and concision, then selects the best.
      5. If the best candidate fails verification, targeted repair is applied.
    """

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

    # ======================================================================
    # Main entry point
    # ======================================================================

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
        selected_styles = list(STYLE_PROMPTS)

        frame_dir = self.work_dir / asset.video_id
        should_fallback = force_fallback_captions or (
            caption_fallback_deadline is not None
            and time.monotonic() >= caption_fallback_deadline
        )

        # --- Stage 1: Frame extraction ---
        frame_started_at = time.monotonic()
        frames: list[Path] = []
        ocr_frames: list[Path] = []
        crop_frames: list[Path] = []
        motion_frames: list[Path] = []
        video_duration = None if self.dry_run else probe_duration_seconds(asset.path)

        if not should_fallback:
            frames = [] if self.dry_run else extract_frames(
                asset.path,
                frame_dir,
                max_frames or self.max_frames,
                frame_profile=frame_profile,
            )
            # Extract complementary evidence frames concurrently
            ocr_frames, crop_frames, motion_frames = self._extract_complementary_frames(
                asset, frames, frame_dir, video_duration,
            )
        timings["frame_extraction_sec"] = time.monotonic() - frame_started_at

        # --- Stage 2: Evidence extraction + caption generation ---
        caption_started_at = time.monotonic()
        if should_fallback:
            evidence = dict(EMPTY_EVIDENCE)
            captions = {
                style: self._fallback_caption(style, evidence)
                for style in selected_styles
            }
        else:
            evidence = self._extract_evidence(
                asset=asset,
                keyframes=frames,
                ocr_frames=ocr_frames,
                crop_frames=crop_frames,
                motion_frames=motion_frames,
                video_duration=video_duration,
            )
            timings["evidence_extraction_sec"] = time.monotonic() - caption_started_at

            # --- Stage 3: Parallel candidate generation + selection ---
            gen_started_at = time.monotonic()
            do_retry = (
                self.enable_style_retry
                if enable_style_retry is None
                else enable_style_retry
            )
            captions = self._generate_and_select_captions(
                evidence, selected_styles, do_retry,
            )
            timings["caption_generation_sec"] = time.monotonic() - gen_started_at

        timings["total_caption_sec"] = time.monotonic() - caption_started_at

        # --- Stage 4: Quality checks ---
        checks = {}
        if self.run_checks:
            check_started_at = time.monotonic()
            checks = self._run_checks_concurrent(selected_styles, evidence, captions)
            timings["quality_check_sec"] = time.monotonic() - check_started_at
        timings["total_process_sec"] = time.monotonic() - started_at

        return {
            "video_id": asset.video_id,
            "source_path": str(asset.path),
            "frames": [str(frame) for frame in frames],
            "frame_count": len(frames),
            "ocr_frame_count": len(ocr_frames),
            "crop_frame_count": len(crop_frames),
            "motion_frame_count": len(motion_frames),
            "duration_seconds": video_duration,
            "sampling_strategy": frames[0].parent.name if frames else "none",
            "evidence": evidence,
            # Keep 'observations' key for backward compatibility with app.py
            "observations": evidence,
            "captions": captions,
            "checks": checks,
            "timings": timings,
        }

    # ======================================================================
    # Complementary frame extraction
    # ======================================================================

    def _extract_complementary_frames(
        self,
        asset: VideoAsset,
        keyframes: list[Path],
        frame_dir: Path,
        video_duration: float | None,
    ) -> tuple[list[Path], list[Path], list[Path]]:
        """Extract OCR, crop, and motion frames concurrently."""
        if self.dry_run or not keyframes:
            return [], [], []

        ocr_frames: list[Path] = []
        crop_frames: list[Path] = []
        motion_frames: list[Path] = []

        try:
            from track2_captioner.video_ingest import (
                extract_crop_frames,
                extract_ocr_frames,
                extract_motion_clip_frames,
            )
        except ImportError:
            logger.warning("Complementary extraction functions not available.")
            return [], [], []

        futures: dict[str, Future] = {}
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="complementary") as pool:
            if getattr(self.settings, "ocr_enabled", True):
                futures["ocr"] = pool.submit(
                    extract_ocr_frames,
                    asset.path,
                    frame_dir / "ocr",
                    3,
                    1280,
                )
            if getattr(self.settings, "crop_enabled", True) and keyframes:
                futures["crop"] = pool.submit(
                    extract_crop_frames,
                    keyframes,
                    frame_dir / "crops",
                )
            if getattr(self.settings, "motion_clip_enabled", True):
                futures["motion"] = pool.submit(
                    extract_motion_clip_frames,
                    asset.path,
                    frame_dir / "motion",
                    2,
                    3.0,
                )

            for name, future in futures.items():
                try:
                    result = future.result(timeout=8.0)
                    if name == "ocr":
                        ocr_frames = result
                    elif name == "crop":
                        crop_frames = result
                    elif name == "motion":
                        motion_frames = result
                except Exception:
                    logger.warning("Complementary extraction '%s' failed.", name, exc_info=True)

        return ocr_frames, crop_frames, motion_frames

    # ======================================================================
    # Evidence extraction (parallel experts)
    # ======================================================================

    def _extract_evidence(
        self,
        asset: VideoAsset,
        keyframes: list[Path],
        ocr_frames: list[Path],
        crop_frames: list[Path],
        motion_frames: list[Path],
        video_duration: float | None,
    ) -> dict[str, Any]:
        """Run parallel perception experts and fuse into a single evidence ledger."""
        if self.dry_run:
            return dict(EMPTY_EVIDENCE)
        assert self.client is not None

        # Run perception experts concurrently
        expert_results: dict[str, dict[str, Any]] = {}
        deadline = getattr(self.settings, "stage_deadline_perception", 12.0)

        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="perception") as pool:
            futures: dict[str, Future] = {}

            # Expert 1: Keyframe perception (primary)
            if keyframes:
                futures["keyframes"] = pool.submit(
                    self._perception_expert_call,
                    keyframes,
                    video_duration,
                    "keyframes",
                )

            # Expert 2: OCR perception on high-res unmodified frames
            if ocr_frames:
                futures["ocr"] = pool.submit(
                    self._perception_expert_call,
                    ocr_frames,
                    video_duration,
                    "ocr",
                )

            # Expert 3: Crop perception for fine detail
            if crop_frames:
                futures["crops"] = pool.submit(
                    self._perception_expert_call,
                    crop_frames[:3],  # Limit to 3 crops to stay within token budget
                    video_duration,
                    "crops",
                )

            for name, future in futures.items():
                try:
                    result = future.result(timeout=deadline)
                    expert_results[name] = result
                except Exception:
                    logger.warning("Perception expert '%s' failed.", name, exc_info=True)

        if not expert_results:
            return dict(EMPTY_EVIDENCE)

        # Fuse evidence from all experts
        return self._fuse_evidence(expert_results)

    def _perception_expert_call(
        self,
        frames: list[Path],
        video_duration: float | None,
        source_name: str,
    ) -> dict[str, Any]:
        """Call the perception model on a set of frames."""
        assert self.client is not None

        content: list[dict[str, Any]] = []
        frame_timestamps = []
        for index, frame in enumerate(frames):
            timestamp = self._frame_timestamp(frame, index, len(frames), video_duration)
            frame_timestamps.append(round(timestamp, 3))

        request = {
            "source": source_name,
            "task": "Extract factual visual evidence from these frames. "
                    "Report observations with confidence levels and timestamp references.",
            "frame_count": len(frames),
            "video_duration_seconds": round(video_duration, 3) if video_duration else None,
            "frame_timestamps_seconds": frame_timestamps,
        }
        content.append({"type": "text", "text": json.dumps(request, indent=2)})

        for index, frame in enumerate(frames):
            ts = frame_timestamps[index]
            content.append({
                "type": "text",
                "text": f"Frame {index + 1}/{len(frames)} at {self._format_timestamp(ts)}.",
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": image_to_data_url(frame)},
            })

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
        parsed = self._parse_or_repair_json(response, f"evidence from {source_name}")
        # Tag claims with their source
        for claim in parsed.get("claims", []):
            if isinstance(claim, dict):
                claim.setdefault("source", source_name)
        return parsed

    def _fuse_evidence(
        self, expert_results: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Fuse observations from multiple experts into a single evidence ledger.

        Only consensus or directly supported facts are preserved. Contradictions
        are flagged. Claims from each expert carry their source tag.
        """
        fused: dict[str, Any] = {
            "summary": "",
            "setting": "",
            "subjects": [],
            "subject_counts": {},
            "objects": [],
            "actions": [],
            "ocr": [],
            "camera_motion": "unknown",
            "claims": [],
            "uncertainties": [],
            "contradictions": [],
        }

        seen_subjects: set[str] = set()
        seen_objects: set[str] = set()
        seen_actions: set[str] = set()
        seen_ocr: set[str] = set()
        all_claims: list[dict[str, Any]] = []

        for source_name, result in expert_results.items():
            if not isinstance(result, dict):
                continue

            # Take the longest/most detailed summary
            summary = str(result.get("summary", "")).strip()
            if len(summary) > len(fused["summary"]):
                fused["summary"] = summary

            # Take the most specific setting
            setting = str(result.get("setting", "")).strip()
            if len(setting) > len(fused["setting"]):
                fused["setting"] = setting

            # Merge subjects
            for subject in self._string_list_from_keys(result, ["subjects"]):
                normalized = subject.lower().strip()
                if normalized not in seen_subjects:
                    seen_subjects.add(normalized)
                    fused["subjects"].append(subject)

            # Merge subject counts
            counts = result.get("subject_counts", {})
            if isinstance(counts, dict):
                for key, value in counts.items():
                    fused["subject_counts"].setdefault(key, value)

            # Merge objects
            for obj in self._string_list_from_keys(result, ["objects", "key_objects"]):
                normalized = obj.lower().strip()
                if normalized not in seen_objects:
                    seen_objects.add(normalized)
                    fused["objects"].append(obj)

            # Merge actions
            for action in self._string_list_from_keys(result, ["actions"]):
                normalized = action.lower().strip()
                if normalized not in seen_actions:
                    seen_actions.add(normalized)
                    fused["actions"].append(action)

            # Merge OCR (treated as untrusted data)
            for text in self._string_list_from_keys(result, ["ocr", "visible_text"]):
                normalized = text.strip()
                if normalized and normalized not in seen_ocr:
                    seen_ocr.add(normalized)
                    fused["ocr"].append(text)

            # Camera motion
            cam = str(result.get("camera_motion", "")).strip()
            if cam and cam != "unknown" and fused["camera_motion"] == "unknown":
                fused["camera_motion"] = cam

            # Collect all claims
            for claim in result.get("claims", []):
                if isinstance(claim, dict):
                    claim.setdefault("source", source_name)
                    all_claims.append(claim)

            # Uncertainties
            for unc in self._string_list_from_keys(result, ["uncertainties"]):
                if unc not in fused["uncertainties"]:
                    fused["uncertainties"].append(unc)

            # Contradictions
            for con in self._string_list_from_keys(result, ["contradictions"]):
                if con not in fused["contradictions"]:
                    fused["contradictions"].append(con)

        # Deduplicate claims by text, keeping highest confidence
        claim_map: dict[str, dict[str, Any]] = {}
        for claim in all_claims:
            text = claim.get("text", "").strip().lower()
            if not text:
                continue
            existing = claim_map.get(text)
            if existing is None or claim.get("confidence", 0) > existing.get("confidence", 0):
                claim_map[text] = claim
                # Track multi-source support
                if existing and existing.get("source") != claim.get("source"):
                    sources = set()
                    for s in [existing.get("source", ""), claim.get("source", "")]:
                        if s:
                            sources.add(s)
                    claim["sources"] = sorted(sources)

        fused["claims"] = list(claim_map.values())

        # Detect contradictions between claims
        self._detect_contradictions(fused)

        if not fused["summary"]:
            fused["summary"] = "Visual evidence extracted from video frames."

        return fused

    def _detect_contradictions(self, evidence: dict[str, Any]) -> None:
        """Simple contradiction detection: flag claims about the same type with low confidence."""
        claims = evidence.get("claims", [])
        type_groups: dict[str, list[dict]] = {}
        for claim in claims:
            claim_type = claim.get("type", "general")
            type_groups.setdefault(claim_type, []).append(claim)

        for claim_type, group in type_groups.items():
            if len(group) > 1:
                confidences = [c.get("confidence", 0.5) for c in group]
                if min(confidences) < 0.5 and max(confidences) > 0.7:
                    low_conf = [c for c in group if c.get("confidence", 0.5) < 0.5]
                    for c in low_conf:
                        contradiction = f"Low-confidence {claim_type} claim: {c.get('text', '')}"
                        if contradiction not in evidence["contradictions"]:
                            evidence["contradictions"].append(contradiction)

    # ======================================================================
    # Parallel candidate generation
    # ======================================================================

    def _generate_and_select_captions(
        self,
        evidence: dict[str, Any],
        styles: list[str],
        enable_retry: bool,
    ) -> dict[str, str]:
        """Generate multiple candidates per style in parallel, then select the best."""
        if self.dry_run:
            return {style: DRY_RUN_CAPTIONS.get(style, "") for style in styles}

        assert self.client is not None
        candidate_count = getattr(self.settings, "candidate_count", 4)

        # Generate candidates for all styles concurrently
        # Each style gets candidate_count independent calls
        style_candidates: dict[str, list[str]] = {style: [] for style in styles}
        futures: dict[Future, tuple[str, int]] = {}

        max_workers = min(len(styles) * candidate_count, 8)
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="candidates") as pool:
            for style in styles:
                for i in range(candidate_count):
                    # Vary temperature for diversity
                    temp = self._candidate_temperature(style, i, candidate_count)
                    future = pool.submit(
                        self._generate_single_candidate,
                        evidence, style, temp,
                    )
                    futures[future] = (style, i)

            deadline = getattr(self.settings, "stage_deadline_candidates", 10.0)
            for future in as_completed(futures, timeout=deadline + 2):
                style, index = futures[future]
                try:
                    caption = future.result(timeout=2.0)
                    if caption:
                        style_candidates[style].append(caption)
                except Exception:
                    logger.warning("Candidate %d for style '%s' failed.", index, style, exc_info=True)

        # Select best candidate per style
        captions: dict[str, str] = {}
        for style in styles:
            candidates = style_candidates[style]
            if not candidates:
                captions[style] = self._fallback_caption(style, evidence)
                continue

            if len(candidates) == 1:
                captions[style] = self._clean_caption(candidates[0])
            else:
                captions[style] = self._select_best_candidate(
                    evidence, style, candidates,
                )

        # Ensure all styles have captions
        captions = self._sanitize_caption_map(styles, captions, evidence)

        # Optional repair pass
        if enable_retry:
            issues = self._caption_issues(styles, captions)
            if issues:
                captions = self._repair_captions(evidence, styles, captions, issues)

        return captions

    def _candidate_temperature(
        self, style: str, index: int, total: int,
    ) -> float:
        """Compute temperature for a candidate to ensure diversity.

        Formal style uses lower temperature; humor styles explore more broadly.
        """
        base = self.settings.temperature if style == "formal" else self.settings.creative_temperature
        if total <= 1:
            return base
        # Spread temperatures across range for diversity
        spread = 0.15
        offset = (index / (total - 1)) * spread - (spread / 2)
        return max(0.1, min(1.0, base + offset))

    def _generate_single_candidate(
        self,
        evidence: dict[str, Any],
        style: str,
        temperature: float,
    ) -> str:
        """Generate a single caption candidate for one style."""
        assert self.client is not None

        style_instruction = load_prompt(STYLE_PROMPTS[style]) if style in STYLE_PROMPTS else ""
        request = {
            "target_style": style,
            "evidence_ledger": evidence,
            "style_instructions": style_instruction,
            "rules": [
                "Use ONLY facts from the evidence ledger. Do not add unsupported details.",
                "Generate exactly one caption in the target style.",
                "The caption must be one concise English sentence, ideally 12 to 30 words.",
                "Figurative language may change framing but CANNOT introduce a new subject, "
                "action, setting, object, or intention not in the evidence.",
                "Never mention observations, prompts, models, frames, timestamps, or analysis mechanics.",
                "Return JSON with key 'caption'.",
            ],
        }

        response = self.client.chat(
            self.settings.caption_model,
            [
                {"role": "system", "content": STYLE_CAPTION_SYSTEM},
                {"role": "user", "content": json.dumps(request, indent=2)},
            ],
            max_tokens=self.settings.caption_max_tokens,
            temperature=temperature,
            reasoning_effort=self.settings.reasoning_effort,
            json_mode=True,
        )
        parsed = parse_json_object(response)
        return str(parsed.get("caption", "")).strip()

    def _select_best_candidate(
        self,
        evidence: dict[str, Any],
        style: str,
        candidates: list[str],
    ) -> str:
        """Use the selector model to score candidates and pick the best."""
        assert self.client is not None

        selector_model = getattr(self.settings, "selector_model", "") or self.settings.caption_model
        request = {
            "task": "Score each candidate caption and select the best one.",
            "target_style": style,
            "evidence_ledger": evidence,
            "candidates": [
                {"index": i, "caption": c} for i, c in enumerate(candidates)
            ],
            "scoring_criteria": [
                "factual_accuracy: Does the caption only contain facts from the evidence? (0.0-1.0)",
                "subject_action_coverage: Does it mention the main subjects and actions? (0.0-1.0)",
                "style_strength: Does it match the target style convincingly? (0.0-1.0)",
                "naturalness: Does it read naturally as a caption? (0.0-1.0)",
                "concision: Is it appropriately concise? (0.0-1.0)",
            ],
            "instructions": [
                "Return the index of the best candidate in 'best_index'.",
                "List any unsupported claims or omissions.",
                "If the best candidate needs repair, provide repair_instructions.",
            ],
        }

        try:
            response = self.client.chat(
                selector_model,
                [
                    {"role": "system", "content": SELECTOR_SYSTEM},
                    {"role": "user", "content": json.dumps(request, indent=2)},
                ],
                max_tokens=self.settings.caption_max_tokens,
                temperature=0.1,
                reasoning_effort=self.settings.reasoning_effort,
                json_mode=True,
            )
            parsed = parse_json_object(response)
            best_index = int(parsed.get("best_index", 0))
            best_index = max(0, min(best_index, len(candidates) - 1))
            selected = self._clean_caption(candidates[best_index])

            # If repair is needed and instructions given, apply repair
            repair = str(parsed.get("repair_instructions", "")).strip()
            if repair and repair.lower() not in ("none", "n/a", ""):
                repaired = self._repair_single_caption(
                    evidence, style, selected, repair,
                )
                if repaired:
                    selected = repaired

            return selected
        except Exception:
            logger.warning("Selector failed for style '%s'; using first candidate.", style, exc_info=True)
            return self._clean_caption(candidates[0])

    def _repair_single_caption(
        self,
        evidence: dict[str, Any],
        style: str,
        caption: str,
        repair_instructions: str,
    ) -> str:
        """Apply targeted repair to a single caption."""
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
                reasoning_effort=self.settings.reasoning_effort,
                json_mode=True,
            )
            parsed = parse_json_object(response)
            repaired = str(parsed.get("caption", "")).strip()
            return self._clean_caption(repaired) if repaired else ""
        except Exception:
            logger.warning("Repair failed for style '%s'.", style, exc_info=True)
            return ""

    def _repair_captions(
        self,
        evidence: dict[str, Any],
        styles: list[str],
        current_captions: dict[str, str],
        issues: list[str],
    ) -> dict[str, str]:
        """Repair captions that have style or length issues."""
        # Identify which styles need repair
        styles_to_repair = set()
        for issue in issues:
            style = issue.split(":")[0].strip()
            if style in current_captions:
                styles_to_repair.add(style)

        repaired = dict(current_captions)
        for style in styles_to_repair:
            repair_instruction = "; ".join(
                i.split(":", 1)[1].strip() for i in issues if i.startswith(style)
            )
            result = self._repair_single_caption(
                evidence, style, current_captions[style], repair_instruction,
            )
            if result:
                repaired[style] = result

        return self._sanitize_caption_map(styles, repaired, evidence)

    # ======================================================================
    # Quality checks
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
                json_mode=True,
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
                "accuracy": "pass" if factual >= 0.6 else "fail",
                "tone": "pass" if style_score >= 0.6 else "fail",
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
    def _normalize_caption_key(key: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", key.lower())

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
