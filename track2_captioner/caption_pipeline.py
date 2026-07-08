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
from track2_captioner.video_ingest import DEFAULT_MAX_FRAMES, VideoAsset, extract_frames


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

RERANK_SCHEMA = json.dumps({
    "winner": 1,
    "reason": "brief reason the selected caption is best",
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

HIDDEN_CONTEXT_MARKERS = {
    "probably",
    "presumably",
    "must be",
    "management",
    "deadline",
    "operator",
    "sensed",
    "defusing",
    "bomb",
    "snack",
    "treat",
    "secret",
    "forgot",
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
                    f"Analyze these {len(frames)} sampled frames in chronological order. "
                    "Capture exact visible facts that would help a judge compare captions: "
                    "setting, subjects, colors, countable objects, actions, scene changes, "
                    "visible text, camera movement, and transcript-backed speech."
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
                "Do not quote visible text, signs, brand names, or organization names in the final caption unless the video would be hard to identify without it.",
                "Do not invent hidden context such as deadlines, snacks, management, secrets, camera-operator motives, intentions, or what someone is probably doing.",
                "Mention the main subject, setting, and primary action when supported.",
                "Keep the final caption to one sentence, ideally 12 to 28 words and no more than 35 words.",
                "Return only the final caption text.",
            ],
            "observations": observations,
        }
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(caption_request, indent=2)},
        ]
        candidates = self._caption_candidates(style, observations, messages, temp)
        if len(candidates) == 1:
            return candidates[0]
        return self._rerank_caption(style, observations, candidates)

    def _caption_candidates(
        self,
        style: str,
        observations: dict[str, Any],
        messages: list[dict[str, Any]],
        temperature: float,
    ) -> list[str]:
        candidates: list[str] = []
        for candidate_index in range(self.settings.caption_candidates):
            candidate_messages = list(messages)
            if candidates:
                candidate_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Write a different valid candidate for the same style. "
                            "Keep it grounded in the same observations. "
                            "Avoid repeating these previous candidates:\n"
                            f"{json.dumps(candidates, indent=2)}"
                        ),
                    }
                )
            response = self._caption_chat(candidate_messages, temperature)
            caption = self._clean_caption(response)
            if self._needs_style_retry(style, caption):
                retry_messages = candidate_messages + [
                    {
                        "role": "user",
                        "content": (
                            "Rewrite the caption. It was too plain for the requested style. "
                            "Keep the same observed facts, but make the target style obvious. "
                            "Return only the rewritten caption text."
                        ),
                    }
                ]
                response = self._caption_chat(retry_messages, temperature)
                caption = self._clean_caption(response)
            if self._needs_grounding_retry(caption):
                retry_messages = candidate_messages + [
                    {
                        "role": "user",
                        "content": (
                            "Rewrite the caption to remove unsupported hidden context. "
                            "Do not mention deadlines, snacks, management, secrets, camera-operator motives, or what someone is probably doing. "
                            "Keep the requested style obvious, but use only visible or audible evidence. "
                            "Return only the rewritten caption text."
                        ),
                    }
                ]
                response = self._caption_chat(retry_messages, max(0.2, temperature - 0.2))
                caption = self._clean_caption(response)
            if self._needs_length_retry(caption):
                retry_messages = candidate_messages + [
                    {
                        "role": "user",
                        "content": (
                            "Rewrite the caption as one concise sentence of 12 to 28 words. "
                            "Keep the target style obvious and preserve only observed facts. "
                            "Return only the rewritten caption text."
                        ),
                    }
                ]
                response = self._caption_chat(retry_messages, max(0.2, temperature - 0.2))
                caption = self._clean_caption(response)
            if caption and caption not in candidates:
                candidates.append(caption)

        if not candidates:
            candidates.append(self._fallback_caption(style, observations))
        return candidates

    def _rerank_caption(self, style: str, observations: dict[str, Any], candidates: list[str]) -> str:
        assert self.client is not None
        rerank_request = {
            "target_style": style,
            "style_requirement": STYLE_DESCRIPTIONS.get(style, ""),
            "selection_rules": [
                "Pick the caption most likely to score highest with an LLM judge.",
                "Factual accuracy is more important than humor.",
                "Reject unsupported concrete details, guessed text, motives, speech, locations, identities, or hidden context.",
                "Strongly prefer candidates that avoid exact sign, brand, or organization names unless the video would be hard to identify without them.",
                "Reject candidates that mention deadlines, snacks, management, secrets, intentions, or what someone is probably doing unless explicitly observed.",
                "Prefer one-sentence captions between 12 and 28 words; reject wordy or multi-sentence candidates when a concise option is accurate.",
                "Prefer concise captions that mention the main subject, setting, and action when supported.",
                "For sarcastic and humorous styles, the style must be obvious but still grounded.",
                "Return strict JSON only.",
            ],
            "observations": observations,
            "candidates": [
                {"id": index, "caption": caption}
                for index, caption in enumerate(candidates, start=1)
            ],
            "response_schema": RERANK_SCHEMA,
        }
        try:
            response = self.client.chat(
                self.settings.rerank_model,
                [
                    {
                        "role": "system",
                        "content": (
                            "You are a strict video-caption reranker. "
                            "Choose the single best candidate for factual accuracy and target style. "
                            "Do not rewrite the caption."
                        ),
                    },
                    {"role": "user", "content": json.dumps(rerank_request, indent=2)},
                ],
                max_tokens=self.settings.rerank_max_tokens,
                temperature=0.0,
                reasoning_effort=self.settings.reasoning_effort,
                json_mode=True,
            )
            parsed = parse_json_object(response)
            winner = int(parsed.get("winner", 1))
            if 1 <= winner <= len(candidates):
                return candidates[winner - 1]
        except Exception:
            logger.warning("Caption reranking failed for style %s.", style, exc_info=True)
        return candidates[0]

    @staticmethod
    def _clean_caption(response: str) -> str:
        return response.strip().strip('"').strip()

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
    def _needs_grounding_retry(caption: str) -> bool:
        normalized = caption.lower()
        return any(marker in normalized for marker in HIDDEN_CONTEXT_MARKERS)

    @staticmethod
    def _needs_length_retry(caption: str) -> bool:
        words = re.findall(r"\b[\w'-]+\b", caption)
        sentence_count = len(re.findall(r"[.!?]+", caption))
        return len(words) > 35 or sentence_count > 1

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
        summary = str(observations.get("summary") or observations.get("setting") or "").strip()
        subjects = ", ".join(str(item) for item in observations.get("subjects", [])[:2])
        actions = ", ".join(str(item) for item in observations.get("actions", [])[:2])
        base = summary or f"The video shows {subjects or 'visible subjects'} with {actions or 'visible activity'}."
        if style == "formal":
            return base
        if style == "sarcastic":
            return f"{base} A very serious moment for ordinary visual evidence."
        if style == "humorous_tech":
            return f"{base} The scene ships its visual update with no rollback needed."
        return f"{base} It is doing its best to make everyday motion look eventful."

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
