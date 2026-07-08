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
    "people": ["visible people and clothing, or empty array"],
    "key_objects": ["important visible objects, colors, signs, or environmental details"],
    "actions": ["important visible action"],
    "mood": "visual mood or atmosphere, conservatively worded",
    "key_moments": ["important visual moment"],
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

CAPTION_SET_SCHEMA = json.dumps({
    "captions": {
        "formal": "one formal caption",
        "sarcastic": "one sarcastic caption",
        "humorous_tech": "one humorous technology caption",
        "humorous_non_tech": "one humorous non-technical caption",
    },
}, indent=2)

CAPTION_SELECTION_SCHEMA = json.dumps({
    "captions": {
        "formal": "chosen exact formal caption",
        "sarcastic": "chosen exact sarcastic caption",
        "humorous_tech": "chosen exact humorous technology caption",
        "humorous_non_tech": "chosen exact humorous non-technical caption",
    },
    "winners": {
        "formal": 1,
        "sarcastic": 2,
        "humorous_tech": 1,
        "humorous_non_tech": 2,
    },
    "reason": "brief reason the selected captions are best",
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
    "deadline",
    "operator",
    "furiously",
    "frantically",
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
    "people": [],
    "key_objects": ["sample object"],
    "actions": ["sample action"],
    "mood": "neutral",
    "key_moments": ["sample moment"],
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
                    "setting, people, subjects, colors, countable objects, actions, mood, "
                    "key moments, scene changes, visible text, camera movement, and "
                    "transcript-backed speech. Do not call the clip animation, CGI, rendered, "
                    "cartoon, painterly, painting-like, impressionistic, or filtered unless "
                    "the sampled frames are clearly non-photographic."
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

        caption_sets = self._caption_sets(styles, observations)
        if not caption_sets:
            return {
                style: self._fallback_caption(style, observations)
                for style in styles
            }
        if len(caption_sets) == 1:
            return self._sanitize_caption_map(styles, caption_sets[0]["captions"], observations)
        return self._rerank_caption_sets(styles, observations, caption_sets)

    def _caption_sets(self, styles: list[str], observations: dict[str, Any]) -> list[dict[str, Any]]:
        caption_models = self.settings.caption_models or (self.settings.caption_model,)
        caption_sets: list[dict[str, Any]] = []
        workers = min(len(caption_models), 2)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_model = {
                pool.submit(self._caption_set_from_model, model, styles, observations): model
                for model in caption_models
            }
            for future in as_completed(future_to_model):
                model = future_to_model[future]
                try:
                    caption_set = future.result()
                    if caption_set["captions"]:
                        caption_sets.append(caption_set)
                except Exception:
                    logger.warning("Caption model %s failed.", model, exc_info=True)

        return caption_sets

    def _caption_set_from_model(
        self,
        model: str,
        styles: list[str],
        observations: dict[str, Any],
    ) -> dict[str, Any]:
        assert self.client is not None
        request = {
            "task": "Generate one grounded final caption for each requested style.",
            "styles": {
                style: STYLE_DESCRIPTIONS.get(style, "")
                for style in styles
            },
            "strict_grounding_rules": [
                "Use only summary, setting, subjects, key_objects, actions, timeline, visible_text, and audio_or_speech as factual evidence.",
                "Use people, mood, and key_moments when present, but keep them conservative.",
                "Never turn anything in uncertainties into a fact.",
                "If the exact location, identity, motive, or text is uncertain, use generic wording instead of guessing.",
                "Never quote or name visible text, signs, brands, or organizations in the final caption; describe them generically if needed.",
                "Do not mention animation, CGI, rendering, shaders, or painterly style unless the observations prove the clip is non-photographic.",
                "Do not mention stylized effects, brushstrokes, painting, filters, impressionistic style, long exposure, or intense emotions unless directly supported.",
                "Do not invent hidden context such as deadlines, snacks, management, secrets, camera-operator motives, intentions, or what someone is probably doing.",
                "Each caption must be one sentence, ideally 12 to 28 words and no more than 35 words.",
                "Return strict JSON only.",
            ],
            "style_notes": {
                "formal": "Professional, objective, factual tone. No jokes, slang, sarcasm, or embellishment.",
                "sarcastic": "Dry, ironic, lightly mocking, and obviously sarcastic while staying factual.",
                "humorous_tech": "Funny with technology or programming references such as API, pipeline, logs, latency, cache, deploy, or debug.",
                "humorous_non_tech": "Funny everyday humor for a general audience, with no technical jargon.",
            },
            "observations": observations,
            "response_schema": CAPTION_SET_SCHEMA,
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a grounded video-caption writer. "
                    "Write concise captions that a strict LLM judge will score as accurate and style-matched. "
                    "Return only valid JSON."
                ),
            },
            {"role": "user", "content": json.dumps(request, indent=2)},
        ]
        response = self._chat_model(
            model,
            messages,
            max_tokens=self.settings.caption_max_tokens,
            temperature=self.settings.creative_temperature,
            json_mode=True,
        )
        parsed = self._parse_or_repair_json(
            response,
            f"caption set from {model}",
            CAPTION_SET_SCHEMA,
            repair_model=model,
        )
        captions = self._extract_caption_map(styles, parsed, observations)
        if self._needs_caption_set_repair(styles, captions):
            captions = self._repair_caption_set(model, styles, observations, captions)
        return {
            "source_model": model,
            "captions": self._sanitize_caption_map(styles, captions, observations),
        }

    def _repair_caption_set(
        self,
        model: str,
        styles: list[str],
        observations: dict[str, Any],
        captions: dict[str, str],
    ) -> dict[str, str]:
        issues = []
        for style in styles:
            caption = captions.get(style, "")
            if not caption:
                issues.append(f"{style}: missing caption")
            elif self._needs_style_retry(style, caption):
                issues.append(f"{style}: style is too weak")
            elif self._needs_grounding_retry(caption):
                issues.append(f"{style}: contains unsupported hidden context")
            elif self._needs_length_retry(caption):
                issues.append(f"{style}: too long or multi-sentence")
        if not issues:
            return captions

        repair_request = {
            "task": "Repair only the listed caption issues.",
            "issues": issues,
            "current_captions": captions,
            "styles": {
                style: STYLE_DESCRIPTIONS.get(style, "")
                for style in styles
            },
            "rules": [
                "Preserve observed facts only.",
                "Make the target style obvious.",
                "Use one sentence per caption, ideally 12 to 28 words and no more than 35 words.",
                "Do not invent hidden context or quote visible signs, brands, or organizations.",
                "Do not mention animation, CGI, rendering, shaders, or painterly style unless the observations prove the clip is non-photographic.",
                "Avoid unsupported artistic technique or intensity words such as stylized, brushstrokes, painting, filters, impressionistic, long exposure, furiously, or frantically.",
                "Return strict JSON only.",
            ],
            "observations": observations,
            "response_schema": CAPTION_SET_SCHEMA,
        }
        messages = [
            {
                "role": "system",
                "content": "You repair grounded video captions and return only valid JSON.",
            },
            {"role": "user", "content": json.dumps(repair_request, indent=2)},
        ]
        response = self._chat_model(
            model,
            messages,
            max_tokens=self.settings.caption_max_tokens,
            temperature=max(0.2, self.settings.creative_temperature - 0.2),
            json_mode=True,
        )
        parsed = self._parse_or_repair_json(
            response,
            f"caption repair from {model}",
            CAPTION_SET_SCHEMA,
            repair_model=model,
        )
        repaired = self._extract_caption_map(styles, parsed, observations)
        merged = dict(captions)
        merged.update({style: caption for style, caption in repaired.items() if caption})
        return merged

    def _rerank_caption_sets(
        self,
        styles: list[str],
        observations: dict[str, Any],
        caption_sets: list[dict[str, Any]],
    ) -> dict[str, str]:
        assert self.client is not None
        candidate_sets = [
            {
                "id": index,
                "source_model": caption_set["source_model"],
                "captions": self._sanitize_caption_map(styles, caption_set["captions"], observations),
            }
            for index, caption_set in enumerate(caption_sets, start=1)
        ]
        rerank_request = {
            "task": "Compare two model-generated caption sets and choose the best final caption for each style.",
            "styles": {
                style: STYLE_DESCRIPTIONS.get(style, "")
                for style in styles
            },
            "selection_rules": [
                "Choose independently per style; the final four captions may come from different source models.",
                "Pick the caption most likely to score highest with an LLM judge for factual accuracy and style match.",
                "Factual accuracy is more important than humor.",
                "Reject unsupported concrete details, guessed text, motives, speech, locations, identities, or hidden context.",
                "Reject candidates that quote or name visible text, signs, brands, or organizations.",
                "Reject media-type guesses such as animation, CGI, render, shader, or painterly style unless clearly proven.",
                "Reject unsupported wording about stylized effects, brushstrokes, painting, filters, impressionistic style, long exposure, or intense emotions.",
                "Reject candidates that mention deadlines, snacks, management, secrets, intentions, or what someone is probably doing unless explicitly observed.",
                "Prefer one-sentence captions between 12 and 28 words; reject wordy or multi-sentence candidates when a concise option is accurate.",
                "Prefer concise captions that mention the main subject, setting, and action when supported.",
                "For sarcastic and humorous styles, the style must be obvious but still grounded.",
                "Do not prefer a caption because of its source model name.",
                "Return the chosen exact captions without rewriting them.",
                "Return strict JSON only.",
            ],
            "observations": observations,
            "candidate_sets": candidate_sets,
            "response_schema": CAPTION_SELECTION_SCHEMA,
        }
        try:
            response = self.client.chat(
                self.settings.rerank_model,
                [
                    {
                        "role": "system",
                        "content": (
                            "You are a strict video-caption judge. "
                            "Choose the best final caption for each requested style. "
                            "Return only valid JSON."
                        ),
                    },
                    {"role": "user", "content": json.dumps(rerank_request, indent=2)},
                ],
                max_tokens=self.settings.rerank_max_tokens,
                temperature=0.0,
                reasoning_effort=self.settings.reasoning_effort,
                json_mode=True,
            )
            parsed = self._parse_or_repair_json(
                response,
                "caption set selection",
                CAPTION_SELECTION_SCHEMA,
                repair_model=self.settings.rerank_model,
            )
            selected = self._extract_caption_map(styles, parsed, observations)
            if selected:
                return self._sanitize_caption_map(styles, selected, observations)
        except Exception:
            logger.warning("Caption set reranking failed.", exc_info=True)
        return self._sanitize_caption_map(styles, caption_sets[0]["captions"], observations)

    @staticmethod
    def _clean_caption(response: str) -> str:
        return response.strip().strip('"').strip()

    @staticmethod
    def _sanitize_caption_text_claims(caption: str, observations: dict[str, Any]) -> str:
        cleaned = caption
        cleaned = re.sub(
            r"\b(?:a|an|the)\s+(?:painterly\s+)?(?:animation|animated clip|cartoon|cgi render|rendered scene)\s+"
            r"(?:shows|depicts|features|captures)\s+",
            "a video shows ",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\b(?:painterly|cgi|rendered|animated|cartoon)\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bis shown in a soft,\s*impressionistic visual style with\b", "has", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bsoft,\s*impressionistic visual style with\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bimpressionistic visual style\b", "motion blur", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bwith\s+stylized\s+visual\s+effects\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bstylized\s+visual\s+effects\b", "visible scenery", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bartistic\s+long-exposure\s+traffic\s+jam\b", "motion-blurred traffic scene", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\blong-exposure\b", "motion-blurred", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bbrushstrokes\b", "motion blur", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bgolden-blur painting\b", "golden motion blur", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bfuzzy golden painting\b", "blurred autumn scene", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bpainting\b", "scene", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bheavy blur filter\b", "heavy motion blur", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bfilter\b", "effect", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bartfully obscured\b", "partly obscured", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bblurry\s+motion\s+blur\b", "motion blur", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\btypes\s+(?:furiously|frantically)\b", "types", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b(?:furiously|frantically)\b", "steadily", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bthe world's most important email\b", "a dramatic message", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bthe world's longest email\b", "a very long message", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bthat report won't finish itself\b", "those keystrokes won't type themselves", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\banimation\s+loop\b", "loop", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\banimation\b", "motion", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bshader\b", "lighting", cleaned, flags=re.IGNORECASE)
        all_caps_phrase = r"[A-Z][A-Z0-9&.'-]*(?:\s+[A-Z][A-Z0-9&.'-]*){1,7}"
        label_words = r"sign|building|banner|logo|text"
        cleaned = re.sub(
            rf"\b(?:under|near|past|beside|outside|by|beneath|below|behind|around)\s+"
            rf"(?:the\s+)?{all_caps_phrase}\s+({label_words})\b",
            r"near a visible \1",
            cleaned,
        )
        cleaned = re.sub(
            rf"\b{all_caps_phrase}\s+({label_words})\b",
            r"a visible \1",
            cleaned,
        )

        visible_text = observations.get("visible_text", [])
        if isinstance(visible_text, list):
            for item in visible_text:
                text = str(item).strip()
                if len(text) < 4:
                    continue
                cleaned = re.sub(re.escape(text), "a visible sign", cleaned, flags=re.IGNORECASE)

        cleaned = re.sub(r"\ba visible sign\s+(sign|building|banner|logo|text)\b", r"a visible \1", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        cleaned = re.sub(r"\s+([,.!?])", r"\1", cleaned)
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]
        return cleaned.strip()

    def _extract_caption_map(
        self,
        styles: list[str],
        parsed: dict[str, Any],
        observations: dict[str, Any],
    ) -> dict[str, str]:
        payload = parsed.get("captions", parsed)
        if not isinstance(payload, dict):
            return {}

        captions: dict[str, str] = {}
        for style in styles:
            value = payload.get(style)
            if isinstance(value, dict):
                value = value.get("caption") or value.get("text")
            caption = self._clean_caption(str(value or ""))
            if caption:
                captions[style] = self._sanitize_caption_text_claims(caption, observations)
        return captions

    def _sanitize_caption_map(
        self,
        styles: list[str],
        captions: dict[str, str],
        observations: dict[str, Any],
    ) -> dict[str, str]:
        cleaned: dict[str, str] = {}
        for style in styles:
            caption = self._clean_caption(str(captions.get(style, "")))
            if not caption:
                caption = self._fallback_caption(style, observations)
            cleaned[style] = self._sanitize_caption_text_claims(caption, observations)
        return cleaned

    def _needs_caption_set_repair(self, styles: list[str], captions: dict[str, str]) -> bool:
        for style in styles:
            caption = captions.get(style, "")
            if (
                not caption
                or self._needs_style_retry(style, caption)
                or self._needs_grounding_retry(caption)
                or self._needs_length_retry(caption)
            ):
                return True
        return False

    def _chat_model(
        self,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        json_mode: bool = False,
    ) -> str:
        assert self.client is not None
        return self.client.chat(
            model,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=self.settings.reasoning_effort,
            json_mode=json_mode,
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
        self,
        response: str,
        label: str,
        expected_schema: str | None = None,
        repair_model: str | None = None,
    ) -> dict[str, Any]:
        try:
            return parse_json_object(response)
        except Exception:
            assert self.client is not None
            model = repair_model or self.settings.model
            schema_hint = ""
            if expected_schema:
                schema_hint = f"\n\nExpected JSON schema:\n{expected_schema}"
            repaired = self.client.chat(
                model,
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
                max_tokens=min(self.settings.max_tokens, 1000),
                temperature=0.0,
                json_mode=True,
            )
            return parse_json_object(repaired)

    @staticmethod
    def _read_transcript(asset: VideoAsset) -> str:
        if asset.transcript_path is None:
            return ""
        return asset.transcript_path.read_text(encoding="utf-8").strip()
