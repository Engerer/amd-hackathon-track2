from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from track2_captioner.config import Settings
from track2_captioner.fireworks_client import FireworksClient, image_to_data_url
from track2_captioner.prompts import STYLE_PROMPTS, load_prompt
from track2_captioner.video_ingest import DEFAULT_MAX_FRAMES, VideoAsset, extract_frames


logger = logging.getLogger(__name__)

FRAME_POSITIONS = ["beginning", "early-middle", "middle", "late-middle", "end"]
CREATIVE_STYLES = {"sarcastic", "humorous_tech", "humorous_non_tech"}

SHARED_VISUAL_SYSTEM = (
    "You are an accuracy-first multimodal video captioner. You receive exactly five silent "
    "frames from one video in chronological order: beginning, early-middle, middle, "
    "late-middle, and end. Inspect all five images before writing. Ground the caption in the "
    "clearly visible subject, setting, and primary action or state. Use a broad description "
    "when a detail is ambiguous. Do not infer audio, speech, identity, intent, emotion, "
    "causality, exact location, off-screen events, or continuity that sparse frames do not "
    "prove. Never let humor replace the literal visible event. Output one concise English "
    "sentence only, with no JSON, label, explanation, markdown, or quotation marks."
)

DRY_RUN_CAPTIONS = {
    "formal": "A sample subject performs a visible action in a simple scene.",
    "sarcastic": "A sample subject performs a visible action, because apparently the scene needed supervision.",
    "humorous_tech": "A sample subject executes the visible action like a process running without errors.",
    "humorous_non_tech": "A sample subject keeps the visible scene moving, giving the moment its daily exercise.",
}


class CaptionPipeline:
    """Generate each requested style with an independent parallel Kimi vision call."""

    def __init__(
        self,
        settings: Settings,
        work_dir: Path,
        dry_run: bool = False,
        max_frames: int = DEFAULT_MAX_FRAMES,
        run_checks: bool = False,
    ) -> None:
        self.settings = settings
        self.work_dir = work_dir
        self.dry_run = dry_run
        self.max_frames = 5
        self.run_checks = False
        self.client = None if dry_run else FireworksClient(
            settings.api_key,
            settings.base_url,
            settings.proxy_url,
            settings.proxy_token,
            max_retries=settings.max_retries,
        )

    def process(self, asset: VideoAsset, styles: list[str] | None = None) -> dict[str, Any]:
        requested = styles or list(STYLE_PROMPTS)
        selected_styles = []
        for style in requested:
            if style not in STYLE_PROMPTS:
                raise ValueError(f"Unsupported requested style: {style!r}")
            if style not in selected_styles:
                selected_styles.append(style)

        frame_dir = self.work_dir / asset.video_id
        frames = [] if self.dry_run else extract_frames(asset.path, frame_dir, 5)
        if not self.dry_run and len(frames) != 5:
            raise ValueError(f"Kimi captioning requires exactly five frames, got {len(frames)}.")

        captions = self._captions(selected_styles, frames)
        return {
            "video_id": asset.video_id,
            "source_path": str(asset.path),
            "frames": [str(frame) for frame in frames],
            "frame_count": len(frames),
            "sampling_strategy": frames[0].parent.name if frames else "none",
            "observations": {},
            "captions": captions,
            "checks": {},
        }

    def _captions(self, styles: list[str], frames: list[Path]) -> dict[str, str]:
        if self.dry_run:
            return {style: DRY_RUN_CAPTIONS[style] for style in styles}

        captions: dict[str, str] = {}
        workers = min(4, len(styles))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="kimi-style") as pool:
            futures = {
                pool.submit(self._caption_style, style, frames): style
                for style in styles
            }
            for future in as_completed(futures):
                style = futures[future]
                try:
                    captions[style] = future.result()
                except Exception:
                    logger.warning("Kimi caption generation failed for %s.", style, exc_info=True)
                    captions[style] = self._fallback_caption(style)

        return {
            style: captions.get(style, self._fallback_caption(style))
            for style in styles
        }

    def _caption_style(self, style: str, frames: list[Path]) -> str:
        assert self.client is not None
        if len(frames) != 5:
            raise ValueError(f"Style call requires exactly five frames, got {len(frames)}.")

        style_prompt = load_prompt(STYLE_PROMPTS[style])
        system_prompt = f"{SHARED_VISUAL_SYSTEM}\n\nTARGET STYLE RULES:\n{style_prompt}"
        content: list[dict[str, Any]] = [{
            "type": "text",
            "text": (
                f"Generate the {style} caption directly from these five chronological video "
                "frames. Preserve the same literal visible event across styles; apply only the "
                "requested tone. Inspect beginning, middle, and end evidence before answering."
            ),
        }]
        for index, frame in enumerate(frames):
            content.append({
                "type": "text",
                "text": f"Frame {index + 1}/5 - {FRAME_POSITIONS[index]} video sample",
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": image_to_data_url(frame)},
            })

        temperature = (
            self.settings.creative_temperature
            if style in CREATIVE_STYLES
            else self.settings.temperature
        )
        response = self.client.chat(
            self.settings.model,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            max_tokens=self.settings.caption_max_tokens,
            temperature=temperature,
            reasoning_effort=self.settings.reasoning_effort,
        )
        caption = self._clean_caption(response)
        return caption or self._fallback_caption(style)

    @staticmethod
    def _clean_caption(text: str) -> str:
        cleaned = text.strip().strip('"').strip()
        cleaned = re.sub(r"^```(?:text)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^(?:caption|formal|sarcastic|humorous[_ -]tech|humorous[_ -]non[_ -]tech)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @staticmethod
    def _fallback_caption(style: str) -> str:
        if style == "formal":
            return "The video presents visible subjects and activity within the scene."
        if style == "sarcastic":
            return "Visible subjects continue their activity, because apparently the scene requires careful supervision."
        if style == "humorous_tech":
            return "The visible scene keeps its activity running like a process with no scheduled downtime."
        return "The visible subjects keep things moving, giving the scene its daily exercise."
