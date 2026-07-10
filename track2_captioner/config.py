from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> None:
        return None


@dataclass(frozen=True)
class Settings:
    api_key: str
    model: str
    caption_model: str
    judge_model: str
    base_url: str = "https://api.fireworks.ai/inference/v1"
    proxy_url: str = ""
    proxy_token: str = ""
    # --- Generation parameters ---
    temperature: float = 0.2
    creative_temperature: float = 0.45
    max_tokens: int = 1200
    caption_max_tokens: int = 800
    check_max_tokens: int = 180
    reasoning_effort: str = "none"
    max_retries: int = 1
    request_timeout_seconds: float = 28.0
    # --- Parallel candidate generation ---
    candidate_count: int = 4
    selector_model: str = ""
    # --- Stage deadlines (seconds) ---
    stage_deadline_perception: float = 12.0
    stage_deadline_candidates: float = 10.0
    stage_deadline_selection: float = 6.0
    # --- Evidence extraction ---
    ocr_enabled: bool = True
    crop_enabled: bool = True
    motion_clip_enabled: bool = True


def _truthy(val: str) -> bool:
    return val.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def load_settings() -> Settings:
    load_dotenv()
    default_model = "accounts/fireworks/models/kimi-k2p6"
    allowed_models = [
        model.strip()
        for model in os.getenv("ALLOWED_MODELS", "").split(",")
        if model.strip()
    ]
    model = (
        os.getenv("FIREWORKS_MODEL")
        or (allowed_models[0] if allowed_models else None)
        or default_model
    )
    caption_model = (
        os.getenv("FIREWORKS_CAPTION_MODEL", "").strip()
        or model
    )
    return Settings(
        api_key=os.getenv("FIREWORKS_API_KEY", ""),
        model=model,
        caption_model=caption_model,
        judge_model=os.getenv("FIREWORKS_JUDGE_MODEL", model),
        base_url=os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"),
        proxy_url=os.getenv("MODEL_PROXY_URL", ""),
        proxy_token=os.getenv("MODEL_PROXY_TOKEN", ""),
        temperature=float(os.getenv("FIREWORKS_TEMPERATURE", "0.2")),
        creative_temperature=float(os.getenv("FIREWORKS_CREATIVE_TEMPERATURE", "0.45")),
        max_tokens=int(os.getenv("FIREWORKS_MAX_TOKENS", "1200")),
        caption_max_tokens=int(os.getenv("FIREWORKS_CAPTION_MAX_TOKENS", "800")),
        check_max_tokens=int(os.getenv("FIREWORKS_CHECK_MAX_TOKENS", "180")),
        reasoning_effort=os.getenv("FIREWORKS_REASONING_EFFORT", "none"),
        max_retries=int(os.getenv("FIREWORKS_MAX_RETRIES", "1")),
        request_timeout_seconds=float(os.getenv("FIREWORKS_REQUEST_TIMEOUT_SECONDS", "28")),
        candidate_count=int(os.getenv("TRACK2_CANDIDATE_COUNT", "4")),
        selector_model=os.getenv("FIREWORKS_SELECTOR_MODEL", "").strip() or caption_model,
        stage_deadline_perception=float(os.getenv("TRACK2_STAGE_DEADLINE_PERCEPTION", "12.0")),
        stage_deadline_candidates=float(os.getenv("TRACK2_STAGE_DEADLINE_CANDIDATES", "10.0")),
        stage_deadline_selection=float(os.getenv("TRACK2_STAGE_DEADLINE_SELECTION", "6.0")),
        ocr_enabled=_truthy(os.getenv("TRACK2_OCR_ENABLED", "true")),
        crop_enabled=_truthy(os.getenv("TRACK2_CROP_ENABLED", "true")),
        motion_clip_enabled=_truthy(os.getenv("TRACK2_MOTION_CLIP_ENABLED", "true")),
    )
