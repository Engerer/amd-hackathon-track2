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
    creative_temperature: float = 0.75
    max_tokens: int = 1200
    caption_max_tokens: int = 520
    check_max_tokens: int = 180
    reasoning_effort: str = "none"
    max_retries: int = 5


def load_settings() -> Settings:
    load_dotenv()
    allowed_models = [
        model.strip()
        for model in os.getenv("ALLOWED_MODELS", "").split(",")
        if model.strip()
    ]
    model = (
        os.getenv("FIREWORKS_MODEL")
        or (allowed_models[0] if allowed_models else None)
        or "accounts/fireworks/models/qwen3p7-plus"
    )
    caption_model = (
        os.getenv("FIREWORKS_CAPTION_MODEL", "").strip()
        or "accounts/fireworks/models/qwen3p7-plus"
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
        creative_temperature=float(os.getenv("FIREWORKS_CREATIVE_TEMPERATURE", "0.75")),
        max_tokens=int(os.getenv("FIREWORKS_MAX_TOKENS", "1200")),
        caption_max_tokens=int(os.getenv("FIREWORKS_CAPTION_MAX_TOKENS", "520")),
        check_max_tokens=int(os.getenv("FIREWORKS_CHECK_MAX_TOKENS", "180")),
        reasoning_effort=os.getenv("FIREWORKS_REASONING_EFFORT", "none"),
        max_retries=int(os.getenv("FIREWORKS_MAX_RETRIES", "5")),
    )
