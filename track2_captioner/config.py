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
    rerank_model: str = "accounts/fireworks/models/glm-5p2"
    judge_model: str = "accounts/fireworks/models/glm-5p2"
    caption_models: tuple[str, ...] = ()
    base_url: str = "https://api.fireworks.ai/inference/v1"
    proxy_url: str = ""
    proxy_token: str = ""
    # --- Generation parameters ---
    temperature: float = 0.2
    creative_temperature: float = 0.6
    max_tokens: int = 1200
    caption_max_tokens: int = 700
    caption_candidates: int = 1
    rerank_max_tokens: int = 700
    check_max_tokens: int = 180
    reasoning_effort: str = "none"
    max_retries: int = 5


def _split_models(value: str) -> tuple[str, ...]:
    models = [model.strip() for model in value.split(",") if model.strip()]
    return tuple(dict.fromkeys(models))


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
        or "accounts/fireworks/models/kimi-k2p6"
    )
    default_caption_models = "accounts/fireworks/models/deepseek-v4-pro,accounts/fireworks/models/glm-5p2"
    caption_models = _split_models(os.getenv("FIREWORKS_CAPTION_MODELS", "")) or _split_models(default_caption_models)
    caption_model = os.getenv("FIREWORKS_CAPTION_MODEL", "").strip() or caption_models[0]
    if caption_model not in caption_models:
        caption_models = (caption_model, *caption_models)

    rerank_model = (
        os.getenv("FIREWORKS_RERANK_MODEL", "").strip()
        or "accounts/fireworks/models/glm-5p2"
    )
    return Settings(
        api_key=os.getenv("FIREWORKS_API_KEY", ""),
        model=model,
        caption_model=caption_model,
        caption_models=caption_models,
        rerank_model=rerank_model,
        judge_model=os.getenv("FIREWORKS_JUDGE_MODEL", rerank_model),
        base_url=os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"),
        proxy_url=os.getenv("MODEL_PROXY_URL", ""),
        proxy_token=os.getenv("MODEL_PROXY_TOKEN", ""),
        temperature=float(os.getenv("FIREWORKS_TEMPERATURE", "0.2")),
        creative_temperature=float(os.getenv("FIREWORKS_CREATIVE_TEMPERATURE", "0.6")),
        max_tokens=int(os.getenv("FIREWORKS_MAX_TOKENS", "1200")),
        caption_max_tokens=int(os.getenv("FIREWORKS_CAPTION_MAX_TOKENS", "700")),
        caption_candidates=max(1, int(os.getenv("FIREWORKS_CAPTION_CANDIDATES", "1"))),
        rerank_max_tokens=int(os.getenv("FIREWORKS_RERANK_MAX_TOKENS", "700")),
        check_max_tokens=int(os.getenv("FIREWORKS_CHECK_MAX_TOKENS", "180")),
        reasoning_effort=os.getenv("FIREWORKS_REASONING_EFFORT", "none"),
        max_retries=int(os.getenv("FIREWORKS_MAX_RETRIES", "5")),
    )
