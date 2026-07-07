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
    judge_model: str
    base_url: str = "https://api.fireworks.ai/inference/v1"
    proxy_url: str = ""
    proxy_token: str = ""


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
        or "accounts/fireworks/models/kimi-k2p5"
    )
    return Settings(
        api_key=os.getenv("FIREWORKS_API_KEY", ""),
        model=model,
        judge_model=os.getenv("FIREWORKS_JUDGE_MODEL", model),
        base_url=os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"),
        proxy_url=os.getenv("MODEL_PROXY_URL", ""),
        proxy_token=os.getenv("MODEL_PROXY_TOKEN", ""),
    )
