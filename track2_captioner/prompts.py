from __future__ import annotations

from pathlib import Path


PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"

STYLE_PROMPTS = {
    "formal": "style_formal.txt",
    "sarcastic": "style_sarcastic.txt",
    "humorous_tech": "style_humorous_tech.txt",
    "humorous_non_tech": "style_humorous_non_tech.txt",
}


def load_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8").strip()
