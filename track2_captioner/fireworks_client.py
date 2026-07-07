from __future__ import annotations

import base64
import json
import mimetypes
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


class FireworksClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        proxy_url: str = "",
        proxy_token: str = "",
    ) -> None:
        if not api_key and not proxy_url:
            raise ValueError("FIREWORKS_API_KEY or MODEL_PROXY_URL is required unless --dry-run is used.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.proxy_url = proxy_url.rstrip("/")
        self.proxy_token = proxy_token

    def chat(self, model: str, messages: list[dict[str, Any]], max_tokens: int = 700) -> str:
        import requests

        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": max_tokens,
            "reasoning_effort": "none",
        }
        if self.proxy_url:
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0 Track2Captioner/1.0",
            }
            if self.proxy_token:
                headers["X-Proxy-Token"] = self.proxy_token
            request = urllib.request.Request(
                self.proxy_url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                details = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Proxy request failed with HTTP {exc.code}: {details}") from exc
            if "content" in payload:
                return str(payload["content"])
            return payload["choices"][0]["message"]["content"]
        else:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=120,
            )
        response.raise_for_status()
        payload = response.json()
        return payload["choices"][0]["message"]["content"]


def image_to_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"
