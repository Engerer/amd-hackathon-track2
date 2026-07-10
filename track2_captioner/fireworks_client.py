from __future__ import annotations

import base64
import json
import logging
import mimetypes
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

RETRY_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


class RetryableHTTPError(Exception):
    """Raised when an HTTP error has a retryable status code."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {detail}")


class FireworksClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        proxy_url: str = "",
        proxy_token: str = "",
        max_retries: int = 1,
        request_timeout_seconds: float = 60.0,
    ) -> None:
        if not api_key and not proxy_url:
            raise ValueError("FIREWORKS_API_KEY or MODEL_PROXY_URL is required unless --dry-run is used.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.proxy_url = proxy_url.rstrip("/")
        self.proxy_token = proxy_token
        self.max_retries = max_retries
        self.request_timeout_seconds = max(5.0, request_timeout_seconds)

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 700,
        temperature: float = 0.2,
        reasoning_effort: str = "none",
        json_mode: bool = False,
    ) -> str:
        import requests

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "reasoning_effort": reasoning_effort,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        if self.proxy_url:
            return self._chat_proxy_with_retry(payload)

        return self._chat_direct_with_retry(requests, payload)

    @staticmethod
    def _content_from_payload(payload: dict[str, Any]) -> str:
        if "content" in payload:
            return str(payload["content"])
        return str(payload["choices"][0]["message"]["content"])

    # ── Proxy path ──────────────────────────────────────────────────────

    def _chat_proxy_with_retry(self, payload: dict[str, Any]) -> str:
        from tenacity import RetryError, before_sleep_log, retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

        @retry(
            stop=stop_after_attempt(self.max_retries + 1),
            wait=wait_exponential_jitter(initial=1, max=32, jitter=2),
            retry=retry_if_exception_type((RetryableHTTPError, TimeoutError, urllib.error.URLError)),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        def _call() -> str:
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
                with urllib.request.urlopen(request, timeout=self.request_timeout_seconds) as response:
                    response_payload = json.loads(response.read().decode("utf-8"))
                return self._content_from_payload(response_payload)
            except urllib.error.HTTPError as exc:
                details = exc.read().decode("utf-8", errors="replace")
                if exc.code in RETRY_STATUS_CODES:
                    raise RetryableHTTPError(exc.code, details) from exc
                raise RuntimeError(f"Proxy request failed with HTTP {exc.code}: {details}") from exc

        try:
            return _call()
        except RetryError as exc:
            raise RuntimeError(f"Proxy request failed after {self.max_retries + 1} attempts") from exc

    # ── Direct Fireworks path ───────────────────────────────────────────

    def _chat_direct_with_retry(self, requests: Any, payload: dict[str, Any]) -> str:
        from tenacity import RetryError, before_sleep_log, retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

        @retry(
            stop=stop_after_attempt(self.max_retries + 1),
            wait=wait_exponential_jitter(initial=1, max=32, jitter=2),
            retry=retry_if_exception_type((RetryableHTTPError, requests.RequestException)),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        def _call() -> str:
            url = f"{self.base_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            response = requests.post(url, headers=headers, json=payload, timeout=self.request_timeout_seconds)
            if response.status_code in RETRY_STATUS_CODES:
                raise RetryableHTTPError(
                    response.status_code,
                    response.text[:500],
                )
            response.raise_for_status()
            return self._content_from_payload(response.json())

        try:
            return _call()
        except RetryError as exc:
            raise RuntimeError(f"Fireworks request failed after {self.max_retries + 1} attempts") from exc


def image_to_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"
