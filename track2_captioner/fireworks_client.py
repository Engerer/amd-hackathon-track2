from __future__ import annotations

import base64
import json
import logging
import mimetypes
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

try:
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential_jitter,
        before_sleep_log,
        RetryError,
    )
except ImportError:
    class RetryError(Exception):
        pass

    def stop_after_attempt(attempts: int) -> int:
        return attempts

    def wait_exponential_jitter(*args: Any, **kwargs: Any) -> None:
        return None

    def before_sleep_log(*args: Any, **kwargs: Any) -> None:
        return None

    def retry_if_exception_type(exception_types: tuple[type[BaseException], ...] | type[BaseException]) -> tuple[type[BaseException], ...]:
        if isinstance(exception_types, tuple):
            return exception_types
        return (exception_types,)

    def retry(
        stop: int = 1,
        wait: Any = None,
        retry: tuple[type[BaseException], ...] = (Exception,),
        before_sleep: Any = None,
        reraise: bool = True,
    ) -> Any:
        attempts = max(1, int(stop))

        def decorator(func: Any) -> Any:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                last_error: BaseException | None = None
                for attempt in range(attempts):
                    try:
                        return func(*args, **kwargs)
                    except retry as exc:
                        last_error = exc
                        if attempt == attempts - 1:
                            raise
                        time.sleep(min(2 ** attempt, 8))
                if last_error is not None:
                    raise last_error
                return func(*args, **kwargs)

            return wrapper

        return decorator

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
        max_retries: int = 5,
    ) -> None:
        if not api_key and not proxy_url:
            raise ValueError("FIREWORKS_API_KEY or MODEL_PROXY_URL is required unless --dry-run is used.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.proxy_url = proxy_url.rstrip("/")
        self.proxy_token = proxy_token
        self.max_retries = max_retries

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
        @retry(
            stop=stop_after_attempt(self.max_retries),
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
                with urllib.request.urlopen(request, timeout=180) as response:
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
            raise RuntimeError(f"Proxy request failed after {self.max_retries} retries") from exc

    # ── Direct Fireworks path ───────────────────────────────────────────

    def _chat_direct_with_retry(self, requests: Any, payload: dict[str, Any]) -> str:
        @retry(
            stop=stop_after_attempt(self.max_retries),
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
            response = requests.post(url, headers=headers, json=payload, timeout=120)
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
            raise RuntimeError(f"Fireworks request failed after {self.max_retries} retries") from exc


def image_to_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"
