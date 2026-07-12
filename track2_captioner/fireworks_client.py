from __future__ import annotations

import base64
import json
import logging
import mimetypes
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable


logger = logging.getLogger(__name__)

RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
MIN_RETRY_REMAINING_SECONDS = 30.0


class RetryableHTTPError(Exception):
    """Raised only for transient HTTP statuses approved for one retry."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {detail}")


class RetryableConnectionError(Exception):
    """Raised for a reset/aborted connection, not ordinary client errors."""


def _looks_like_connection_reset(exc: BaseException) -> bool:
    detail = str(exc).lower()
    return any(
        marker in detail
        for marker in ("connection reset", "connection aborted", "broken pipe")
    )


class FireworksClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        proxy_url: str = "",
        proxy_token: str = "",
        max_attempts: int = 2,
    ) -> None:
        if not api_key and not proxy_url:
            raise ValueError("FIREWORKS_API_KEY or MODEL_PROXY_URL is required unless --dry-run is used.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.proxy_url = proxy_url.rstrip("/")
        self.proxy_token = proxy_token
        self.max_attempts = max(1, max_attempts)

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 700,
        temperature: float = 0.2,
        reasoning_effort: str = "none",
        json_mode: bool = False,
        json_schema: dict[str, Any] | None = None,
        deadline: float | None = None,
    ) -> str:
        import requests

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "reasoning_effort": reasoning_effort,
        }
        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": json_schema},
            }
        elif json_mode:
            payload["response_format"] = {"type": "json_object"}

        if self.proxy_url:
            return self._with_retry(lambda: self._chat_proxy(payload, deadline), deadline)
        return self._with_retry(
            lambda: self._chat_direct(requests, payload, deadline),
            deadline,
        )

    @staticmethod
    def _content_from_payload(payload: dict[str, Any]) -> str:
        if "content" in payload:
            return str(payload["content"])
        return str(payload["choices"][0]["message"]["content"])

    @staticmethod
    def _request_timeout(deadline: float | None) -> float:
        if deadline is None:
            return 25.0
        remaining = deadline - time.monotonic() - MIN_RETRY_REMAINING_SECONDS
        return max(0.1, min(25.0, remaining))

    @staticmethod
    def _has_retry_time(deadline: float | None) -> bool:
        return deadline is None or deadline - time.monotonic() > MIN_RETRY_REMAINING_SECONDS

    def _with_retry(self, call: Callable[[], str], deadline: float | None) -> str:
        for attempt in range(1, self.max_attempts + 1):
            try:
                return call()
            except (
                RetryableHTTPError,
                RetryableConnectionError,
                ConnectionResetError,
                TimeoutError,
                socket.timeout,
            ):
                if attempt >= self.max_attempts or not self._has_retry_time(deadline):
                    raise
                logger.warning(
                    "Transient model request failure; retrying attempt %d/%d.",
                    attempt + 1,
                    self.max_attempts,
                    exc_info=True,
                )
                delay = min(1.0, max(0.0, (deadline - time.monotonic()) if deadline else 1.0))
                if delay:
                    time.sleep(delay)
        raise AssertionError("retry loop exhausted unexpectedly")

    def _chat_proxy(self, payload: dict[str, Any], deadline: float | None) -> str:
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
            with urllib.request.urlopen(request, timeout=self._request_timeout(deadline)) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
            return self._content_from_payload(response_payload)
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            if exc.code in RETRY_STATUS_CODES:
                raise RetryableHTTPError(exc.code, details) from exc
            raise RuntimeError(f"Proxy request failed with HTTP {exc.code}: {details}") from exc
        except urllib.error.URLError as exc:
            reason = exc.reason
            if isinstance(reason, (TimeoutError, socket.timeout)):
                raise TimeoutError(str(exc)) from exc
            if isinstance(reason, ConnectionResetError) or _looks_like_connection_reset(exc):
                raise RetryableConnectionError(str(exc)) from exc
            raise RuntimeError(f"Proxy connection failed: {exc}") from exc

    def _chat_direct(self, requests: Any, payload: dict[str, Any], deadline: float | None) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self._request_timeout(deadline),
            )
        except requests.Timeout as exc:
            raise TimeoutError(str(exc)) from exc
        except requests.ConnectionError as exc:
            if _looks_like_connection_reset(exc):
                raise RetryableConnectionError(str(exc)) from exc
            raise RuntimeError(f"Fireworks connection failed: {exc}") from exc

        if response.status_code in RETRY_STATUS_CODES:
            raise RetryableHTTPError(response.status_code, response.text[:500])
        response.raise_for_status()
        return self._content_from_payload(response.json())


def image_to_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"
