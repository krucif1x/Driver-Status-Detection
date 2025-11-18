# http_client.py
"""
Provides robust, generic, and reusable HTTP clients with
retry logic, backoff, and timeouts.

This file should NOT know anything about "Drowsiness Events".
Its only responsibility is to make HTTP requests.

Explanation: This file's only job is to make HTTP requests. It is generic and robust. It knows nothing about "drowsiness"; it only knows how to GET and POST data.

Key Features:

    HttpTriggerClient: A class that holds a connection to the server.

    RetryPolicy: This is what makes your transmission "solid." If the network fails, it automatica
"""
import logging
import time
import random
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union, Protocol
import httpx

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class RetryPolicy:
    retries: int = 3
    backoff_initial_s: float = 0.5
    backoff_factor: float = 2.0
    jitter_s: float = 0.1  # small random jitter

def _normalize_path(url_path: str) -> str:
    if url_path.startswith("http://") or url_path.startswith("https://"):
        return url_path
    path = url_path if url_path.startswith("/") else f"/{url_path}"
    while "//" in path:
        path = path.replace("//", "/")
    return path

class SyncHttpClient(Protocol):
    def post(self, url_path: str, *, json: Optional[Dict[str, Any]] = None, **kw) -> httpx.Response: ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc, tb): ...

class HttpTriggerClient:
    """
    Sync HTTP client wrapper (Keep-Alive, timeout, retries).
    Responsibility: perform HTTP requests with bounded retry/backoff.
    """
    def __init__(
        self,
        base_url: str,
        timeout_s: float = 5.0,
        default_headers: Optional[Dict[str, str]] = None,
        http2: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_s),
            headers=default_headers or {"Content-Type": "application/json"},
            http2=http2,
        )

    def request(
        self,
        method: str,
        url_path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        retry: Union[RetryPolicy, int] = RetryPolicy(),
    ) -> httpx.Response:
        path = _normalize_path(url_path)
        policy = retry if isinstance(retry, RetryPolicy) else RetryPolicy(retries=int(retry))
        delay = policy.backoff_initial_s
        last_exc: Optional[Exception] = None

        for attempt in range(1, policy.retries + 2):
            try:
                resp = self._client.request(
                    method=method.upper(),
                    url=path,
                    params=params,
                    json=json,
                    headers=headers,
                )
                if 200 <= resp.status_code < 300:
                    logger.debug("HTTP %s %s -> %s attempt=%d", method.upper(), path, resp.status_code, attempt)
                    return resp
                last_exc = RuntimeError(f"{resp.status_code} {resp.text[:200]}")
                logger.warning("HTTP %s error attempt=%d status=%d", method.upper(), attempt, resp.status_code)
            except httpx.RequestError as e:
                last_exc = e
                logger.warning("Network error attempt=%d: %s", attempt, e)

            if attempt <= policy.retries:
                time.sleep(delay + random.uniform(0, policy.jitter_s))
                delay *= policy.backoff_factor

        if last_exc:
            raise last_exc
        raise RuntimeError("Unknown HTTP failure")

    def post(self, url_path: str, *, json: Optional[Dict[str, Any]] = None, **kw) -> httpx.Response:
        return self.request("POST", url_path, json=json, **kw)

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    def __enter__(self):
        self._client.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._client.__exit__(exc_type, exc, tb)


# Note: I am omitting the AsyncHttpTriggerClient for brevity,
# but you would include it here if you needed async support.