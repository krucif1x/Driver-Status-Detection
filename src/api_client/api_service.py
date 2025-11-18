# api_service.py
"""
Provides a simple service for interacting with the Drowsiness Detection API.

This file's responsibility is to connect the "data" (event.py)
with the "transport" (http_client.py) and "configuration" (config.py).

Role: This is your Service Layer or "The Coordinator."

Explanation: This file is the "manager" that brings everything together. It's the brains of your client. It knows what data to send (from event.py) and where to send it (from config.py), and it uses the "messenger" (from http_client.py) to do the work.

Key Function: send_drowsiness_event(self, event: DrowsinessEvent)

    payload = event.to_transport_payload(): Build exactly the JSON the server expects.

    response = self.client.post(...): Send it as an HTTP POST to the drowsiness path.

    logger.info(...): Log the outcome.
"""
import logging
import uuid
import requests
from typing import Optional

from .event import DrowsinessEvent
from . import config

log = logging.getLogger(__name__)

class ApiResult:
    def __init__(self, success: bool, status_code: int, text: str, correlation_id: str):
        self.success = success
        self.status_code = status_code
        self.text = text
        self.correlation_id = correlation_id
        self.error: Optional[str] = None

class ApiService:
    def __init__(self, base_url: str = config.SERVER_BASE_URL, timeout: float = config.DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.path = config.DROWSINESS_EVENT_PATH
        log.info("[API] target=%s%s timeout=%.1fs", self.base_url, self.path, self.timeout)

    def _url(self) -> str:
        return f"{self.base_url}{self.path}"

    def send_drowsiness_event(self, event: DrowsinessEvent) -> ApiResult:
        cid = str(uuid.uuid4())
        idem = str(uuid.uuid4())
        headers = {
            "Content-Type": "application/json",
            config.CORRELATION_HEADER: cid,
            config.IDEMPOTENCY_HEADER: idem,
        }
        payload = event.to_transport_payload()
        try:
            resp = requests.post(self._url(), json=payload, headers=headers, timeout=self.timeout)
            ok = 200 <= resp.status_code < 300
            result = ApiResult(ok, resp.status_code, resp.text, cid)
            if not ok:
                result.error = f"HTTP {resp.status_code}"
            return result
        except requests.RequestException as e:
            r = ApiResult(False, 0, "", cid)
            r.error = str(e)
            return r