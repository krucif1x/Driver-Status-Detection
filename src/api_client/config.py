"""
API client configuration (env-friendly) with correct server path.
"""
from typing import Final
import os

SERVER_BASE_URL: Final = os.getenv("DS_REMOTE_URL", "http://203.100.57.59:3000").rstrip("/")
API_VERSION: Final = os.getenv("DS_API_VERSION", "v1")
# Correct path has /api prefix
DROWSINESS_EVENT_PATH: Final = f"/api/{API_VERSION}/drowsiness"

DEFAULT_TIMEOUT: Final = float(os.getenv("DS_REMOTE_TIMEOUT", "10"))
CORRELATION_HEADER: Final = "X-Correlation-Id"
IDEMPOTENCY_HEADER: Final = "X-Idempotency-Key"