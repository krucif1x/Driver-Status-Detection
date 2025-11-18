"""
Utility for encoding frames to JPEG bytes and Base64 (no file saving).
"""
import cv2
import base64
import logging
from typing import Tuple, Optional

def encode_frame_to_jpeg_and_base64(frame_bgr, quality: int = 85) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Encode an OpenCV BGR frame to JPEG bytes and Base64 string.
    Returns (jpeg_bytes, base64_string) or (None, None) if encoding fails.
    """
    if frame_bgr is None:
        logging.warning("[SCREENSHOT] Cannot encode: frame is None")
        return None, None
    try:
        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            logging.error("[SCREENSHOT] cv2.imencode failed")
            return None, None
        jpeg_bytes = buf.tobytes()
        b64 = base64.b64encode(jpeg_bytes).decode("ascii")
        return jpeg_bytes, b64
    except Exception as e:
        logging.error(f"[SCREENSHOT] Error encoding JPEG: {e}")
        return None, None