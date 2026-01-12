from __future__ import annotations

import os
import time

import cv2
from picamera2 import Picamera2


def _has_gui() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def main() -> int:
    try:
        picam2 = Picamera2()
    except RuntimeError as e:
        message = str(e)
        if "Device or resource busy" in message or "Pipeline handler in use by another process" in message:
            print("Camera is busy (already in use by another process).")
            print("Common fixes:")
            print("  - Stop this app if it's running: `bash stop.sh` or `sudo systemctl stop driver-status-detection.service`")
            print("  - Close anything using the camera (browser, Teams/Zoom, etc.)")
            print("  - Find holders: `lsof /dev/media0 /dev/media1` or `fuser -v /dev/media0 /dev/media1`")
        raise

    try:
        picam2.configure(
            picam2.create_preview_configuration(main={"format": "XRGB8888", "size": (640, 480)})
        )
        picam2.start()
        time.sleep(1.0)  # allow camera to warm up

        if not _has_gui():
            # Headless mode: grab a few frames and write one to disk.
            frame = None
            for _ in range(30):
                frame = picam2.capture_array()
                if frame is not None:
                    break
            if frame is None:
                raise RuntimeError("Failed to capture any frames from Picamera2")
            out_path = os.path.join(os.path.dirname(__file__), "raspicam_frame.jpg")
            cv2.imwrite(out_path, frame)
            print(f"No GUI detected; wrote a test frame to: {out_path}")
            return 0

        while True:
            frame = picam2.capture_array()
            if frame is None:
                continue
            cv2.imshow("Raspberry Pi Camera Feed", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        try:
            picam2.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())