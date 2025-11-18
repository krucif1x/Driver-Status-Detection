"""
Unified camera interface for Picamera2 or OpenCV webcam.

Environment variables:
- DS_CAMERA_SOURCE: picamera2 | opencv | auto (default: auto)
- DS_CAMERA_INDEX: device index for OpenCV (default: 0)
- DS_CAMERA_RES: resolution like 640x480 (default: 640x480)
"""
import os
import cv2
import time
import logging
import numpy as np
from typing import Optional, Tuple

log = logging.getLogger(__name__)

# Safe Picamera2 import
try:
    from picamera2 import Picamera2
    HAVE_PICAM2 = True
except (ImportError, RuntimeError):
    HAVE_PICAM2 = False
    Picamera2 = None


class Camera:
    """Unified camera with auto-fallback: Picamera2 → OpenCV webcam."""
    
    def __init__(self, source: str = "auto", resolution: Tuple[int, int] = (640, 480)):
        # Parse environment
        self.source = os.getenv("DS_CAMERA_SOURCE", source).lower()
        self.device_index = int(os.getenv("DS_CAMERA_INDEX", "0"))
        
        # Parse resolution from env (e.g., 640x480)
        res_env = os.getenv("DS_CAMERA_RES")
        if res_env and "x" in res_env:
            try:
                w, h = res_env.split("x")
                resolution = (int(w), int(h))
            except ValueError:
                pass
        
        self.resolution = resolution
        self.picam2 = None
        self.cap = None
        self.backend = None
        self.ready = False
        
        # Initialize
        self._init()
        
        if self.ready:
            log.info("✓ Camera ready: %s @ %sx%s", self.backend, *self.resolution)
        else:
            log.error("✗ Camera failed to initialize")
    
    def _init(self):
        """Initialize camera based on source preference."""
        if self.source == "opencv":
            self._init_opencv()
        elif self.source == "picamera2":
            self._init_picamera2()
        else:  # auto
            if HAVE_PICAM2 and self._init_picamera2():
                return
            log.info("Falling back to OpenCV webcam...")
            self._init_opencv()
    
    def _init_picamera2(self) -> bool:
        """Try Picamera2. Returns True if successful."""
        if not HAVE_PICAM2:
            return False
        
        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={"size": self.resolution, "format": "RGB888"}
            )
            self.picam2.configure(config)
            self.picam2.start()
            time.sleep(0.5)  # Warmup
            
            # Verify capture works
            test_frame = self.picam2.capture_array()
            if test_frame is None or test_frame.size == 0:
                raise RuntimeError("Capture test failed")
            
            self.backend = "picamera2"
            self.ready = True
            return True
            
        except Exception as e:
            log.warning("Picamera2 init failed: %s", e)
            if self.picam2:
                try:
                    self.picam2.stop()
                    self.picam2.close()
                except:
                    pass
                self.picam2 = None
            return False
    
    def _init_opencv(self) -> bool:
        """Try OpenCV at the specified device_index."""
        # This line uses the index you set in main.py
        idx_to_try = self.device_index
        
        try:
            cap = cv2.VideoCapture(idx_to_try)
            if not cap.isOpened():
                log.error("OpenCV camera at index %d failed to open.", idx_to_try)
                return False
            
            # Configure
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            time.sleep(0.1)
            
            # Test read
            ret, frame = cap.read()
            if ret and frame is not None:
                self.cap = cap
                self.device_index = idx_to_try
                self.backend = "opencv"
                self.ready = True
                log.info("OpenCV camera ready: /dev/video%d", idx_to_try)
                return True
            
            # If read test fails
            cap.release()
            log.error("OpenCV camera at index %d read test failed.", idx_to_try)
            return False
            
        except Exception as e:
            log.error("Failed to init OpenCV at index %d: %s", idx_to_try, e)
            return False
    
    def read(self) -> Optional[np.ndarray]:
        """Capture frame in RGB format."""
        if not self.ready:
            return None
        
        try:
            if self.backend == "picamera2":
                frame = self.picam2.capture_array()
                # Picamera2 returns BGR despite RGB888 config
                if frame is not None and frame.ndim == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return frame
            
            elif self.backend == "opencv":
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return None
        
        except Exception as e:
            log.debug("Capture error: %s", e)
            return None
    
    def release(self):
        """Release camera resources."""
        if self.picam2:
            try:
                self.picam2.stop()
                self.picam2.close()
            except:
                pass
            self.picam2 = None
        
        if self.cap:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None
        
        self.ready = False
        log.info("Camera released")
    
    def close(self):
        """Alias for release()."""
        self.release()