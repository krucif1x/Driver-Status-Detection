import numpy as np
from typing import Optional, Tuple, Union
from queue import Queue

from src.utils.ear.calculation import EAR
from src.services.user_manager import UserManager
from src.detectors.ear_calibration.calibration import calibrate
from src.utils.user.thread import start_user_check_thread, stop_user_check_thread
from src.utils.landmark.processor import landmark_processor
from src.utils.ear.average_ear import average_ear
from src.utils.ui.feedback import feedback

class EARCalibrator:
    """
    Optimized EAR calibration with efficient processing and robust filtering.
    """
    CALIBRATION_DURATION_S = 10
    FACE_LOST_TIMEOUT_S = 3.5
    MIN_VALID_SAMPLES = 20
    
    # Performance optimizations
    USER_CHECK_INTERVAL = 15      # Check for existing users every 15 frames (not every frame)
    DISPLAY_UPDATE_INTERVAL = 3   # Update display every 3 frames (reduces overhead)
    EAR_BOUNDS = (0.06, 0.60)     # Valid EAR range
    STABILITY_WINDOW = 20         # Rolling window for stability check
    
    # Pre-allocated buffers for efficiency
    PREALLOCATE_SIZE = 300        # ~10s at 30fps

    def __init__(self, camera, face_mesh, user_manager: UserManager):
        self.camera = camera  # Camera instance from camera.py
        self.face_mesh = face_mesh
        self.user_manager = user_manager
        self.ear_calculator = EAR()
        
        # Pre-allocate array for EAR values (avoid dynamic resizing)
        self._ear_buffer = np.zeros(self.PREALLOCATE_SIZE, dtype=np.float32)
        self._ear_count = 0

        # Background user recognition
        self._user_check_queue = Queue(maxsize=1)
        self._user_check_result = None
        self._user_check_thread = None

    def calibrate(self) -> Union[None, float, Tuple[str, object]]:
        """Run calibration and return result."""
        return calibrate(self)
    
    def manage_user_check_thread(self, start=True):
        """Start or stop background user recognition thread."""
        if start:
            start_user_check_thread(self)
        else:
            stop_user_check_thread(self)

    def _process_landmarks_optimized(self, results, frame_shape):
        """Process MediaPipe landmarks efficiently."""
        return landmark_processor(self, results, frame_shape)
    
    def average_ear(self) -> Optional[float]:
        """Calculate robust average EAR from collected samples."""
        return average_ear(self)
    
    def feedback(self, frame, ear, elapsed, status_msg, num_samples):
        """Display calibration feedback on frame."""
        feedback(self, frame, ear, elapsed, status_msg, num_samples)
