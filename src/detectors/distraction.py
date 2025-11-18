import time
import logging
from collections import deque

log = logging.getLogger(__name__)

class DistractionDetector:
    """
    Detects if the driver is looking away relative to their RESTING position.
    Uses Pitch, Yaw, and Roll for stable detection and persistence.
    """
    # --- CONFIGURATION (Tuned for Responsiveness and Phone Use) ---
    YAW_THRESH = 30.0
    PITCH_DOWN_THRESH = 16.0
    PITCH_UP_THRESH = 50.0
    ROLL_THRESH = 25.0       # Threshold for head tilt
    
    TIME_THRESH = 1.0        # Seconds before logging as confirmed event

    def __init__(self):
        self.start_time = None
        self.is_distracted = False
        
        # Calibration state
        self.rest_pitch = None
        self.rest_yaw = None
        self.rest_roll = None # <--- NEW: Must calibrate Roll
        
        # Stabilization Buffer (5 frame buffer)
        self.history = deque(maxlen=5)

    def analyze(self, pitch: float, yaw: float, roll: float) -> tuple[bool, bool]: # <--- UPDATED SIGNATURE
        """
        Calculates delta relative to the resting position and detects stable distraction.
        Returns: (is_looking_away_stable, should_log_event)
        """
        # 1. IMMEDIATE AUTO-CALIBRATION
        if self.rest_pitch is None:
            # Calibrate Roll as well
            if abs(pitch) < 180 and abs(yaw) < 180 and abs(roll) < 180: 
                self.rest_pitch = pitch
                self.rest_yaw = yaw
                self.rest_roll = roll # <--- Calibrate Roll
                log.info(f"Distraction Calibrated: Zero set to P={pitch:.1f} Y={yaw:.1f} R={roll:.1f}")
            return False, False

        # 2. Calculate DELTA (Offset from Resting Pose)
        delta_pitch = pitch - self.rest_pitch
        delta_yaw = abs(yaw - self.rest_yaw)
        delta_roll = abs(roll - self.rest_roll) # <--- Calculate Roll Delta

        # 3. Instantaneous Check: True if ANY axis exceeds its threshold.
        is_bad_angle = False
        
        # Check Yaw (Side-to-side)
        if delta_yaw > self.YAW_THRESH:
            is_bad_angle = True
            
        # Check Pitch Down (Phone/Sleep)
        elif delta_pitch > self.PITCH_DOWN_THRESH: 
            is_bad_angle = True
            
        # Check Pitch Up (Yawn/Stretching)
        elif delta_pitch < -self.PITCH_UP_THRESH:
            is_bad_angle = True
            
        # Check Roll (Head Tilt/Lean)
        elif delta_roll > self.ROLL_THRESH: # <--- NEW TRIGGER CHECK
            is_bad_angle = True

        # 4. Stabilization (Voting System)
        self.history.append(is_bad_angle)
        is_stable_distraction = sum(self.history) >= 3

        # 5. Handle Timer & Logging
        if is_stable_distraction:
            # The system stays HERE as long as the threshold is violated
            if self.start_time is None:
                self.start_time = time.time()
            
            elapsed = time.time() - self.start_time
            
            if elapsed > self.TIME_THRESH:
                if not self.is_distracted:
                    self.is_distracted = True
                    return True, True
                return True, False
            
            return True, False
        
        else:
            # Reset (only executed when the head is safely back in range)
            self.start_time = None
            self.is_distracted = False
            return False, False