import time
import logging
from collections import deque

log = logging.getLogger(__name__)

class DistractionDetector:
    """
    Detects driver distraction using ABSOLUTE head pose thresholds.
    """
    
    # --- UPDATED THRESHOLDS (More permissive) ---
    # Widened to prevent false positives for normal head movement
    
    YAW_THRESH = 45.0           # Was 40.0. 45 allows checking side mirrors.
    PITCH_DOWN_THRESH = 25.0    # Was 20.0. 25 allows looking at speedometer.
    PITCH_UP_THRESH = 25.0      # Was 20.0.
    ROLL_THRESH = 35.0          # Was 30.0. Allows relaxing neck.
    
    TIME_THRESH = 2.5           
    
    # Defaults (Can be adjusted via keyboard shortcuts in detection_loop)
    EXPECTED_PITCH = 0.0        
    EXPECTED_YAW = 0.0          
    EXPECTED_ROLL = 0.0         
    
    STABILIZATION_FRAMES = 5
    
    # --- NEW HAND ZONES (Normalized 0.0 to 1.0) ---
    # Phone: Upper 50% of screen, Outer 25% of width
    PHONE_ZONE_Y_MAX = 0.5 
    PHONE_ZONE_X_LEFT = 0.25
    PHONE_ZONE_X_RIGHT = 0.75
    
    # Wheel: Bottom 40% of screen
    WHEEL_ZONE_Y_MIN = 0.6

    def __init__(self, camera_pitch_offset: float = 0.0, camera_yaw_offset: float = 0.0):
        self.start_time = None
        self.is_distracted = False
        
        # Apply offsets
        self.EXPECTED_PITCH = camera_pitch_offset
        self.EXPECTED_YAW = camera_yaw_offset
        
        self.history = deque(maxlen=self.STABILIZATION_FRAMES)
        self.frame_count = 0
        self.total_distractions = 0
        
        # New state for hands
        self.is_holding_phone = False
        self.hands_on_wheel = 0
        
        log.info(
            f"DistractionDetector ready (Absolute Thresholds)\n"
            f"  Baseline: P={self.EXPECTED_PITCH:.1f} Y={self.EXPECTED_YAW:.1f}\n"
            f"  Limits: Yaw=±{self.YAW_THRESH} P_Down=+{self.PITCH_DOWN_THRESH} P_Up=-{self.PITCH_UP_THRESH}"
        )

    def _is_valid_pose(self, pitch: float, yaw: float, roll: float) -> bool:
        return (
            abs(pitch) < 90 and 
            abs(yaw) < 90 and 
            abs(roll) < 90 and
            not (pitch == 0 and yaw == 0 and roll == 0)
        )
        
    def detect_hand_distractions(self, hands_data):
        """
        Analyzes hand positions with improved sensitivity.
        """
        phone_detected = False
        hands_on_wheel_count = 0

        # DEBUG: Uncomment this line to prove data is arriving!
        # if hands_data: log.info(f"Hands detected: {len(hands_data)}")

        if not hands_data:
            self.hands_on_wheel = 0
            self.is_holding_phone = False
            return

        for hand_landmarks in hands_data:
            # WRIST = Index 0
            # MIDDLE_FINGER_TIP = Index 12
            wrist = hand_landmarks[0]
            finger = hand_landmarks[12]
            
            # 1. PHONE DETECTION
            # Use FINGER TIP for height check (more sensitive than average)
            # Relaxed X-zone: Check outer 30% (0.3 and 0.7) instead of 25%
            if finger[1] < self.PHONE_ZONE_Y_MAX:
                if finger[0] < 0.30 or finger[0] > 0.70:
                    phone_detected = True
                    # log.info(f"Phone Detected! Y:{finger[1]:.2f} X:{finger[0]:.2f}")

            # 2. WHEEL DETECTION
            # Use WRIST for wheel check (hand can be open or closed)
            if wrist[1] > self.WHEEL_ZONE_Y_MIN:
                hands_on_wheel_count += 1

        self.is_holding_phone = phone_detected
        self.hands_on_wheel = hands_on_wheel_count
    
    def analyze(self, pitch: float, yaw: float, roll: float, hands_data: list = None) -> tuple[bool, bool]:
        self.frame_count += 1
        
        # Run Hand Analysis if data is provided
        if hands_data is not None:
            self.detect_hand_distractions(hands_data)
        
        if not self._is_valid_pose(pitch, yaw, roll):
            self.history.append(False)
            return False, False
        
        # Calculate deviation
        delta_pitch = pitch - self.EXPECTED_PITCH
        delta_yaw = abs(yaw - self.EXPECTED_YAW)
        delta_roll = abs(roll - self.EXPECTED_ROLL)

        is_bad_angle = False
        violations = []
        
        # -- HEAD POSE CHECKS --
        # Check Yaw
        if delta_yaw > self.YAW_THRESH:
            is_bad_angle = True
            violations.append(f"Yaw {delta_yaw:.0f}")
        
        # Check Pitch Down
        if delta_pitch > self.PITCH_DOWN_THRESH:
            is_bad_angle = True
            violations.append(f"Down {delta_pitch:.0f}")
        
        # Check Pitch Up
        if delta_pitch < -self.PITCH_UP_THRESH:
            is_bad_angle = True
            violations.append(f"Up {abs(delta_pitch):.0f}")
        
        # Check Roll
        if delta_roll > self.ROLL_THRESH:
            is_bad_angle = True
            violations.append(f"Roll {delta_roll:.0f}")
            
        # --- HAND DISTRACTION CHECKS ---
        if self.is_holding_phone:
            is_bad_angle = True
            violations.append("HOLDING PHONE")
            
            
        self.history.append(is_bad_angle)
        is_stable_distraction = sum(self.history) >= 3

        if is_stable_distraction:
            if self.start_time is None:
                self.start_time = time.time()
            
            elapsed = time.time() - self.start_time
            
            if elapsed > self.TIME_THRESH:
                if not self.is_distracted:
                    self.is_distracted = True
                    self.total_distractions += 1
                    log.warning(f"🚨 DISTRACTION: {', '.join(violations)}")
                    return True, True  
                return True, False  
            return True, False 
        
        else:
            if self.start_time is not None:
                self.start_time = None
            self.is_distracted = False
            return False, False
    
    def get_status(self, pitch, yaw, roll):
        # Helper for UI Debugging
        
        status = super().get_status(pitch, yaw, roll) if hasattr(super(), 'get_status') else {}
        return {
            'deltas': {'pitch': pitch - self.EXPECTED_PITCH, 'yaw': abs(yaw - self.EXPECTED_YAW), 'roll': abs(roll)},
            'is_distracted': self.is_distracted,
            'distraction_duration': time.time() - self.start_time if self.start_time else 0,
            'total_distractions': self.total_distractions,
            'hands': {
                'holding_phone': self.is_holding_phone,
                'on_wheel_count': self.hands_on_wheel
            }
        }
    
    def adjust_camera_offset(self, pitch_offset: float = None, yaw_offset: float = None):
        if pitch_offset is not None:
            self.EXPECTED_PITCH = pitch_offset
            log.info(f"Camera pitch offset adjusted to {pitch_offset:.1f}")
        if yaw_offset is not None:
            self.EXPECTED_YAW = yaw_offset
            log.info(f"Camera yaw offset adjusted to {yaw_offset:.1f}")
    
    def get_statistics(self):
        return {'total_distractions': self.total_distractions}