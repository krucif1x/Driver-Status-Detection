"""
Enhanced rule-based classifier for YAWN/SMILE/LAUGH/NEUTRAL with temporal smoothing.
"""
import math
import logging
import numpy as np
from collections import deque
from typing import List, Tuple, Optional

from src.utils.ear.constants import M_MAR, L_EAR, R_EAR

def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

class MouthExpressionClassifier:
    """Rule-based classifier for YAWN/SMILE/LAUGH/NEUTRAL with temporal smoothing."""
    
    # Temporal smoothing
    PERSIST_FRAMES = 6
    EMA_ALPHA = 0.25
    
    # Thresholds (relaxed for real-world use)
    TH_YAWN_ASPECT = 0.9
    TH_YAWN_OPEN = 0.40
    TH_YAWN_AREA = 0.12
    
    TH_SMILE_ASPECT = 0.6
    TH_SMILE_WIDTH = 0.45
    TH_SMILE_OPEN_MAX = 0.35
    TH_SMILE_CURVE = -0.020
    TH_SMILE_AREA_MIN = 0.05
    
    TH_LAUGH_WIDTH = 0.50
    TH_LAUGH_OPEN = 0.28
    TH_LAUGH_VELOCITY = 0.05
    
    DEBUG = False  # Enable for diagnostic logs

    def __init__(self):
        # EMA state
        self._ema = {'open': None, 'width': None, 'aspect': None, 'curve': None, 'area': None, 'cheek': None}
        
        # Velocity tracking
        self._prev = {'open': None, 'width': None}
        self._velocity = {'open': 0.0, 'width': 0.0}
        
        # Temporal smoothing
        self._history = deque(maxlen=self.PERSIST_FRAMES)
        
        # Eye baseline for smile detection
        self._baseline_ear = None
        self._ear_samples = deque(maxlen=30)
        
        self._frame_count = 0

    def _update_ema(self, key: str, value: float) -> float:
        """Update exponential moving average."""
        prev = self._ema[key]
        self._ema[key] = value if prev is None else self.EMA_ALPHA * value + (1.0 - self.EMA_ALPHA) * prev
        return self._ema[key]

    def _normalize_scale(self, landmarks: List[Tuple[int, int]]) -> float:
        """Get normalization scale (eye distance or mouth width)."""
        try:
            # Try eye distance first
            d = _dist(landmarks[L_EAR[0]], landmarks[R_EAR[0]])
            if d > 1e-3:
                return d
        except Exception:
            pass
        
        try:
            # Fallback to mouth width
            d = _dist(landmarks[M_MAR[0]], landmarks[M_MAR[3]])
            return max(d, 1.0)
        except Exception:
            return 1.0

    def _mouth_polygon_area(self, landmarks: List[Tuple[int, int]], scale: float) -> float:
        """Calculate normalized mouth area using shoelace formula."""
        try:
            outer = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
            pts = np.array([landmarks[i] for i in outer if i < len(landmarks)])
            
            if len(pts) < 3:
                return 0.0
            
            x, y = pts[:, 0], pts[:, 1]
            area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            return area / (scale * scale)
        except Exception:
            return 0.0

    def _cheek_activation(self, landmarks: List[Tuple[int, int]], scale: float) -> float:
        """Measure cheek elevation (negative = raised cheeks for smile)."""
        try:
            cheeks = (landmarks[205], landmarks[425])  # left, right
            eyes = (landmarks[33], landmarks[263])  # right, left
            mouth = (landmarks[61], landmarks[291])  # left, right
            
            eye_y = 0.5 * (eyes[0][1] + eyes[1][1])
            mouth_y = 0.5 * (mouth[0][1] + mouth[1][1])
            midline_y = 0.5 * (eye_y + mouth_y)
            
            left_elev = (cheeks[0][1] - midline_y) / scale
            right_elev = (cheeks[1][1] - midline_y) / scale
            
            return 0.5 * (left_elev + right_elev)
        except Exception:
            return 0.0

    def _eye_squint_ratio(self, landmarks: List[Tuple[int, int]]) -> Optional[float]:
        """Calculate eye squint ratio (for smile detection)."""
        try:
            def ear_6pt(indices):
                p = [landmarks[i] for i in indices]
                v1 = _dist(p[1], p[5])
                v2 = _dist(p[2], p[4])
                h = _dist(p[0], p[3])
                return (v1 + v2) / (2.0 * h + 1e-6)
            
            left_ear = ear_6pt(L_EAR[:6])
            right_ear = ear_6pt(R_EAR[:6])
            current_ear = 0.5 * (left_ear + right_ear)
            
            self._ear_samples.append(current_ear)
            if self._baseline_ear is None and len(self._ear_samples) >= 10:
                self._baseline_ear = np.median(self._ear_samples)
            
            if self._baseline_ear and self._baseline_ear > 0:
                return current_ear / self._baseline_ear
            return None
        except Exception:
            return None

    def _extract_features(self, landmarks: List[Tuple[int, int]]) -> dict:
        """Extract mouth features."""
        p61, p82, p87, p291, p317, p312 = (landmarks[i] for i in M_MAR)
        scale = self._normalize_scale(landmarks)

        # Basic metrics
        v1 = _dist(p82, p312) / scale
        v2 = _dist(p87, p317) / scale
        mouth_open = 0.5 * (v1 + v2)
        mouth_width = _dist(p61, p291) / scale
        aspect = mouth_open / (mouth_width + 1e-6)
        
        # Corner curve (negative = smile)
        cx = 0.5 * (p61[0] + p291[0])
        cy = 0.5 * (p61[1] + p291[1])
        corner_curve = 0.5 * ((p61[1] - cy) + (p291[1] - cy)) / max(scale, 1.0)
        
        # Advanced metrics
        mouth_area = self._mouth_polygon_area(landmarks, scale)
        cheek_elev = self._cheek_activation(landmarks, scale)
        eye_ratio = self._eye_squint_ratio(landmarks)
        
        # Velocity tracking
        if self._prev['open'] is not None:
            self._velocity['open'] = abs(mouth_open - self._prev['open'])
            self._velocity['width'] = abs(mouth_width - self._prev['width'])
        self._prev['open'] = mouth_open
        self._prev['width'] = mouth_width
        
        return {
            'open': mouth_open,
            'width': mouth_width,
            'aspect': aspect,
            'curve': corner_curve,
            'area': mouth_area,
            'cheek': cheek_elev,
            'eye_ratio': eye_ratio,
            'vel_open': self._velocity['open'],
            'vel_width': self._velocity['width'],
        }

    def classify(self, landmarks: List[Tuple[int, int]], img_h: int | None = None) -> str:
        """Classify mouth expression: YAWN/SMILE/LAUGH/NEUTRAL."""
        self._frame_count += 1
        
        try:
            feat = self._extract_features(landmarks)
        except Exception as e:
            if self.DEBUG:
                logging.warning(f"[EXPR] Feature extraction failed: {e}")
            self._history.append("NEUTRAL")
            return self._stable_label()

        # Update EMA smoothing
        for key in ['open', 'width', 'aspect', 'curve', 'area', 'cheek']:
            if key in feat:
                self._update_ema(key, feat[key])

        # Decision tree (priority: YAWN > LAUGH > SMILE > NEUTRAL)
        label = "NEUTRAL"
        
        # YAWN: High aspect ratio + large opening OR large area
        if ((self._ema['aspect'] >= self.TH_YAWN_ASPECT and self._ema['open'] >= self.TH_YAWN_OPEN) or
            self._ema['area'] >= self.TH_YAWN_AREA):
            label = "YAWN"
            if self.DEBUG and self._frame_count % 30 == 0:
                logging.info(f"[YAWN] aspect={self._ema['aspect']:.3f}, open={self._ema['open']:.3f}, area={self._ema['area']:.3f}")
        
        # LAUGH: Wide mouth + high opening + velocity
        elif (self._ema['width'] >= self.TH_LAUGH_WIDTH and
              self._ema['open'] >= self.TH_LAUGH_OPEN and
              (feat['vel_open'] >= self.TH_LAUGH_VELOCITY or feat['vel_width'] >= self.TH_LAUGH_VELOCITY)):
            label = "LAUGH"
            if self.DEBUG and self._frame_count % 30 == 0:
                logging.info(f"[LAUGH] width={self._ema['width']:.3f}, open={self._ema['open']:.3f}, vel={feat['vel_open']:.3f}")
        
        # SMILE: Wide mouth + low aspect + upward curve
        elif (self._ema['width'] >= self.TH_SMILE_WIDTH and
              self._ema['aspect'] <= self.TH_SMILE_ASPECT and
              self._ema['open'] <= self.TH_SMILE_OPEN_MAX):
            # Optional bonus checks
            bonus_curve = self._ema['curve'] and self._ema['curve'] <= self.TH_SMILE_CURVE
            bonus_area = self._ema['area'] and self._ema['area'] >= self.TH_SMILE_AREA_MIN
            
            if bonus_curve or bonus_area or self._ema['cheek'] is None:
                label = "SMILE"
                if self.DEBUG and self._frame_count % 30 == 0:
                    logging.info(f"[SMILE] width={self._ema['width']:.3f}, aspect={self._ema['aspect']:.3f}, curve={self._ema['curve']:.3f}")

        # Debug log for NEUTRAL
        if label == "NEUTRAL" and self.DEBUG and self._frame_count % 60 == 0:
            logging.info(f"[NEUTRAL] aspect={self._ema['aspect']:.3f}, width={self._ema['width']:.3f}, open={self._ema['open']:.3f}")

        self._history.append(label)
        return self._stable_label()

    def _stable_label(self) -> str:
        """Return most common label from recent history."""
        if not self._history:
            return "NEUTRAL"
        vals, counts = np.unique(np.array(self._history, dtype=object), return_counts=True)
        return str(vals[int(np.argmax(counts))])

    def reset_baseline(self):
        """Reset eye baseline calibration."""
        self._baseline_ear = None
        self._ear_samples.clear()