from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class EyeCfg:
    all: List[int]
    ear: List[int]
    frames: int

@dataclass(frozen=True)
class MouthCfg:
    mar: List[int]
    outline: List[int]
    thresh: float
    frames: int

@dataclass(frozen=True)
class CalibCfg:
    dur: int
    factor: float

@dataclass(frozen=True)
class BufCfg:
    dur: float

@dataclass(frozen=True)
class BlinkCfg:
    cnt: int
    closed: bool

@dataclass
class FpsCfg:
    prev: float
    fps: float

# Landmark indexes for eyes and mouth (for EAR/MAR)
L_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
R_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
# Standard MediaPipe indices for EAR calculation (order matters!)
L_EAR = [263, 387, 385, 362, 380, 373]   # Left eye: [P1, P2, P3, P4, P5, P6]
R_EAR = [33, 160, 158, 133, 153, 144]    # Right eye: [P1, P2, P3, P4, P5, P6]
M_MAR = [61, 82, 312, 291, 317, 87]
M_OUT = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]

# Face mesh landmark indexes (from face_landmarks.json)
SILHOUETTE = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]

LIPS_UPPER_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LIPS_LOWER_OUTER = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
LIPS_UPPER_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
LIPS_LOWER_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

RIGHT_EYE_UPPER0 = [246, 161, 160, 159, 158, 157, 173]
RIGHT_EYE_LOWER0 = [33, 7, 163, 144, 145, 153, 154, 155, 133]
RIGHT_EYE_UPPER1 = [247, 30, 29, 27, 28, 56, 190]
RIGHT_EYE_LOWER1 = [130, 25, 110, 24, 23, 22, 26, 112, 243]
RIGHT_EYE_UPPER2 = [113, 225, 224, 223, 222, 221, 189]
RIGHT_EYE_LOWER2 = [226, 31, 228, 229, 230, 231, 232, 233, 244]
RIGHT_EYE_LOWER3 = [143, 111, 117, 118, 119, 120, 121, 128, 245]

RIGHT_EYEBROW_UPPER = [156, 70, 63, 105, 66, 107, 55, 193]
RIGHT_EYEBROW_LOWER = [35, 124, 46, 53, 52, 65]

RIGHT_EYE_IRIS = [473, 474, 475, 476, 477]

LEFT_EYE_UPPER0 = [466, 388, 387, 386, 385, 384, 398]
LEFT_EYE_LOWER0 = [263, 249, 390, 373, 374, 380, 381, 382, 362]
LEFT_EYE_UPPER1 = [467, 260, 259, 257, 258, 286, 414]
LEFT_EYE_LOWER1 = [359, 255, 339, 254, 253, 252, 256, 341, 463]
LEFT_EYE_UPPER2 = [342, 445, 444, 443, 442, 441, 413]
LEFT_EYE_LOWER2 = [446, 261, 448, 449, 450, 451, 452, 453, 464]
LEFT_EYE_LOWER3 = [372, 340, 346, 347, 348, 349, 350, 357, 465]

LEFT_EYEBROW_UPPER = [383, 300, 293, 334, 296, 336, 285, 417]
LEFT_EYEBROW_LOWER = [265, 353, 276, 283, 282, 295]

LEFT_EYE_IRIS = [468, 469, 470, 471, 472]

MIDWAY_BETWEEN_EYES = [168]
NOSE_TIP = [1]
NOSE_BOTTOM = [2]
NOSE_RIGHT_CORNER = [98]
NOSE_LEFT_CORNER = [327]
RIGHT_CHEEK = [205]
LEFT_CHEEK = [425]

# Indices for PnP Head Pose Estimation
# Order: [Nose Tip, Chin, Left Eye Corner, Right Eye Corner, Left Mouth Corner, Right Mouth Corner]
# HEAD_POSE_IDX = [1, 152, 263, 33, 61, 291]
HEAD_POSE_IDX = [1, 9, 10, 33, 50, 54, 61, 84, 93, 103, 117, 133, 
    145, 150, 153, 162, 172, 181, 234, 263, 283, 284, 291, 312, 
    327, 338, 350, 356, 361, 373, 405, 425, 466
]

HAND_LANDMARKS = {
    "WRIST": 0,
    "THUMB_CMC": 1,
    "THUMB_MCP": 2,
    "THUMB_IP": 3,
    "THUMB_TIP": 4,
    "INDEX_FINGER_MCP": 5,
    "INDEX_FINGER_PIP": 6,
    "INDEX_FINGER_DIP": 7,
    "INDEX_FINGER_TIP": 8,
    "MIDDLE_FINGER_MCP": 9,
    "MIDDLE_FINGER_PIP": 10,
    "MIDDLE_FINGER_DIP": 11,
    "MIDDLE_FINGER_TIP": 12,
    "RING_FINGER_MCP": 13,
    "RING_FINGER_PIP": 14,
    "RING_FINGER_DIP": 15,
    "RING_FINGER_TIP": 16,
    "PINKY_MCP": 17,
    "PINKY_PIP": 18,
    "PINKY_DIP": 19,
    "PINKY_TIP": 20
}

# Configuration constants
CONSEC_FRAMES = 30
MAR_THRES = 0.23  # or 0.27, depending on your data
CALIBRATION_DURATION = 20     # seconds
DROWSINESS_FACTOR = 0.8
BUFFER_DURATION = 1.0          # seconds    

# Config instances
LEFT_EYE = EyeCfg(L_EYE, L_EAR, CONSEC_FRAMES)
RIGHT_EYE = EyeCfg(R_EYE, R_EAR, CONSEC_FRAMES)
MOUTH = MouthCfg(M_MAR, M_OUT, MAR_THRES, CONSEC_FRAMES)
CALIB = CalibCfg(CALIBRATION_DURATION, DROWSINESS_FACTOR)
BUF = BufCfg(BUFFER_DURATION)
BLINK = BlinkCfg(0, False)
FPS = FpsCfg(0, 0)