import cv2
import numpy as np
import math

from src.utils.ear.constants import HEAD_POSE_IDX

# The 3D Model Points must match the size and order of the HEAD_POSE_IDX list (33 points).
MODEL_POINTS_33 = np.array([
    (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0), (-200.0, 50.0, -120.0), (200.0, 50.0, -120.0),
    (0.0, 100.0, -100.0), (0.0, 200.0, -150.0), (0.0, -200.0, -100.0), (-300.0, 100.0, -150.0),
    (300.0, 100.0, -150.0), (-100.0, -300.0, -50.0), (100.0, -300.0, -50.0), (0.0, 50.0, 0.0),
    (0.0, -50.0, 0.0), (-150.0, 0.0, -150.0), (150.0, 0.0, -150.0), (0.0, -100.0, -100.0),
    (-50.0, 200.0, -150.0), (50.0, 200.0, -150.0), (-250.0, 0.0, -100.0), (250.0, 0.0, -100.0),
    (0.0, 0.0, -200.0), (0.0, 0.0, 20.0), (0.0, 0.0, -20.0), (-350.0, 170.0, -135.0),
    (350.0, 170.0, -135.0), (-200.0, 150.0, -120.0), (200.0, 150.0, -120.0), (-150.0, -50.0, -120.0),
    (150.0, -50.0, -120.0)
], dtype=np.float64)


class HeadPoseEstimator:
    """
    Estimates head orientation (Pitch, Yaw, Roll) using the PnP algorithm.
    Uses 33 points for highly robust estimation.
    """
    def __init__(self):
        # 1. Define the 3D Face Model (Now uses the 33-point array)
        self.model_points = MODEL_POINTS_33 # <--- FIXED

        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1)) 

    def calculate_pose(self, face_landmarks, img_w, img_h):
        """
        Calculates Pitch, Yaw, and Roll from 2D landmarks using robust math.
        """
        # 1. Initialize Camera Matrix (approximate focal length)
        if self.camera_matrix is None:
            focal_length = img_w
            center = (img_w / 2, img_h / 2)
            self.camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype="double"
            )

        # 2. Extract the 2D Image Points from MediaPipe
        # HEAD_POSE_IDX now contains 33 indices, so this extraction is correct.
        image_points = np.array([
            (face_landmarks.landmark[i].x * img_w, face_landmarks.landmark[i].y * img_h)
            for i in HEAD_POSE_IDX
        ], dtype="double")

        # 3. Run PnP Algorithm
        # The number of points now matches (33 vs 33), preventing the crash.
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None

        # 4. Convert Rotation Vector to Rotation Matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # 5. ROBUST ANGLE CALCULATION (Trigonometry on Rotation Matrix)
        
        pitch = math.atan2(rotation_matrix[1, 2], rotation_matrix[2, 2])
        yaw = math.atan2(-rotation_matrix[0, 2], math.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[0, 1]**2))
        roll = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        
        # Convert radians to degrees
        pitch = np.degrees(pitch)
        yaw = np.degrees(yaw)
        roll = np.degrees(roll)

        return pitch, yaw, roll