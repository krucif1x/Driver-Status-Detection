"""
HEAD POSE ESTIMATOR - YOUR WORKING VERSION + CAMERA SPECS
Minimal changes to your original working code.
"""

import cv2
import numpy as np
import logging
from src.utils.landmarks.constants import MODEL_POINTS, LANDMARK_INDICES

logger = logging.getLogger(__name__)

class HeadPoseEstimator:
    """Simple head pose estimator with camera specs."""

    def __init__(self, camera_specs=None):
        """
        Args:
            camera_specs: Optional dict with 'focal_mm', 'sensor_w_mm', 'sensor_h_mm'
                         If None, uses simple focal_length = img_w approximation
        """
        self.model_points = MODEL_POINTS
        self.landmark_indices = LANDMARK_INDICES
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1))

        # Camera specs (optional - if None, will use simple approximation)
        self.camera_specs = camera_specs or {
            "focal_mm": 4.74,
            "sensor_w_mm": 6.45,
            "sensor_h_mm": 3.63
        }
        self.use_camera_specs = camera_specs is not None

        # PnP state
        self.rvec = None
        self.tvec = None

        # Smoothing state
        self.prev_pitch = 0.0
        self.prev_yaw = 0.0
        self.prev_roll = 0.0
        self.first_frame = True

        # Smoothing parameters
        self.DEADZONE_THRESH = 0.5
        self.ALPHA_PITCH = 0.3
        self.ALPHA_YAW = 0.5
        self.ALPHA_ROLL = 0.3

        logger.info("HeadPoseEstimator initialized")

    def _unwrap_angle(self, prev_deg: float, curr_deg: float, period: float = 180.0, threshold: float = 90.0) -> float:
        """
        Enforce continuity by shifting curr_deg by ±period to minimize jump vs prev_deg.
        Use period=180 because RQDecomp can switch branches by ~180 deg.
        """
        delta = curr_deg - prev_deg
        if delta > threshold:
            curr_deg -= period
        elif delta < -threshold:
            curr_deg += period
        return curr_deg

    def calculate_pose(self, face_landmarks, img_w, img_h):
        """Calculate head pose angles."""
        try:
            # Initialize camera matrix
            if self.camera_matrix is None:
                if self.use_camera_specs:
                    # Use accurate camera specs
                    focal_length_x = (self.camera_specs["focal_mm"] / self.camera_specs["sensor_w_mm"]) * img_w
                    focal_length_y = (self.camera_specs["focal_mm"] / self.camera_specs["sensor_h_mm"]) * img_h
                    center = (img_w / 2, img_h / 2)
                    self.camera_matrix = np.array([
                        [focal_length_x, 0, center[0]],
                        [0, focal_length_y, center[1]],
                        [0, 0, 1]
                    ], dtype=np.float64)
                    logger.info(f"Camera matrix: fx={focal_length_x:.2f}, fy={focal_length_y:.2f}")
                else:
                    # Simple approximation (your original)
                    focal_length = img_w
                    center = (img_w / 2, img_h / 2)
                    self.camera_matrix = np.array([
                        [focal_length, 0, center[0]],
                        [0, focal_length, center[1]],
                        [0, 0, 1]
                    ], dtype=np.float64)

            # Get 2D image points
            image_points = np.array([
                (face_landmarks.landmark[i].x * img_w,
                 face_landmarks.landmark[i].y * img_h)
                for i in self.landmark_indices
            ], dtype=np.float64)

            # Solve PnP
            if self.rvec is None:
                success, self.rvec, self.tvec = cv2.solvePnP(
                    self.model_points, image_points,
                    self.camera_matrix, self.dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
            else:
                success, self.rvec, self.tvec = cv2.solvePnP(
                    self.model_points, image_points,
                    self.camera_matrix, self.dist_coeffs,
                    rvec=self.rvec, tvec=self.tvec,
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )

            if not success:
                return (self.prev_pitch, self.prev_yaw, self.prev_roll)

            # Convert to rotation matrix
            rmat, _ = cv2.Rodrigues(self.rvec)

            # Extract Euler angles using OpenCV
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

            pitch_raw = angles[0]
            yaw_raw = angles[1]
            roll_raw = angles[2]

            # Normalize roll to [-90, +90] range
            if roll_raw > 90:
                roll_raw = roll_raw - 180
            elif roll_raw < -90:
                roll_raw = roll_raw + 180

            # First frame: no smoothing
            if self.first_frame:
                self.prev_pitch = pitch_raw
                self.prev_yaw = yaw_raw
                self.prev_roll = roll_raw
                self.first_frame = False
                return (pitch_raw, yaw_raw, roll_raw)

            # Apply deadzone (ignore small jitter)
            if abs(pitch_raw - self.prev_pitch) < self.DEADZONE_THRESH:
                pitch_raw = self.prev_pitch
            if abs(yaw_raw - self.prev_yaw) < self.DEADZONE_THRESH:
                yaw_raw = self.prev_yaw
            if abs(roll_raw - self.prev_roll) < self.DEADZONE_THRESH:
                roll_raw = self.prev_roll

            # Clamp to prevent impossible angles
            pitch_raw = max(-90, min(90, pitch_raw))
            yaw_raw = max(-90, min(90, yaw_raw))
            roll_raw = max(-90, min(90, roll_raw))

            # Apply exponential moving average
            smooth_pitch = (self.ALPHA_PITCH * pitch_raw) + ((1 - self.ALPHA_PITCH) * self.prev_pitch)
            smooth_yaw = (self.ALPHA_YAW * yaw_raw) + ((1 - self.ALPHA_YAW) * self.prev_yaw)
            smooth_roll = (self.ALPHA_ROLL * roll_raw) + ((1 - self.ALPHA_ROLL) * self.prev_roll)

            # Update state
            self.prev_pitch = smooth_pitch
            self.prev_yaw = smooth_yaw
            self.prev_roll = smooth_roll

            return (smooth_pitch, smooth_yaw, smooth_roll)

        except Exception as e:
            logger.error(f"Pose calculation error: {e}")
            return (self.prev_pitch, self.prev_yaw, self.prev_roll)

    def reset(self):
        """Reset estimator state."""
        self.rvec = None
        self.tvec = None
        self.prev_pitch = 0.0
        self.prev_yaw = 0.0
        self.prev_roll = 0.0
        self.first_frame = True
        logger.info("Head pose estimator reset")


def _put_hud(image_bgr: np.ndarray, lines: list[str]) -> None:
    y = 22
    for line in lines:
        cv2.putText(image_bgr, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image_bgr, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22


if __name__ == "__main__":
    import time

    from src.infrastructure.hardware.camera import Camera
    from src.mediapipe.face_mesh import FaceMeshModel

    cam = Camera(source="auto", resolution=(640, 480))
    if not getattr(cam, "ready", True):
        raise SystemExit("Camera failed to initialize.")

    face_mesh = FaceMeshModel(max_num_faces=1, refine_landmarks=True)
    estimator = HeadPoseEstimator(camera_specs=None)  # set dict to enable camera-spec focal lengths

    window = "HeadPoseEstimator - live test (q/esc quit)"
    fps_ema = 0.0
    alpha = 0.1

    try:
        while True:
            frame_rgb = cam.read(color="rgb")
            if frame_rgb is None:
                continue

            t0 = time.perf_counter()
            results = face_mesh.process(frame_rgb)

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            faces = getattr(results, "multi_face_landmarks", None) or []
            if faces:
                h, w = frame_rgb.shape[:2]
                pitch, yaw, roll = estimator.calculate_pose(faces[0], w, h)
                pose_line = f"pitch={pitch:0.1f}  yaw={yaw:0.1f}  roll={roll:0.1f}"
            else:
                pose_line = "no face"

            dt = max(1e-6, time.perf_counter() - t0)
            fps = 1.0 / dt
            fps_ema = fps if fps_ema <= 0 else (1 - alpha) * fps_ema + alpha * fps

            _put_hud(frame_bgr, [pose_line, f"fps: {fps_ema:0.1f}"])
            cv2.imshow(window, frame_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        try:
            face_mesh.close()
        except Exception:
            pass
        try:
            cam.release()
        except Exception:
            pass
        cv2.destroyAllWindows()