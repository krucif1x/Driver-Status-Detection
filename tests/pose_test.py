import sys
import time
import cv2
import numpy as np

# Try to import Picamera2 if available
try:
    from picamera2 import Picamera2
    HAS_PICAM2 = True
except Exception:
    HAS_PICAM2 = False

import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def frames_from_camera():
    """
    Yields RGB frames. Uses Picamera2 if available; otherwise OpenCV VideoCapture(0).
    """
    if HAS_PICAM2:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (1280, 720)})
        picam2.configure(config)
        picam2.start()
        try:
            while True:
                frame_rgb = picam2.capture_array()  # Already RGB
                yield frame_rgb
        finally:
            picam2.stop()
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if not cap.isOpened():
            print("ERROR: Cannot open webcam.")
            return
        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                yield frame_rgb
        finally:
            cap.release()


def main():
    window = "Pose Visualizer"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    prev_t = time.time()
    fps = 0.0

    with mp_pose.Pose(
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        for frame_rgb in frames_from_camera():
            # Inference
            results = pose.process(frame_rgb)

            # Draw landmarks on a copy
            vis = frame_rgb.copy()
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    vis,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
                )

            # FPS
            now = time.time()
            dt = max(1e-6, now - prev_t)
            prev_t = now
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

            cv2.putText(
                vis,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            # Show (convert RGB -> BGR for OpenCV)
            cv2.imshow(window, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()