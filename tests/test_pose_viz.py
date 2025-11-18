import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import time
import logging

# --- Assumed Imports based on your project structure ---
from src.infrastructure.hardware.camera import Camera
from src.mediapipe.mediapipe_wrapper import MediaPipeFaceModel
from src.mediapipe.head_pose import HeadPoseEstimator 
from src.utils.ui.visualization import Visualizer
from src.utils.ui.metrics_tracker import FpsTracker, RollingAverage

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def test_pose_visualization():
    """Initializes components and runs a dedicated loop for head pose testing."""
    log.info("Starting Head Pose Visualization Test...")
    
    # 1. Component Initialization
    try:
        camera = Camera(source='auto', resolution=(640, 480))
        face_mesh = MediaPipeFaceModel(max_num_faces=1, refine_landmarks=True)
        pose_estimator = HeadPoseEstimator()
        visualizer = Visualizer()
        
        if not camera.ready:
             raise RuntimeError("Camera failed to initialize.")
             
    except Exception as e:
        log.error(f"Initialization failed: {e}")
        return

    fps_tracker = FpsTracker()
    pitch_avg = RollingAverage(duration_sec=2.0)
    yaw_avg = RollingAverage(duration_sec=2.0)
    roll_avg = RollingAverage(duration_sec=2.0)

    # 2. Main Test Loop
    while True:
        frame_bgr = camera.read()
        if frame_bgr is None:
            continue
        
        # Frame is BGR, MediaPipe needs RGB
        results = face_mesh.process(frame_bgr)
        display_frame = frame_bgr.copy() # Use BGR copy for drawing (OpenCV works best on its native format)

        # Update FPS using metrics_tracker
        fps = fps_tracker.update()
        
        pitch, yaw, roll = 0, 0, 0
        status = "NO FACE"
        
        if results.multi_face_landmarks:
            raw_landmarks = results.multi_face_landmarks[0]
            h, w = display_frame.shape[:2]

            # 3. Calculate Pose
            pose = pose_estimator.calculate_pose(raw_landmarks, w, h)
            
            if pose:
                pitch, yaw, roll = pose
                status = "DETECTING"

                # Update rolling averages
                pitch_smooth = pitch_avg.update(pitch)
                yaw_smooth = yaw_avg.update(yaw)
                roll_smooth = roll_avg.update(roll)
                
                # 4. Draw Debugging Values
                visualizer.draw_detection_hud(
                    display_frame,
                    user_name="TEST",
                    status=status,
                    color=(0, 255, 0), # Green
                    fps=fps,
                    ear=0.3,
                    mar=0.1,
                    blink_count=0,
                    mouth_expression="NEUTRAL",
                    pose=(pitch_smooth, yaw_smooth, roll_smooth) # <-- Smoothed visualization data
                )
                
                # Print stable angles to console for numerical debugging
                if int(fps) % 5 == 0:
                     log.info(f"P: {pitch_smooth:.1f}, Y: {yaw_smooth:.1f}, R: {roll_smooth:.1f}")

        # 5. Display Result
        cv2.imshow("Head Pose Visualization Test", display_frame)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break

    # 6. Cleanup
    camera.release()
    face_mesh.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_pose_visualization()