import logging
import cv2
import time
from src.utils.ui.visualization import Visualizer
from src.utils.ui.metrics_tracker import FpsTracker, RollingAverage
from src.utils.ear.calculation import EAR, MAR
from src.utils.ear.constants import L_EAR, R_EAR, M_MAR
from src.mediapipe.head_pose import HeadPoseEstimator 
from src.detectors.drowsiness import DrowsinessDetector
from src.detectors.distraction import DistractionDetector
from src.detectors.expression import MouthExpressionClassifier 

log = logging.getLogger(__name__)

class DetectionLoop:
    def __init__(self, camera, face_mesh, buzzer, user_manager, system_logger, vehicle_vin, fps, detector_config_path, initial_user_profile=None, **kwargs):
        self.camera = camera
        self.face_mesh = face_mesh
        self.user_manager = user_manager
        self.logger = system_logger
        self.visualizer = Visualizer()
        
        self.detector = DrowsinessDetector(self.logger, fps, detector_config_path)
        self.distraction_detector = DistractionDetector()
        self.head_pose_estimator = HeadPoseEstimator()
        self.expression_classifier = MouthExpressionClassifier()
        self.ear_calculator = EAR()
        self.mar_calculator = MAR()
        self.fps_tracker = FpsTracker()
        self.ear_smoother = RollingAverage(1.0, fps)
        
        self.user = initial_user_profile
        self.current_mode = 'DETECTING' if initial_user_profile else 'WAITING_FOR_USER'
        self.detector.set_active_user(self.user)
        self._frame_idx = 0

    def run(self):
        while True:
            self.process_frame()
            if cv2.waitKey(1) & 0xFF in (27, ord('q')): break

    def process_frame(self):
        frame = self.camera.read()
        if frame is None: return
        fps = self.fps_tracker.update()
        results = self.face_mesh.process(frame)
        display = frame.copy()

        if self.current_mode == 'DETECTING':
            self._handle_detecting(frame, display, results, fps)
        elif self.current_mode == 'WAITING_FOR_USER':
            self._handle_waiting(frame, display, results)
        
        cv2.imshow("System", cv2.cvtColor(display, cv2.COLOR_RGB2BGR))

    def _handle_detecting(self, frame, display, results, fps):
        if not results.multi_face_landmarks: return
        
        h, w = frame.shape[:2]
        raw_lms = results.multi_face_landmarks[0]
        pose = self.head_pose_estimator.calculate_pose(raw_lms, w, h)
        pitch, yaw, roll = pose if pose else (0,0,0)
        
        # Visual Debug for Head Pose (Helps you sit correctly)
        cv2.putText(display, f"P:{int(pitch)} Y:{int(yaw)}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        # --- MODIFIED: Passing ROLL to the Distraction Detector ---
        looking_away, should_log = self.distraction_detector.analyze(pitch, yaw, roll)
        
        if looking_away:
            status, color = "LOOKING AWAY", (0, 255, 255)
            if should_log:
                status, color = "DISTRACTED!", (0, 0, 255)
                self.logger.alert("distraction")
                self.logger.log_event(self.user.user_id, "DISTRACTION", 2.0, 0.0, frame)
            
            self.visualizer.draw_detection_hud(display, f"User {self.user.user_id}", status, color, fps, 0, 0, 0, "IGNORED", (pitch, yaw, roll))
            return

        # Normal Processing
        lms = [(int(l.x*w), int(l.y*h)) for l in raw_lms.landmark]
        coords = {'left_eye': [lms[i] for i in L_EAR], 'right_eye': [lms[i] for i in R_EAR], 'mouth': [lms[i] for i in M_MAR]}
        
        left = self.ear_calculator.calculate(coords['left_eye'])
        right = self.ear_calculator.calculate(coords['right_eye'])
        mar = self.mar_calculator.calculate(coords['mouth'])
        ear = (left + right) / 2.0
        avg_ear = self.ear_smoother.update(ear)
        
        expr = self.expression_classifier.classify(lms, h)
        self.detector.set_last_frame(frame)
        status, color = self.detector.detect(avg_ear, mar, expr)
        
        self.visualizer.draw_landmarks(display, coords)
        self.visualizer.draw_detection_hud(display, f"User {self.user.user_id}", status, color, fps, avg_ear, mar, 0, expr, (pitch, yaw, roll))

    def _handle_waiting(self, frame, display, results):
        # (Simplified for brevity - matches your existing logic)
        user = self.user_manager.find_best_match(frame)
        if user:
            self.user = user
            self.detector.set_active_user(user)
            self.current_mode = 'DETECTING'
        self.visualizer.draw_no_user_text(display)