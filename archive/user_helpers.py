import cv2
import time
from src.pipeline.calibration.ear_calibrator import EARCalibrator

def verify_or_register_user(camera, face_mesh, user_manager):
    print("--- User Verification ---")
    frame = camera.capture_frame()
    if frame is None:
        print("Error: Failed to capture frame for user check.")
        camera.stop()
        return None
    initial_frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR) if frame.shape[2] == 4 else frame.copy()
    recognized_user = user_manager.find_best_match(initial_frame)
    if not recognized_user:
        recognized_user = calibrate_and_register(camera, face_mesh, user_manager)
    else:
        print(f"Welcome back, {recognized_user.name}!")
    return recognized_user

def calibrate_and_register(camera, face_mesh, user_manager):
    print("User not recognized. Starting calibration to create a new profile.")
    ear_calibrator = EARCalibrator(camera, face_mesh, user_manager)
    new_ear_threshold = ear_calibrator.calibrate()
    while new_ear_threshold is None:
        print("Calibration stopped or failed. Waiting for user to enter frame...")
        while True:
            frame = camera.capture_frame()
            if frame is None:
                continue
            live_frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR) if frame.shape[2] == 4 else frame.copy()
            cv2.putText(live_frame, "Waiting for user to enter frame...", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Drowsiness System", live_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                print("ESC pressed. Exiting...")
                camera.stop()
                return None
            ear_calibrator = EARCalibrator(camera, face_mesh, user_manager)
            new_ear_threshold = ear_calibrator.calibrate()
            if new_ear_threshold is not None:
                break
    print("Calibration successful! Registering new user.")
    time.sleep(2.0)
    frame = camera.capture_frame()
    registration_frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR) if frame.shape[2] == 4 else frame.copy()
    new_user_profile = user_manager.register_new_user(registration_frame, new_ear_threshold)
    if new_user_profile is None:
        print("Registration failed (no face detected). Exiting.")
        camera.stop()
        return None
    print(f"New user registered: {new_user_profile.name}")
    return new_user_profile