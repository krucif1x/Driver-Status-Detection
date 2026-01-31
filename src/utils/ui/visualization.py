import cv2
import numpy as np


class Visualizer:
    """
    A class dedicated to handling all drawing and visualization operations on the
    video frame. This centralizes display logic and keeps the main loop clean.
    """

    def __init__(self):
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX

        self.COLOR_YELLOW = (255, 255, 0)
        self.COLOR_GREEN = (0, 255, 0)
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_ORANGE = (255, 165, 0)
        self.COLOR_CYAN = (0, 255, 255)
        self.COLOR_RED = (255, 0, 0)

    def draw_landmarks(self, image: np.ndarray, coords: dict):
        for key in ["left_eye", "right_eye", "mouth"]:
            for point in coords.get(key, []):
                cv2.circle(image, point, 1, self.COLOR_GREEN, -1)

    def draw_no_user_text(self, image: np.ndarray):
        h, w, _ = image.shape
        text = "Looking for user..."

        (text_width, text_height), _ = cv2.getTextSize(text, self.FONT, 0.9, 2)
        pos_x = (w - text_width) // 2
        pos_y = (h + text_height) // 2

        cv2.putText(image, text, (pos_x, pos_y), self.FONT, 0.9, self.COLOR_YELLOW, 2)

    def draw_face_not_detected(self, image: np.ndarray, user_name: str):
        status_text = "STATUS: FACE NOT DETECTED"
        user_text = f"TRACKING: {user_name}"
        cv2.putText(image, status_text, (10, 30), self.FONT, 0.7, self.COLOR_ORANGE, 2)
        cv2.putText(image, user_text, (10, 60), self.FONT, 0.7, self.COLOR_WHITE, 2)

    def draw_detection_hud(
        self,
        image: np.ndarray,
        user_name: str,
        status: str,
        color: tuple,
        fps: float,
        ear: float,
        mar: float,
        blink_count: int,
        mouth_expression: str = "NEUTRAL",
        pose: tuple = None,
        # hybrid-weighted diagnostics (optional)
        perclos: float = None,
        drowsy_score: float = None,
        score_drowsy: bool = None,
    ):
        h, w, _ = image.shape

        # --- TOP LEFT (User + Status) ---
        cv2.putText(image, f"User: {user_name}", (10, 30), self.FONT, 0.9, self.COLOR_CYAN, 2)
        cv2.putText(image, f"Status: {status}", (10, 70), self.FONT, 0.9, color, 2)

        # --- MIDDLE LEFT (Expression + Head pose) ---
        y_pos = 120
        cv2.putText(image, f"Expr: {mouth_expression}", (10, y_pos), self.FONT, 0.7, self.COLOR_YELLOW, 2)

        if pose:
            pitch, yaw, roll = pose
            y_pos += 30
            cv2.putText(image, f"Pitch: {int(pitch)}", (10, y_pos), self.FONT, 0.6, self.COLOR_WHITE, 1)
            y_pos += 25
            cv2.putText(image, f"Yaw:   {int(yaw)}", (10, y_pos), self.FONT, 0.6, self.COLOR_WHITE, 1)
            y_pos += 25
            cv2.putText(image, f"Roll:  {int(roll)}", (10, y_pos), self.FONT, 0.6, self.COLOR_WHITE, 1)

        # --- BOTTOM LEFT (Biometrics) ---
        ear_text = f"EAR: {ear:.3f}" if ear is not None else "EAR: --"
        mar_text = f"MAR: {mar:.3f}" if mar is not None else "MAR: --"
        blink_text = f"Blinks: {int(blink_count)}"

        y = h - 10
        cv2.putText(image, blink_text, (10, y), self.FONT, 0.7, self.COLOR_WHITE, 2)
        y -= 30
        cv2.putText(image, mar_text, (10, y), self.FONT, 0.7, self.COLOR_YELLOW, 2)
        y -= 30
        cv2.putText(image, ear_text, (10, y), self.FONT, 0.7, self.COLOR_YELLOW, 2)

        # NEW: weighted diagnostics (only if provided)
        if drowsy_score is not None:
            y -= 30
            tag = "ON" if bool(score_drowsy) else "OFF"
            cv2.putText(
                image,
                f"DScore: {float(drowsy_score):.2f} ({tag})",
                (10, y),
                self.FONT,
                0.7,
                self.COLOR_ORANGE,
                2,
            )

        if perclos is not None:
            y -= 30
            cv2.putText(
                image,
                f"PERCLOS: {float(perclos):.2f}",
                (10, y),
                self.FONT,
                0.7,
                self.COLOR_ORANGE,
                2,
            )

        # --- TOP RIGHT (FPS) ---
        fps_text = f"FPS: {fps:.2f}"
        (text_width, _), _ = cv2.getTextSize(fps_text, self.FONT, 0.7, 2)
        cv2.putText(image, fps_text, (w - text_width - 10, 30), self.FONT, 0.7, self.COLOR_GREEN, 2)

    def draw_no_face_text(self, display):
        h, w = display.shape[:2]
        cv2.putText(
            display,
            "NO FACE DETECTED",
            (max(10, w // 2 - 150), h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )

    def draw_mode(self, display, mode: str):
        cv2.putText(
            display,
            f"MODE: {mode}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            1,
        )