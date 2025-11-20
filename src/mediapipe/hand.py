import cv2
import numpy as np
from mediapipe.python.solutions import hands
from src.utils.ear.constants import HAND_LANDMARKS

class MediaPipeHandsWrapper:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.model = hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    @staticmethod
    def preprocess(image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def infer(self, image: np.ndarray, preprocessed: bool = False):
        if not preprocessed:
            image = self.preprocess(image)
        result = self.model.process(image)
        
        hands_data = []
        if result.multi_hand_landmarks:
            for hand_landmark in result.multi_hand_landmarks:
                # Returns list of lists: [ [(x,y,z)... hand 1], [(x,y,z)... hand 2] ]
                hands_data.append([
                    (lm.x, lm.y, lm.z) for lm in hand_landmark.landmark
                ])
        return hands_data

    def get_landmark(self, single_hand_data, landmark_name: str):
        """
        Fixed logic: Pass a SINGLE hand's data list to this function, 
        not the list of all hands.
        """
        idx = HAND_LANDMARKS.get(landmark_name)
        
        # Safety check: ensure data exists and index is valid
        if idx is not None and single_hand_data and len(single_hand_data) > idx:
            return single_hand_data[idx]
        return None

    def close(self):
        """Release resources explicitly."""
        self.model.close()