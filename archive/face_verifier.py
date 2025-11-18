import os
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
import cv2
import time

from archive.database import load_user_database

class UserVerifier:
    """
    High-level API for user verification using face encodings.
    """
    def __init__(self, database_path=None, recognition_threshold=0.8):
        """
        Initialize the verifier, load database, and set up models.
        """
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(keep_all=False, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.database_path = database_path
        self.recognition_threshold = recognition_threshold
        self.names, self.ear_thresholds, self.face_encodings = self.load_users()

    def load_users(self):
        """Load users from the database."""
        return load_user_database(self.database_path)

    def get_face_encoding(self, image_frame):
        """
        Extracts a face encoding from an image frame using MTCNN and ResNet.
        Returns None if no face is detected.
        """
        pil_image = Image.fromarray(cv2.cvtColor(image_frame, cv2.COLOR_BGR2RGB))
        img_cropped = self.mtcnn(pil_image)
        if img_cropped is None:
            return None
        with torch.no_grad():
            embedding_tensor = self.resnet(img_cropped.unsqueeze(0).to(self.device))
        return embedding_tensor.cpu().numpy().flatten()

    def verify(self, image_frame):
        """
        Verifies the user from the provided image frame.
        Returns user profile dict if match found, else None.
        """
        if not self.face_encodings:
            print("No users registered.")
            return None

        current_encoding = self.get_face_encoding(image_frame)
        if current_encoding is None:
            print("No face detected.")
            return None

        encodings = [np.array(enc, dtype=np.float32).flatten() for enc in self.face_encodings]
        encodings = [enc for enc in encodings if enc.shape == current_encoding.shape]
        if not encodings:
            print("No valid encodings found for matching.")
            return None

        known_encodings = np.stack(encodings, axis=0)
        distances = np.linalg.norm(known_encodings - current_encoding, axis=1)
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]

        if min_dist < self.recognition_threshold:
            print(f"Match found for {self.names[min_idx]} with distance {min_dist:.2f}")
            return {
                "name": self.names[min_idx],
                "ear_threshold": self.ear_thresholds[min_idx],
                "face_encoding": self.face_encodings[min_idx]
            }

        print("No match found.")
        return None

    def verify_with_camera(self, timeout=10):
        """
        Starts the camera and verifies the user interactively.
        Returns user profile dict if match found, else None.
        """
        try:
            from picamera2 import Picamera2
        except ImportError:
            print("Error: Picamera2 not found. Cannot run camera verification.")
            return None

        print("Starting camera for verification...")
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
        picam2.configure(config)
        picam2.start()
        time.sleep(1.0)
        start_time = time.time()
        user_profile = None

        while time.time() - start_time < timeout:
            frame_rgb = picam2.capture_array()
            feedback_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(feedback_frame, "Look at the camera for verification...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Drowsiness System", feedback_frame)
            if cv2.waitKey(1) == 27:
                print("Verification cancelled by user.")
                break
            user_profile = self.verify(frame_rgb)
            if user_profile is not None:
                print("Face detected and verified.")
                break
        picam2.stop()
        return user_profile

if __name__ == "__main__":
    verifier = UserVerifier()
    # Example: verify from camera
    user = verifier.verify_with_camera()
    if user:
        print(f"User recognized: {user['name']}")
    else:
        print("User not recognized.")