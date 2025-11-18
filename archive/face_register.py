import os
import numpy as np
from datetime import datetime
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import cv2

from archive.database import load_user_database, save_user_database

class UserRegistrar:
    """
    High-level API for user registration and duplicate checking using face encodings.
    """
    def __init__(self, database_path=None, duplicate_threshold=0.7):
        """
        Initialize the registrar, load database, and set up models.
        """
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(keep_all=False, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.database_path = database_path
        self.duplicate_threshold = duplicate_threshold
        try:
            self.names, self.ear_thresholds, self.face_encodings = self.load_users()
        except Exception as e:
            print(f"Error loading user database: {e}")
            self.names, self.ear_thresholds, self.face_encodings = [], [], []

    def load_users(self):
        """Load users from the database."""
        return load_user_database(self.database_path)

    def save_users(self):
        """Save users to the database."""
        try:
            save_user_database(self.names, self.ear_thresholds, self.face_encodings, self.database_path)
        except Exception as e:
            print(f"Error saving user database: {e}")

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
        return embedding_tensor.cpu().numpy().flatten().tolist()

    def find_duplicate(self, face_encoding, threshold=None):
        """
        Checks if the given face encoding matches any existing user.
        Returns (is_duplicate, index).
        """
        if threshold is None:
            threshold = self.duplicate_threshold
        encodings = [np.array(enc, dtype=np.float32).flatten() for enc in self.face_encodings]
        face_encoding_np = np.array(face_encoding, dtype=np.float32).flatten()
        if not encodings:
            return False, None
        # Ensure shape consistency
        encodings = [enc for enc in encodings if enc.shape == face_encoding_np.shape]
        if not encodings:
            return False, None
        distances = np.linalg.norm(np.stack(encodings) - face_encoding_np, axis=1)
        min_idx = np.argmin(distances)
        if distances[min_idx] < threshold:
            return True, min_idx
        return False, None

    def register(self, image_frame, ear_threshold, user_name=None):
        """
        Registers a new user with the provided image frame and EAR threshold.
        If duplicate is found, returns the existing user profile.
        Allows custom user name.
        """
        print("Registering new user...")

        face_encoding = self.get_face_encoding(image_frame)
        if face_encoding is None:
            print("Registration failed: No face could be detected.")
            return None

        # Check for duplicate
        is_dup, dup_idx = self.find_duplicate(face_encoding)
        if is_dup:
            print(f"Duplicate found: {self.names[dup_idx]}")
            return {
                "name": self.names[dup_idx],
                "ear_threshold": self.ear_thresholds[dup_idx],
                "face_encoding": self.face_encodings[dup_idx]
            }

        # Use provided name or generate one
        if user_name is None:
            user_name = f"User_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Shape consistency check
        if self.face_encodings:
            expected_shape = np.array(self.face_encodings[0]).shape
            if np.array(face_encoding).shape != expected_shape:
                print(f"Error: New encoding shape {np.array(face_encoding).shape} does not match expected {expected_shape}. Registration aborted.")
                return None

        self.names.append(user_name)
        self.ear_thresholds.append(ear_threshold)
        self.face_encodings.append(face_encoding)
        self.save_users()

        print(f"Successfully registered {user_name} with EAR threshold {ear_threshold:.2f}")
        return {
            "name": user_name,
            "ear_threshold": ear_threshold,
            "face_encoding": face_encoding
        }