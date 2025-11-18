import logging
import torch
import time
import numpy as np  # <--- Added this
from typing import List, Optional
from threading import Lock

from src.infrastructure.data.user.profile import UserProfile
from src.infrastructure.data.user.database import UserDatabase
from src.infrastructure.data.user.repository import UserRepository
from src.utils.models.model_loader import FaceModelLoader
from src.utils.preprocessing.image_validator import ImageValidator
from src.utils.face.encoding_extractor import FaceEncodingExtractor
from src.utils.face.similarity_matcher import SimilarityMatcher

class UserManager:
    """
    Thread-safe user face recognition with optimized caching.
    Delegates work to specialized components (SRP).
    """
    
    def __init__(
        self, 
        database_file: str = r"data\drowsiness_events.db",
        recognition_threshold: float = 0.90,
        duplicate_threshold: float = 0.92,
        input_color: str = "RGB"
    ):
        logging.info(f"UserManager initializing with database: {database_file}")
        
        self.recognition_threshold = recognition_threshold
        self.duplicate_threshold = duplicate_threshold
        self.input_color = (input_color or "RGB").upper()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self._lock = Lock()
        
        # Components
        self.db = UserDatabase(database_file)
        self.repo = UserRepository(self.db)
        self.model_loader = FaceModelLoader(device=str(self.device))
        self.validator = ImageValidator(self.input_color)
        self.encoder = FaceEncodingExtractor(self.model_loader, self.validator, self.device)
        self.matcher = SimilarityMatcher()

        # Caches & Stats
        self.users: List[UserProfile] = []
        self._user_id_map: dict[int, UserProfile] = {}
        self._stats = {
            'total_recognitions': 0,
            'successful_matches': 0,
            'avg_recognition_time': 0.0
        }
        
        self._load_users()
        logging.info("UserManager initialized successfully")

    def _load_users(self):
        """Load all user profiles from database into memory cache."""
        try:
            with self._lock:
                self.users = self.repo.load_all()
                self._user_id_map = {u.user_id: u for u in self.users}
                self.matcher.build_matrix(self.users)
            logging.info(f"Loaded {len(self.users)} user profile(s)")
        except Exception as e:
            logging.error(f"Error loading users: {e}")
            self.users = []
            self._user_id_map = {}

    def find_best_match(self, image_frame) -> Optional[UserProfile]:
        """Find best matching user from image frame."""
        start = time.time()
        encoding = self.encoder.extract(image_frame)
        
        if encoding is None:
            return None

        with self._lock:
            best_user, sim = self.matcher.best_match(encoding, self.users)

        self._stats['total_recognitions'] += 1

        if best_user and sim >= self.recognition_threshold:
            self._stats['successful_matches'] += 1
            self.repo.update_last_seen(best_user.user_id)
            
            elapsed = time.time() - start
            self._stats['avg_recognition_time'] = (
                self._stats['avg_recognition_time'] * 0.9 + elapsed * 0.1
            )
            return best_user

        return None

    # --- NEW METHOD 1: Identity Verification ---
    def verify_identity(self, frame, current_user, threshold=0.7) -> bool:
        """
        Quickly check if the face in the frame matches the current_user.
        Removes math logic from DetectionLoop.
        """
        if current_user is None: 
            return False
        
        encoding = self.encoder.extract(frame)
        if encoding is None or not encoding.any(): 
            return False
        
        # Cosine Similarity logic
        norm_enc = encoding / (np.linalg.norm(encoding) + 1e-8)
        sim = float(np.dot(current_user.face_encoding, norm_enc))
        
        return sim >= threshold

    # --- NEW METHOD 2: Registration Sequence ---
    def register_sequence(self, camera, ear_threshold) -> Optional[UserProfile]:
        """
        Handles the 'Take Photo -> Save to DB' flow.
        Removes the capture loop from DetectionLoop.
        """
        reg_frame = None
        # Try to get a good encoding frame for up to 20 attempts
        for _ in range(20):
            frame = camera.read()
            if frame is None: continue

            encoding = self.encoder.extract(frame)
            if encoding is not None and encoding.any():
                reg_frame = frame
                break
            time.sleep(0.02)
            
        if reg_frame is None:
            logging.error("Failed to capture valid registration frame")
            return None
            
        # Get ID and Save
        user_id = self.repo.get_next_user_id()
        return self.register_new_user(reg_frame, ear_threshold, user_id)

    def register_new_user(self, image_frame, ear_threshold: float, user_id: int) -> Optional[UserProfile]:
        """Register new user with face encoding and EAR threshold."""
        if ear_threshold is None: return None

        encoding = self.encoder.extract(image_frame)
        if encoding is None: return None

        with self._lock:
            duplicate, sim = self.matcher.best_match(encoding, self.users)

        if duplicate and sim >= self.duplicate_threshold:
            logging.warning(f"Duplicate face detected (similarity={sim:.3f})")
            return duplicate

        new_user = UserProfile(0, user_id, ear_threshold, encoding)

        try:
            new_user.id = self.repo.save_user(new_user)
            with self._lock:
                self.users.append(new_user)
                self._user_id_map[user_id] = new_user
                self.matcher.build_matrix(self.users)
            
            logging.info(f"Registered new user: {new_user}")
            return new_user
        except Exception as e:
            logging.error(f"User registration failed: {e}")
            return None

    def get_user_by_server_id(self, user_id: int) -> Optional[UserProfile]:
        with self._lock:
            return self._user_id_map.get(user_id)