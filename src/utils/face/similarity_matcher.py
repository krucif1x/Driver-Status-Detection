import numpy as np
from typing import Optional, Tuple, List
from src.infrastructure.data.user.profile import UserProfile

class SimilarityMatcher:
    def __init__(self):
        self._mat: Optional[np.ndarray] = None

    def build_matrix(self, users: List[UserProfile]):  # ✅ Renamed from build() to build_matrix()
        """Build similarity matrix from user face encodings."""
        if not users:
            self._mat = None
            return
        encs = [u.face_encoding for u in users]
        self._mat = np.stack(encs, axis=0).astype(np.float32)
        norms = np.linalg.norm(self._mat, axis=1, keepdims=True) + 1e-8
        self._mat /= norms

    def best_match(self, encoding: np.ndarray, users: List[UserProfile]) -> Tuple[Optional[UserProfile], float]:
        """Find best matching user for given encoding."""
        if not users or self._mat is None:
            return None, -1.0
        q = encoding / (np.linalg.norm(encoding) + 1e-8)
        sims = self._mat @ q
        i = int(np.argmax(sims))
        return users[i], float(sims[i])