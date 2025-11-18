import numpy as np
import torch
import torch.nn.functional as F
import logging, time
from typing import Optional
from src.utils.preprocessing.image_validator import ImageValidator
from src.utils.models.model_loader import FaceModelLoader

class FaceEncodingExtractor:
    MIN_DET_PROB = 0.90

    def __init__(self, model_loader: FaceModelLoader, validator: ImageValidator, device: torch.device):
        self.models = model_loader
        self.validator = validator
        self.device = device

    def extract(self, frame) -> Optional[np.ndarray]:
        t0 = time.time()
        try:
            frame = self.validator.preprocess(frame)
        except ValueError:
            return None

        try:
            faces, probs = self.models.mtcnn(frame, return_prob=True)
        except TypeError:
            res = self.models.mtcnn(frame)
            faces, probs = (res, None) if res is not None else (None, None)

        if faces is None or not isinstance(faces, torch.Tensor):
            return None
        if faces.ndim == 3:
            faces = faces.unsqueeze(0)
        if faces.shape[0] == 0:
            return None

        idx = 0
        if probs is not None and len(probs) == faces.shape[0]:
            pa = np.array(probs, dtype=np.float32).reshape(-1)
            idx = int(np.argmax(pa))
            if float(pa[idx]) < self.MIN_DET_PROB:
                return None

        face = faces[idx].unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.models.resnet(face)
            emb = F.normalize(emb, p=2, dim=1)
        logging.debug(f"Encoding in {(time.time()-t0)*1000:.1f}ms")
        return emb.cpu().numpy().flatten()