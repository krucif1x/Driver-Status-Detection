import logging
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple, Union
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image


@dataclass
class FaceEncodingMetadata:
    detection_prob: float = 0.0
    num_faces_detected: int = 0
    extraction_time_ms: float = 0.0
    face_selected_index: int = -1


class FaceRecognizer:
    """
    Single-responsibility component: given a frame, return a normalized 512-dim face encoding.

    Encapsulates:
      - preprocess/validation (RGB/BGR normalization)
      - model loading (MTCNN + InceptionResnetV1)
      - face detection + best-face selection
      - embedding extraction + normalization
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        input_color: str = "RGB",
        min_detection_prob: float = 0.95,
        keep_all: bool = True,
        image_size: int = 160,
        margin: int = 40,
        detector_max_side: int = 320,  # NEW (perf): downscale for detection
    ):
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.input_color = (input_color or "RGB").upper()
        self.min_detection_prob = float(min_detection_prob)

        self._keep_all = bool(keep_all)
        self._image_size = int(image_size)
        self._margin = int(margin)
        self.detector_max_side = int(detector_max_side)

        self._mtcnn: Optional[MTCNN] = None
        self._resnet: Optional[InceptionResnetV1] = None

        self._stats = {
            "total_extractions": 0,
            "successful_extractions": 0,
            "failed_no_face": 0,
            "failed_low_confidence": 0,
            "failed_preprocessing": 0,
            "avg_extraction_time_ms": deque(maxlen=500),  
        }

        logging.info(
            "FaceRecognizer initialized: device=%s input_color=%s min_prob=%.2f",
            str(self.device),
            self.input_color,
            self.min_detection_prob,
        )

    @property
    def mtcnn(self) -> MTCNN:
        if self._mtcnn is None:
            logging.info("Loading MTCNN...")
            self._mtcnn = MTCNN(
                keep_all=self._keep_all,
                image_size=self._image_size,
                margin=self._margin,
                device=self.device,
                post_process=False,
            )
        return self._mtcnn

    @property
    def resnet(self) -> InceptionResnetV1:
        if self._resnet is None:
            logging.info("Loading InceptionResnetV1...")
            self._resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
            for p in self._resnet.parameters():
                p.requires_grad = False
        return self._resnet

    @lru_cache(maxsize=128)
    def _shape_ok(self, shape: tuple, dtype_str: str) -> bool:
        return len(shape) == 3 and shape[2] == 3

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        if not isinstance(frame, np.ndarray):
            raise ValueError("Frame must be np.ndarray")

        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        if not self._shape_ok(frame.shape, str(frame.dtype)):
            raise ValueError(f"Invalid image shape: {frame.shape}")

        if self.input_color == "BGR":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # PERF: downscale large frames for MTCNN
        h, w = frame.shape[:2]
        max_side = max(h, w)
        if self.detector_max_side > 0 and max_side > self.detector_max_side:
            scale = self.detector_max_side / float(max_side)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return frame

    def extract(self, frame: Any, return_metadata: bool = False) -> Union[Optional[np.ndarray], Tuple[Optional[np.ndarray], Dict[str, Any]]]:
        t0 = time.time()
        self._stats["total_extractions"] += 1

        md = FaceEncodingMetadata()
        metadata_dict: Dict[str, Any] = {
            "detection_prob": md.detection_prob,
            "num_faces_detected": md.num_faces_detected,
            "extraction_time_ms": md.extraction_time_ms,
            "face_selected_index": md.face_selected_index,
        }

        try:
            frame = self._preprocess(frame)
        except ValueError as e:
            logging.debug("Preprocessing failed: %s", e)
            self._stats["failed_preprocessing"] += 1
            return (None, metadata_dict) if return_metadata else None

        pil_frame = Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame

        try:
            faces, probs = self.mtcnn(pil_frame, return_prob=True)
        except TypeError:
            res = self.mtcnn(pil_frame)
            faces, probs = (res, None) if res is not None else (None, None)

        if faces is None or not isinstance(faces, torch.Tensor):
            self._stats["failed_no_face"] += 1
            return (None, metadata_dict) if return_metadata else None

        if faces.ndim == 3:
            faces = faces.unsqueeze(0)

        if faces.shape[0] == 0:
            self._stats["failed_no_face"] += 1
            return (None, metadata_dict) if return_metadata else None

        md.num_faces_detected = int(faces.shape[0])
        md.face_selected_index = 0
        md.detection_prob = 1.0

        idx = 0
        if probs is not None and len(probs) == faces.shape[0]:
            probs_array = np.array(probs, dtype=np.float32).reshape(-1)
            idx = int(np.argmax(probs_array))
            md.face_selected_index = idx
            md.detection_prob = float(probs_array[idx])

            if md.detection_prob < self.min_detection_prob:
                self._stats["failed_low_confidence"] += 1
                metadata_dict.update(md.__dict__)
                return (None, metadata_dict) if return_metadata else None

        face = faces[idx].unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.resnet(face)
            embedding = F.normalize(embedding, p=2, dim=1)

        encoding = embedding.cpu().numpy().flatten()

        if not self.validate_encoding(encoding):
            metadata_dict.update(md.__dict__)
            return (None, metadata_dict) if return_metadata else None

        extraction_time_ms = (time.time() - t0) * 1000.0
        md.extraction_time_ms = extraction_time_ms

        self._stats["successful_extractions"] += 1
        self._stats["avg_extraction_time_ms"].append(extraction_time_ms)

        metadata_dict.update(md.__dict__)
        return (encoding, metadata_dict) if return_metadata else encoding

    def validate_encoding(self, encoding: np.ndarray) -> bool:
        if encoding is None:
            return False
        if len(encoding) != 512:
            logging.warning("Invalid encoding size: %s (expected 512)", len(encoding))
            return False
        if np.isnan(encoding).any() or np.isinf(encoding).any():
            logging.warning("Encoding contains NaN/Inf values")
            return False

        norm = float(np.linalg.norm(encoding))
        if norm < 0.5 or norm > 1.5:
            logging.warning("Encoding norm out of range: %.3f (expected ~1.0)", norm)
            return False

        if float(np.abs(encoding).max()) > 10.0:
            logging.warning("Encoding values out of range: max=%.3f", float(np.abs(encoding).max()))
            return False

        return True