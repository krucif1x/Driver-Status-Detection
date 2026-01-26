from abc import ABC, abstractmethod
from math import dist
from typing import List, Tuple


class AspectRatio(ABC):
    @abstractmethod
    def calculate(self, landmarks: List[Tuple[float, float]]) -> float:
        raise NotImplementedError


class EAR(AspectRatio):
    def calculate(self, landmarks: List[Tuple[float, float]]) -> float:
        """
        Eye Aspect Ratio.
        Expects 6 points: [Corner1, Top1, Top2, Corner2, Bot2, Bot1]
        """
        a = dist(landmarks[1], landmarks[5])
        b = dist(landmarks[2], landmarks[4])
        c = dist(landmarks[0], landmarks[3])

        if c < 1e-6:
            return 0.0
        return (a + b) / (2.0 * c)


class MAR(AspectRatio):
    def calculate(self, landmarks: List[Tuple[float, float]]) -> float:
        a = dist(landmarks[1], landmarks[5])
        b = dist(landmarks[2], landmarks[4])
        c = dist(landmarks[0], landmarks[3])

        if c < 1e-6:
            return 0.0
        return (a + b) / (2.0 * c)