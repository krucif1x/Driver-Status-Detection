import time
from collections import deque

class FpsTracker:
    """Tracks and calculates Frames Per Second."""
    def __init__(self):
        self.prev_time = time.time()
        self.current_fps = 0.0

    def update(self) -> float:
        """Updates time and returns current FPS."""
        curr_time = time.time()
        elapsed = curr_time - self.prev_time
        self.prev_time = curr_time
        
        if elapsed > 0:
            self.current_fps = 1.0 / elapsed
        return self.current_fps

class RollingAverage:
    """
    An optimized rolling average calculator using O(1) updates.
    Replaces the manual deque management in DetectionLoop.
    """
    def __init__(self, duration_sec: float, target_fps: float = 30.0):
        buffer_size = max(1, int(target_fps * duration_sec))
        self.buffer = deque(maxlen=buffer_size)
        self.current_sum = 0.0

    def update(self, value: float) -> float:
        """
        Adds a new value and returns the smoothed average.
        Handles the running sum efficiently.
        """
        if value is None:
            return self.get_average()

        # If buffer is full, subtract the oldest value before it falls off
        if len(self.buffer) == self.buffer.maxlen:
            self.current_sum -= self.buffer[0]

        self.buffer.append(value)
        self.current_sum += value
        
        return self.current_sum / len(self.buffer)

    def get_average(self) -> float:
        return self.current_sum / len(self.buffer) if self.buffer else 0.0