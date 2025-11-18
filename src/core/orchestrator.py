import logging
import cv2
import sqlite3
import yaml
import os
import numpy as np  # <--- Make sure this is imported
from pathlib import Path

from src.infrastructure.hardware.camera import Camera
from src.infrastructure.hardware.buzzer import Buzzer
from src.services.user_manager import UserManager
from src.services.remote_logger import RemoteLogWorker
from src.services.system_logger import SystemLogger
from src.infrastructure.data.drowsiness_events.repository import DrowsinessEventRepository
from src.mediapipe.mediapipe_wrapper import MediaPipeFaceModel
from src.core.detection_loop import DetectionLoop

log = logging.getLogger(__name__)

class DrowsinessSystem:
    CONFIG_PATH = "config/detector_config.yaml"
    DB_PATH = "data/drowsiness_events.db"

    def __init__(self):
        self._ensure_paths()
        self.config = self._load_config()
        sys_cfg = self.config.get('system', {})
        self.vin = sys_cfg.get('vin', os.getenv('DS_VIN', 'VIN-0001'))
        self.fps = float(sys_cfg.get('target_fps', 30.0))

    def run(self):
        try:
            self._init_resources()
            log.info(f"System Ready. VIN: {self.vin}")
            print("\n>>> PRESS 'Q' TO EXIT <<<\n")
            
            DetectionLoop(
                camera=self.camera, 
                face_mesh=self.face_mesh, 
                buzzer=self.buzzer, 
                user_manager=self.user_manager, 
                system_logger=self.system_logger, 
                vehicle_vin=self.vin, 
                fps=self.fps, 
                detector_config_path=self.CONFIG_PATH
            ).run()
            
        except KeyboardInterrupt:
            log.info("User requested shutdown.")
        except Exception as e:
            log.error(f"Fatal: {e}", exc_info=True)
        finally:
            self._cleanup()

    def _init_resources(self):
        # 1. Database
        self.conn = sqlite3.connect(self.DB_PATH, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.event_repo = DrowsinessEventRepository(self.conn)
        
        # 2. Services
        self.user_manager = UserManager(database_file=self.DB_PATH)
        self.remote_worker = RemoteLogWorker(self.DB_PATH, os.getenv("DS_REMOTE_URL"), True)
        self.buzzer = Buzzer(pin=18)
        self.system_logger = SystemLogger(self.buzzer, self.remote_worker, self.event_repo, self.vin)
        
        # 3. Hardware
        self.camera = Camera(source='auto', resolution=(640, 480))
        if not self.camera.ready:
            raise RuntimeError("Camera failed to open")
            
        # 4. AI Model (Warmup with generated black frame)
        self.face_mesh = MediaPipeFaceModel(max_num_faces=1, refine_landmarks=True)
        
        # --- THE FIX IS HERE ---
        # Create a blank black image (480x640) in memory. No file needed.
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.face_mesh.process(dummy_frame)
        # -----------------------

    def _cleanup(self):
        log.info("Shutting down...")
        if hasattr(self, 'remote_worker') and self.remote_worker: self.remote_worker.close()
        if hasattr(self, 'conn') and self.conn: self.conn.close()
        if hasattr(self, 'buzzer') and self.buzzer: self.buzzer.off()
        if hasattr(self, 'camera') and self.camera: self.camera.release()
        if hasattr(self, 'face_mesh') and self.face_mesh: self.face_mesh.close()
        cv2.destroyAllWindows()

    def _ensure_paths(self):
        Path("data").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)

    def _load_config(self):
        if Path(self.CONFIG_PATH).exists():
             with open(self.CONFIG_PATH) as f: return yaml.safe_load(f)
        return {}