import cv2
import base64
import logging
import numpy as np
from datetime import datetime
from typing import Optional

from src.services.remote_logger import RemoteLogWorker
from src.infrastructure.data.repository import UnifiedRepository
from src.infrastructure.data.models import DrowsinessEvent

UNKNOWN_USER_ID = 0

class SystemLogger:
    """
    The Coordinator. Handles Buzzer, Local DB, and Remote Push.
    """
    def __init__(
        self, 
        buzzer=None, 
        remote_worker: Optional[RemoteLogWorker] = None, 
        event_repo: Optional[UnifiedRepository] = None,
        vehicle_vin: str = "VIN-0001",
        local_quality: int = 85,
        remote_quality: int = 70
    ):
        self.buzzer = buzzer
        self.remote = remote_worker
        self.repo = event_repo
        self.vehicle_vin = vehicle_vin
        self.local_quality = local_quality
        self.remote_quality = remote_quality

    def log_event(self, user_id: int, event_type: str, duration: float = 0.0, 
                  value: float = 0.0, frame: Optional[np.ndarray] = None,
                  alert_category: str = None, alert_detail: str = None, 
                  severity: str = None):
        
        # DEBUG: Entry point
        logging.info(f"🔍 log_event CALLED: user_id={user_id}, type={event_type}, category={alert_category}, detail={alert_detail}")
        logging.info(f"🔍 Parameters: duration={duration}, value={value}, frame={'Present' if frame is not None else 'None'}")
        logging.info(f"🔍 Repository: {self.repo is not None}, Remote: {self.remote is not None}")
        
        timestamp = datetime.now()
        remote_allowed = self.remote and self.remote.enabled and user_id != UNKNOWN_USER_ID
        
        logging.info(f"🔍 Remote allowed: {remote_allowed} (enabled={self.remote.enabled if self.remote else 'N/A'}, user_id={user_id})")

        jpeg_local = None
        jpeg_remote = None

        if frame is not None:
            logging.info(f"🔍 Encoding frame: shape={frame.shape}, dtype={frame.dtype}")
            jpeg_local = self._encode_jpeg(frame, self.local_quality)
            if jpeg_local:
                logging.info(f"✓ Frame encoded for local: {len(jpeg_local)} bytes")
            else:
                logging.error(f"❌ Failed to encode frame for local storage")
                
            if remote_allowed:
                h, w = frame.shape[:2]
                target_w = 640
                if w > target_w:
                    scale = target_w / float(w)
                    resized = cv2.resize(frame, (target_w, int(round(h * scale))))
                    logging.info(f"🔍 Resized frame: {w}x{h} -> {resized.shape[1]}x{resized.shape[0]}")
                else:
                    resized = frame
                jpeg_remote = self._encode_jpeg(resized, self.remote_quality)
                if jpeg_remote:
                    logging.info(f"✓ Frame encoded for remote: {len(jpeg_remote)} bytes")
                else:
                    logging.error(f"❌ Failed to encode frame for remote")
        else:
            logging.warning("⚠️ No frame provided to log_event")

        # LOCAL DATABASE LOGGING
        if self.repo:
            logging.info(f"📝 Creating DrowsinessEvent for database...")
            try:
                event = DrowsinessEvent(
                    vehicle_identification_number=self.vehicle_vin,
                    user_id=user_id,
                    status=event_type.lower(),
                    time=timestamp,
                    img_drowsiness=jpeg_local,
                    img_path=None,
                    duration=duration,
                    value=value,
                    alert_category=alert_category,
                    alert_detail=alert_detail,
                    severity=severity
                )
                logging.info(f"✓ DrowsinessEvent created successfully")
                
                row_id = self.repo.add_event(event)
                if row_id > 0:
                    logging.info(f"✅ Event saved to DB: ID={row_id}, {alert_category or 'Event'}: {alert_detail or event_type} ({severity or 'N/A'})")
                else:
                    logging.error(f"❌ Failed to save event to database! row_id={row_id}")
            except Exception as e:
                logging.error(f"❌ Exception creating/saving event: {e}", exc_info=True)
        else:
            logging.error("❌ No repository available! Events not being saved.")

        # REMOTE LOGGING
        if remote_allowed and self.remote:
            logging.info(f"📤 Sending to remote: vin={self.vehicle_vin}, user={user_id}, status={event_type}")
            try:
                self.remote.send_or_queue(
                    vehicle_vin=self.vehicle_vin,
                    user_id=user_id,
                    status=event_type,
                    time_dt=timestamp,
                    raw_jpeg=jpeg_remote,
                    alert_category=alert_category,
                    alert_detail=alert_detail,
                    severity=severity
                )
                logging.info(f"✓ Remote send_or_queue called successfully")
            except Exception as e:
                logging.error(f"❌ Exception sending to remote: {e}", exc_info=True)
        elif not remote_allowed:
            logging.info(f"⊘ Remote logging skipped (remote_allowed={remote_allowed})")
        
        logging.info(f"🏁 log_event completed")

    def signal(self, kind: str):
        """
        Finite buzzer signals (safe on machines without buzzer).
        """
        if not self.buzzer:
            return

        try:
            k = (kind or "").lower()

            if k == "startup":
                # 2 short pulses
                self.buzzer.pulse(0.15, background=True)
                # slight delay then another pulse
                import threading
                threading.Timer(0.30, lambda: self.buzzer.pulse(0.15, background=True)).start()

            elif k == "identity_prompt":
                # short periodic beep for a moment
                self.buzzer.beep_for(on_time=0.10, off_time=0.10, duration_sec=1.0)

            elif k == "calibration_prompt":
                # Requirement: 1s buzz intervals for 10 seconds
                self.buzzer.beep_for(on_time=1.0, off_time=1.0, duration_sec=10.0)

            elif k == "calibration_success":
                # 3 short pulses
                self.buzzer.pulse(0.12, background=True)
                import threading
                threading.Timer(0.25, lambda: self.buzzer.pulse(0.12, background=True)).start()
                threading.Timer(0.50, lambda: self.buzzer.pulse(0.12, background=True)).start()

            elif k == "calibration_fail":
                # long beep
                self.buzzer.pulse(0.8, background=True)

            elif k == "drowsy":
                # stronger pattern for a few seconds
                self.buzzer.beep_for(on_time=0.25, off_time=0.10, duration_sec=2.5)

            elif k == "distraction":
                self.buzzer.beep_for(on_time=0.10, off_time=0.10, duration_sec=2.0)

        except Exception:
            return

    def alert(self, level: str = "warning"):
        """Backwards-compatible; uses finite patterns now."""
        if not self.buzzer:
            return
        lvl = (level or "warning").lower()
        if lvl == "critical":
            self.signal("drowsy")
        elif lvl == "distraction":
            self.signal("distraction")
        else:
            self.buzzer.beep_for(0.10, 0.10, 1.5)
            
    def stop_alert(self):
        if self.buzzer: self.buzzer.off()

    def _encode_jpeg(self, frame, quality):
        try:
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), 
                                 [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
            return bytes(buf) if ok else None
        except: return None