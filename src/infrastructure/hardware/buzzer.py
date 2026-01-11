import logging
import threading
import time

log = logging.getLogger(__name__)


class Buzzer:
    def __init__(self, pin: int = 18):
        self._buzzer = None
        try:
            from gpiozero import Buzzer as _GPIOBuzzer  # type: ignore
            self._buzzer = _GPIOBuzzer(pin)
        except Exception as e:
            # No GPIO on laptops/PCs, or gpiozero not installed
            self._buzzer = None
            log.warning("Buzzer unavailable (%s). Continuing without buzzer.", e)

    def available(self) -> bool:
        return self._buzzer is not None

    def __bool__(self) -> bool:
        return self.available()

    def beep(self, on_time: float = 0.1, off_time: float = 0.1, background: bool = True):
        if not self._buzzer:
            return
        try:
            # gpiozero supports beep(on_time, off_time, n=None, background=True)
            self._buzzer.beep(on_time=on_time, off_time=off_time, background=background)
        except Exception:
            return

    def off(self):
        if not self._buzzer:
            return
        try:
            self._buzzer.off()
        except Exception:
            return

    def pulse(self, duration_sec: float = 0.2, background: bool = True):
        """Single beep: ON for duration_sec then OFF."""
        if not self._buzzer:
            return

        def _run():
            try:
                self._buzzer.on()
                time.sleep(max(0.0, float(duration_sec)))
            except Exception:
                pass
            finally:
                try:
                    self._buzzer.off()
                except Exception:
                    pass

        if background:
            threading.Thread(target=_run, daemon=True).start()
        else:
            _run()

    def beep_for(self, on_time: float, off_time: float, duration_sec: float):
        """Beep pattern for a fixed duration, then stop."""
        if not self._buzzer:
            return
        try:
            self.beep(on_time=on_time, off_time=off_time, background=True)
            threading.Timer(max(0.0, float(duration_sec)), self.off).start()
        except Exception:
            return