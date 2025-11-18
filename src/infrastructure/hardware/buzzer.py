class Buzzer:
    def __init__(self, pin=17):
        try:
            from gpiozero import Buzzer
            self.buzzer = Buzzer(pin)
            self.ready = True
        except Exception as e:
            print(f"Warning: Could not initialize buzzer ({e}). Continuing without buzzer.")
            self.buzzer = None
            self.ready = False

    def beep(self, on_time=0.5, off_time=0.5, background=True):
        if self.buzzer and self.ready:
            self.buzzer.beep(on_time=on_time, off_time=off_time, background=background)

    def off(self):
        if self.buzzer and self.ready:
            self.buzzer.off()