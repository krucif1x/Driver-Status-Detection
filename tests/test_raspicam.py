from picamera2 import Picamera2
import cv2
import time

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()
time.sleep(1.0)  # Allow camera to warm up

while True:
    frame = picam2.capture_array()
    if frame is None:
        continue
    cv2.imshow("Raspberry Pi Camera Feed", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

picam2.stop()
cv2.destroyAllWindows()