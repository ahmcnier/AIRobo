from pathlib import Path

from gpiozero import LED
from time import sleep
import RPi.GPIO as GPIO
import time
import cv2

led = LED(17)

# Open first camera (usually /dev/video0)
cap1 = cv2.VideoCapture(0)

# Open second camera (usually /dev/video1)
cap2 = cv2.VideoCapture(2)

# Open second camera (usually /dev/video2)
cap3 = cv2.VideoCapture(4)

cameras = [cap1, cap2, cap3]

#test US sensor
TRIG = 23
ECHO = 24

GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

def get_distance():
    GPIO.output(TRIG, False)
    time.sleep(0.1)

    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()

    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150  # Speed of sound constant
    distance = round(distance, 2)

    return distance

try:
    while True:
        dist = get_distance()
        print(f"Distance: {dist} cm")
        time.sleep(1)

        #light up LED when object is closer than 30cm away
        if dist >= 30:
            print("LED off")
            led.off()
        else:
            print("LED on")
            led.on()

        index = 0
        for cam in cameras:
            ret, frame = cam.read()
            cam.release()

            if not ret:
                print("Camera %i failed", index)

            out_dir = Path("camera-images")
            out_dir.mkdir(exist_ok=True)
            filename = out_dir / f"photo_{index}.jpg"

            cv2.imwrite(str(filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(f"Saved JPEG to {filename}")

except KeyboardInterrupt:
    GPIO.cleanup()
    # Release both cameras and close windows
    cap1.release()
    cap2.release()
    cap3.release()
    cv2.destroyAllWindows()