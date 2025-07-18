from gpiozero import LED
from time import sleep
import RPi.GPIO as GPIO
import time
import cv2

led = LED(17)

# Open first camera (usually /dev/video0)
cap1 = cv2.VideoCapture(0)

# Open second camera (usually /dev/video1)
cap2 = cv2.VideoCapture(1)

if not cap1.isOpened():
    print("Camera 0 failed to open.")
    exit()

if not cap2.isOpened():
    print("Camera 1 failed to open.")
    exit()

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

        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            print("❌ Failed to read from one of the cameras.")
            break

        # Show both camera feeds in different windows
        cv2.imshow("Camera 1", frame1)
        cv2.imshow("Camera 2", frame2)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    GPIO.cleanup()
    # Release both cameras and close windows
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()