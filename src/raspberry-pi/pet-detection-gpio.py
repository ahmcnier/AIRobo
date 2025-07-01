from gpiozero import LED
from time import sleep

led = LED(17)
print("LED on")
led.on()
sleep(20)