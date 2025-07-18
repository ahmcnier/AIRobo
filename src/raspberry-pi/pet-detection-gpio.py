from gpiozero import LED
from time import sleep

led = LED(17)
for i in range(10):
    print("LED on")
    led.on()
    sleep(2)
    print("LED off")
    led.off()