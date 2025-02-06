import RPi.GPIO as GPIO
import time

BUTTON_PIN = 17

running = False




def pressed(channel):
	global running
	running = not running
	if running: 
		print("Pressed: on")
	else:
		print("Pressed: off")
		
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.add_event_detect(BUTTON_PIN, GPIO.FALLING, callback=pressed, bouncetime=300)


try:
	while True:
		if running:
			print("run code")
			time.sleep(1)
		else:
			time.sleep(0.1)
except KeyboardInterrupt:
	print("Exiting")
finally:
	GPIO.cleanup()
			
