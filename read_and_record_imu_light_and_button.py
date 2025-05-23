# from Izzy's ChatGPT on 1/29/25
#with further human edits

import time
import board
import busio
import adafruit_bno055
import csv
import datetime

import RPi.GPIO as GPIO
from gpiozero import LED

GPIO.cleanup()
GPIO.setmode(GPIO.BCM)

led = LED(17)
BUTTON_PIN = 27

# GPIO.setwarnings(False)
# GPIO.setmode(GPIO.BOARD)
# GPIO.setup(17, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

def button_pressed(channel):
    global running
    running = not running
    if running:
        print("Pressed: on")
    else:
        print("Pressed: off")
        
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.add_event_detect(BUTTON_PIN, GPIO.FALLING, callback=button_pressed, bouncetime=300)
running = False

# Initialize I2C connection
i2c = busio.I2C(3, 2)  # Uses GPIO2 (SDA) and GPIO3 (SCL)
sensor = adafruit_bno055.BNO055_I2C(i2c, address=0x28)  # Default BNO055 I2C address

print("Reading BNO055 Acceleration Data...")



# Function to read acceleration data
def get_acceleration():
    accel = sensor.linear_acceleration  # Returns a tuple (x, y, z) in m/s²
    if accel is None:
        return (0, 0, 0)  # If no data, return zeros
    return accel

# Function to read temperature data
def get_temperature():
    temp = sensor.temperature  # Returns temp ?
    if temp is None:
        return 0  # If no data, return zeros
    return temp

# Function to read gyroscopic data
def get_gyroscope():
    gyro_output = sensor.gyro  # Returns a tuple (x, y, z) in rad/s
    if gyro_output is None:
        return (0, 0, 0)  # If no data, return zeros
    return gyro_output

# Continuous data reading loop
try:
    with open('imu_output.csv', 'w', newline="") as file:
        writer = csv.writer(file)
            
        writer.writerow(["Time", "AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ"])
        
    
    
    
    while True:
        if running:
            led.on()
            #if GPIO.input(17) == GPIO.HIGH:
                #print("Started")
            with open('imu_output.csv', 'a', newline="") as file:
                writer = csv.writer(file)
                
                #writer.writerow(["Time", "AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ"])
                
                x, y, z = get_acceleration()
                t = get_temperature()
                a, b, c = get_gyroscope()
                #print(f"Acceleration -> X: {x:.2f} m/s², Y: {y:.2f} m/s², Z: {z:.2f} m/s²")
                #print(f"Temperature -> T: {t:.2f}")
                #print(f"Gyroscope -> X: {a:.2f} rad/s, Y: {b:.2f} rad/s, Z: {c:.2f} rad/s")
                #print("~~~~")
                
                writer.writerow([datetime.datetime.now(), x, y, z, a, b, c])

                
                time.sleep(0.05) #Read every 0.1 seconds
        else:
            print("Not running code")
            led.off()
    print("\nExiting program. Goodbye!")
finally:
    GPIO.cleanup()
