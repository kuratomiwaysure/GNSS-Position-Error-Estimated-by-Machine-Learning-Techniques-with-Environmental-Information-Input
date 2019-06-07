
"--------------------------------------------------------------------------------------------"
"-----------------This solution is based on the solution found in:---------------------------"
"https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/"
"--------------------------------------------------------------------------------------------"

from __future__ import print_function
from picamera import PiCamera
from picamera.array import PiRGBArray
from imutils.video.pivideostream import PiVideoStream
from imutils.video import FPS
import datetime
import imutils
import time
import cv2
import RPi.GPIO as GPIO
from time import sleep
import sys

done = False

#Starts taking pictures once input button is pressed
def startCameraSequence():

    resolution = (1200,1008) #Max resolution (3280,2464), Min resolution (64,64)
    framerate = 10

    vs = PiVideoStream(resolution,framerate).start()
    time.sleep(2.0)
    fps = FPS().start()
    start_time = time.time()
    wait_again = True

    #Loop over frames using threaded stream
    while wait_again:

        #Gets timestamp
        now = datetime.datetime.now()
        now_print = now.strftime("%H:%M:%S.%f")

        #Reads the current frame at capture thread
        frame = vs.read()

        #Writes the frame into folder with time stamp
        #Change "/home/pi/Desktop/camera_test/" with desired saving folder"
        cv2.imwrite("/home/pi/Desktop/camera_test/"+now_print+".jpg", frame)

        #Update FPS counter
        fps.update()

        pressed_again = GPIO.input(18)
        if(pressed_again):
            wait_again = False

    #Stop the timer and display FPS information
    fps.stop()
    vs.stop()

    #Updates time taking pictures
    end_time = time.time()
    elapsed = end_time - start_time

    #Writes the elapsed time of capture
    #Change "/home/pi/Desktop/camera_test/" with desired saving folder"
    f= open("/home/pi/Desktop/camera_test/sequenceResults.txt","w+")
    f.write("Done. Elapsed time (s): "+str(elapsed))
    f.close()

#Setup GPIO pin 18 (6th pin from top to bottom on right hand pins) as input
# --- Input to start capturing sequence ---
GPIO.setmode(GPIO.BCM)
GPIO.setup(18,GPIO.IN)

#Setup GPIO pin 16 (3th pin from bottom to top on right hand pins) as output (RED LED)
GPIO.setmode(GPIO.BCM)
GPIO.setup(16,GPIO.OUT)

#Setup GPIO pin 20 (2nd pin from bottom to top on right hand pins) as output (YELLOW LED)
GPIO.setmode(GPIO.BCM)
GPIO.setup(20,GPIO.OUT)

#Setup GPIO pin 21 (1st pin from bottom to top on right hand pins) as output (GREEN LED)
GPIO.setmode(GPIO.BCM)
GPIO.setup(21,GPIO.OUT)

wait = True
GPIO.output(16,True)

while(wait):
    pressed = GPIO.input(18)
    if(pressed):
        GPIO.output(16,False)
        GPIO.output(20,True)
        sleep(10)
        GPIO.output(20,False)
        GPIO.output(21,True)
        startCameraSequence()
        GPIO.output(21,False)
        wait = False
        done = True

#Blinking GREEN LED indicating program done
while(done):
    sleep(1)
    GPIO.output(21,True)
    sleep(1)
    GPIO.output(21,False)
