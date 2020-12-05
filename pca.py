import cv2            #openCV 
import numpy as np    #python - advacaned math
import picamera       #picamera
import picamera.array #picamera array
from Adafruit_PCA9685 import PCA9685 # Motor driver
import time
# import RPi.GPIO as GPIO #pin info
# import time

class Motor:
    def __init__(self, channel, offset):
        self.mChannel = channel
        self.offset = offset
        
        self.mPwm = PCA9685(address=0x40)
        self.mPwm.set_pwm_freq(60)
    
    def setPos(self, pos):
        pulse = (650-150) * pos / 180 + 150 + self.offset
        self.mPwm.set_pwm(self.mChannel, 0, int(pulse))
    
    def cleanUp(self):
        self.setPos(90)
        time.sleep(1)

servo = Motor(channel=0, offset=-10)
#servo = Motor(channel=1, offset=-10)
servo.setPos(5)


with picamera.PiCamera() as camera:    # load picamera -> camera
    with picamera.array.PiRGBArray(camera) as stream:   # stream data -> stream
        camera.resolution = (320, 220)
        camera.framerate = 30
        rows, cols = camera.resolution.height, camera.resolution.width
        x_medium = cols // 2
        y_medium = rows // 2
        
        x_center = cols // 2
        
        #print(camera.resolution.width)

#cap = cv2.VideoCapture(0) openCV -> webcam, usb camera
        position = 90
        while True:
            camera.capture(stream, "bgr", use_video_port=True) #use picamera
            #_, frame = cap.read()    -> [color code] frame -> stream.array

            hsv_frame = cv2.cvtColor(stream.array, cv2.COLOR_BGR2HSV)
            
            #red color
            low_red = np.array([161, 155, 84])
            high_red = np.array([179, 255,255])
            red_mask = cv2.inRange(hsv_frame, low_red, high_red)
            contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=lambda x:cv2.contourArea(x), reverse=True)
            
            if not contours:
                position = 90
                x_medium = x_center
            else:
                for cnt in contours:
                    (x, y, w, h) = cv2.boundingRect(cnt)
                    #cv2.rectangle(stream.array, (x,y), (x+w, y+h), (0, 255, 0), 2)
                    x_medium = (x+x+w) // 2
                    y_medium = (y+y+h) // 2
                    break
        
            cv2.line(stream.array, (x_medium, 0), (x_medium, 480), (0, 255, 0), 2)
            cv2.line(stream.array, (0, y_medium), (800, y_medium), (0, 255, 0), 2)
            cv2.imshow("Video",stream.array) # imshow camera -> color -> window
            #cv2.imshow("mask", red_mask)
            
            key = cv2.waitKey(1)
            if key == 27:
                break
            
            if 0 < position < 180:
                if x_medium < x_center-20:
                    position += 2
                elif x_medium > x_center+20:
                    position -= 2
            else:
                x_medium = x_center
                position = 90
                
            print(position)
                            
            servo.setPos(position)
            # picamera required
            stream.seek(0)
            stream.truncate()

    
#cap.release()
servo.cleanUp()
cv2.destroyAllWindows()


