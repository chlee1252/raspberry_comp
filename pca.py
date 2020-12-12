import cv2            #openCV 
import numpy as np    #python - advacaned math
import picamera       #picamera
import picamera.array #picamera array
from Adafruit_PCA9685 import PCA9685 # Motor driver
import time
# import RPi.GPIO as GPIO #pin info
# import time

xml = "haarcascade_frontalface_default.xml"    # 얼굴인식 xml 파일
faceCascade = cv2.CascadeClassifier(xml)       # xml파일 읽기

class Motor:
    '''
        모터 동작을 위한 class
    '''
    def __init__(self, channel, offset, add=0x40):
        # 모터 초기값 및 설정
        self.mChannel = channel
        self.offset = offset
        
        self.mPwm = PCA9685(address=add)
        self.mPwm.set_pwm_freq(50)
    
    def setPos(self, pos):
        # 각도를 받아서 모터에 전달
        pulse = (650-150) * pos / 180 + 150 + self.offset  # 각도를 모터 Position 값으로 변형 하는 식
        self.mPwm.set_pwm(self.mChannel, 0, int(pulse))
    
    def cleanUp(self):
        # 프로그램이 끝났을때 90도로 바꾸고 sleep
        self.setPos(90)
        time.sleep(1)

# 모터 시작
servo = Motor(channel=0, offset=-10)
servo.setPos(90)


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
            camera.capture(stream, "bgr", use_video_port=True) # 카메라 불러오기
            #_, frame = cap.read()    -> [color code] frame -> stream.array

            # hsv_frame = cv2.cvtColor(stream.array, cv2.COLOR_BGR2HSV) 색 인식 할때
            
            # 얼굴 감지 시작
            gray = cv2.cvtColor(stream.array, cv2.COLOR_BGR2GRAY)
            
            # 얼굴 감지 옵션
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),   # 얼굴 크기 최솟값
                flags = cv2.CASCADE_SCALE_IMAGE)
            
            # 얼굴이 있는지 없는지 확인 (배열 길이 -> len())
            if len(faces):
                # faces = np.sort(faces, axis=0)[::-1]
                for (x,y,w,h) in faces:
                    cv2.rectangle(stream.array, (x,y), (x+w, y+h), (0, 255, 0), 2) # 사각형 테두리 그리는거
                    x_medium = (x+x+w) // 2                      # 얼굴 가운데 x축
                    y_medium = (y+y+h) // 2                      # 얼굴 가운데 y춗
                    break
            else:
                # 얼굴 감지 안되면 -> 원상 복귀
                position = 90
                x_medium = x_center
                
            #red color
#             low_red = np.array([161, 155, 84])
#             high_red = np.array([179, 255,255])
#             red_mask = cv2.inRange(hsv_frame, low_red, high_red)
#             contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#             contours = sorted(contours, key=lambda x:cv2.contourArea(x), reverse=True)
#             for cnt in contours:
#                 (x, y, w, h) = cv2.boundingRect(cnt)
#                 #cv2.rectangle(stream.array, (x,y), (x+w, y+h), (0, 255, 0), 2)
#                 x_medium = (x+x+w) // 2
#                 y_medium = (y+y+h) // 2
#                 break
        
            # 라인 그리기
            cv2.line(stream.array, (x_medium, 0), (x_medium, 480), (0, 255, 0), 2)
            cv2.line(stream.array, (0, y_medium), (800, y_medium), (0, 255, 0), 2)

            # 비디오 송출
            cv2.imshow("Video",stream.array) # imshow camera -> color -> window
            #cv2.imshow("mask", red_mask)

            if 0 < position < 180:
                if x_medium < x_center-30:
                    position += 1
                elif x_medium > x_center+30:
                    position -= 1
            else:
                x_medium = x_center
                position = 90
# #                 
            print(position)   # 디버그 코드

            # 포지션 바꿈      
            servo.setPos(position)

            # ESC 종료
            key = cv2.waitKey(1)
            if key == 27:
                break

            # picamera 필수
            stream.seek(0)
            stream.truncate()

    
#cap.release()

# 클린업
servo.cleanUp()
cv2.destroyAllWindows()

