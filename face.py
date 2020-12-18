import numpy as np
import imutils
import time
import cv2
import os

# 얼굴 감지 엔진을 훈련 시킬 데이터
protoPath = "deploy.prototxt"
modelPath = "res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# 웹캠을 사용하여 비디오 시작
vs = cv2.VideoCapture(0)

while True:
    # 비디오 사이즈
    _, frame = vs.read()
    frame = imutils.resize(frame,width=600)
    (h, w) = frame.shape[:2]
    
    # blob 만들기: 빛, 색, 주위와 다른 부위를 찾는 것
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),1.0, (300,300),(104.0,177.0,123.0), swapRB=False, crop=False)
    
    # 받아 온 이미지에서 OpenCV의 딥러닝 베이스 얼굴 감지 엔진을 사용하여
    # 얼굴 찾기
    detector.setInput(imageBlob)
    detections = detector.forward()

    # 적어도 하나의 얼굴이 찾아졌을때
    if len(detections) > 0:
        # 찾아진 얼굴에서 가장 확률이 높은 얼굴 확률 가져오기
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0,0,i,2]

        # 찾아 온 확률이 50% 이상일 때
        if confidence > 0.5:
            # 얼굴 크기에 따라 박스 크기 만들기
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # 얼굴 ROI값 (얼굴의 좌표) 추출 후 가지고 오기
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]
            
            # 얼굴이 충분히 클때만 네모 표시하기
            if fW < 20 or fH < 20:
                continue
            
            # 화면에 네모 띄우기
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                (0, 0, 255), 2)
        else:
            # 모터 돌리기
            print("No Faces")

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.release()