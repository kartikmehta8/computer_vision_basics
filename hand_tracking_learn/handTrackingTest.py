import cv2 as cv
from handTrackingModule import handDetector
import time

prevTime = 0
currTime = 0
cap = cv.VideoCapture(0)

detector = handDetector()

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    landmarkList = detector.findPosition(img)
        
    if len(landmarkList) != 0:
        print(landmarkList[4])

    currTime = time.time()
    fps = 1/(currTime-prevTime)
    prevTime = currTime

    cv.putText(img, str(int(fps)), (10,70), cv.FONT_HERSHEY_PLAIN, 3, (255,0,255), 2)

    cv.imshow('Video', img)
    cv.waitKey(1)