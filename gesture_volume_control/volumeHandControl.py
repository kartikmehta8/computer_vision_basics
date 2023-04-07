import cv2 as cv
import time
import numpy as np
from handTrackingModule import handDetector
import math

######################
wCam, hCam = 640, 480
######################

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0

detector = handDetector()

while True:
    success, img = cap.read()

    img = detector.findHands(img)

    lndmrkList = detector.findPosition(img, draw=False)

    if len(lndmrkList) != 0:
        print(lndmrkList[4])
        print(lndmrkList[8])

        # Thumb
        x1, y1 = lndmrkList[4][1], lndmrkList[4][2]
        # Forefinger
        x2, y2 = lndmrkList[8][1], lndmrkList[8][2]

        cx,cy = (x1+x2)//2, (y1+y2)//2

        cv.circle(img, (x1,y1), 15, (255,0,255), cv.FILLED)
        cv.circle(img, (x2,y2), 15, (255,0,255), cv.FILLED)
        cv.line(img, (x1,y1), (x2,y2), (255,0,255), 3)
        cv.circle(img, (cx,cy), 10, (255,0,255), cv.FILLED)

        length = math.hypot(x2-x1, y2-y1)
        
        if length < 50:
            cv.circle(img, (cx,cy), 10, (0,255,0), cv.FILLED)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv.putText(img, 'FPS: '+str(int(fps)), (40,70), cv.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)

    cv.imshow('Video', img)
    cv.waitKey(1)