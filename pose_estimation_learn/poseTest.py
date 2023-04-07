import cv2 as cv
from poseModule import poseDetector
import time

cap = cv.VideoCapture(0)
pTime = 0

detector = poseDetector()

while True:
    success, img = cap.read()
    img = detector.findPose(img)
    landmarkList = detector.findPosition(img)
    
    if len(landmarkList) != 0:
        print(landmarkList)

        # 4 for ex - is a type of body part number (for tracking)
        cv.circle(img, (landmarkList[4][1], landmarkList[4][2]), 10, (0,0,255), cv.FILLED)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv.putText(img, str(int(fps)), (70,50), cv.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
    cv.imshow('Video', img)

    cv.waitKey(5)