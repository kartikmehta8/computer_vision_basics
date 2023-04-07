import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture('videos/video1.mp4')
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)

while True:
    success, img = cap.read()

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img, detection)

            ih, iw, ic = img.shape

            bboxC = detection.location_data.relative_bounding_box
            bbox = int(bboxC.xmin*iw), int(bboxC.ymin*ih), int(bboxC.width*iw), int(bboxC.height*ih)

            cv.rectangle(img, bbox, (255,0,255), 2)
            cv.putText(img, str(int(detection.score[0]*100))+'%', (bbox[0], bbox[1]-20), cv.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv.putText(img, 'FPS: '+str(int(fps)), (50,50), cv.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)

    cv.imshow('Video', img)
    cv.waitKey(10)