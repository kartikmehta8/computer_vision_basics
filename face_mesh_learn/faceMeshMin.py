import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture('videos/video1.mp4')
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)

drawSpec = mpDraw.DrawingSpec(thickness=2, circle_radius=2)

while True:
    success, img = cap.read()

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        for faceLndmrks in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLndmrks, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)

    for id, lm in enumerate(faceLndmrks.landmark):
        ih, iw, ic = img.shape

        x, y = int(lm.x*iw), int(lm.y*ih)

        print(id, x, y)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv.putText(img, 'FPS: '+str(int(fps)), (50,50), cv.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)

    cv.imshow('Video', img)
    cv.waitKey(10)