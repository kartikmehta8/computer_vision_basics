import cv2 as cv
import mediapipe as mp
import time


class FaceMeshDetector():

    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):
        
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        # minDetectionCon & minTrackCon not used
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=2, circle_radius=2)


    def findFaceMesh(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)

        faces = []
        if results.multi_face_landmarks:
            for faceLndmrks in results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLndmrks, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec, self.drawSpec)

                face = []

                for id, lm in enumerate(faceLndmrks.landmark):
                    ih, iw, ic = img.shape

                    x, y = int(lm.x*iw), int(lm.y*ih)

                    # cv.putText(img, str(id), (x, y), cv.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
                    face.append([x,y])
            
                faces.append(face)

        return img, faces

def main():

    cap = cv.VideoCapture(0)
    pTime = 0

    detector = FaceMeshDetector()

    while True:
        success, img = cap.read()

        img, faces = detector.findFaceMesh(img)
        
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv.putText(img, 'FPS: '+str(int(fps)), (50,50), cv.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)

        cv.imshow('Video', img)
        cv.waitKey(10)


if __name__ == '__main__':
    main()