import cv2
from playsound import playsound
from mtcnn import MTCNN
import dlib
from math import dist
import keyboard


class OfflineTraining:

    @staticmethod
    def audio_listen(audio):
        playsound(audio)

    @staticmethod
    def snapshot():
        cam_port = 0
        cam = cv2.VideoCapture(cam_port)
        result, image = cam.read()
        return result, image

    @staticmethod
    def save_snap(img, str):
        cv2.imshow(str, img)
        cv2.imwrite(str + ".png", img)
        cv2.destroyWindow(str)

    @staticmethod
    def face_detect(img, obj):
        output = obj.detect_faces(img)
        return output

    @staticmethod
    def show(img, str):
        cv2.imshow(str, img)

    @staticmethod
    def boundingbox(img, x, y, width, height, str):
        cv2.rectangle(img, pt1=(x, y), pt2=(x + width, y + height), color=(255, 0, 0), thickness=3)
        dlibrect = dlib.rectangle(x, y, x + width, y + height)
        # OfflineTraining.show(img, str)
        return dlibrect

    @staticmethod
    def grayScale(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray

    @staticmethod
    def landmarks(face_landmarks, img):
        eyepoints = []
        for i in range(36, 48):
            x = face_landmarks.part(i).x
            y = face_landmarks.part(i).y
            eyepoints.append([x, y])
            cv2.circle(img, (x, y), 1, (0, 255, 255), 1)
        OfflineTraining.show(img, "Facial Landmarks")
        cv2.destroyAllWindows()
        return eyepoints

    @staticmethod
    def euclideanDistance(point1, point2):
        x = dist(point1, point2)
        return x

    @staticmethod
    def EARCaluculation(x1, x2, x3):
        ear = (x1 + x2) / x3
        return ear

    @staticmethod
    def P80Caluculation(ear):
        minEar = (ear * 80) / 100
        return minEar


class OnlineMonitoring(OfflineTraining):

    @staticmethod
    def VideoCap():
        cap = cv2.VideoCapture(0)
        return cap

    @staticmethod
    def onlineFunc(cap):
        while True:
            Ndrowsy = 0
            for i in range(10):
                res, img = cap.read()
                if res:
                    onm.save_snap(img, "frame")
                    img = cv2.imread("frame.png")
                    out = onm.face_detect(img, detectors)
                    if len(out) != 0:
                        x, y, width, height = out[0]['box']
                        dlibrect = onm.boundingbox(img, x, y, width, height, "frame")
                        gray = onm.grayScale(img)
                        # onm.show(gray, "Gray scale")
                        face_landmarks = face_detector(gray, dlibrect)
                        eyepoints = onm.landmarks(face_landmarks, img)
                        print(eyepoints)
                        x1 = onm.euclideanDistance(eyepoints[1], eyepoints[5])
                        x2 = onm.euclideanDistance(eyepoints[2], eyepoints[4])
                        x3 = onm.euclideanDistance(eyepoints[0], eyepoints[3])
                        EARL = onm.EARCaluculation(x1, x2, x3)
                        print("EAR of left eye closed", EARL)

                        x4 = onm.euclideanDistance(eyepoints[7], eyepoints[11])
                        x5 = onm.euclideanDistance(eyepoints[8], eyepoints[10])
                        x6 = onm.euclideanDistance(eyepoints[6], eyepoints[9])

                        EARR = onm.EARCaluculation(x4, x5, x6)
                        print("EAR of left eye opened", EARR)
                        EAR = (EARR + EARL) / 2
                        print("AVg EAR", EAR)
                        if EAR < off.P80Caluculation(EARO) or EAR < EARC:
                            Ndrowsy = Ndrowsy + 1
                    else:
                        Ndrowsy = Ndrowsy + 1
                else:
                    Ndrowsy = Ndrowsy + 1
            perclos = Ndrowsy / 10
            if perclos > 0.4:
                onm.audio_listen("alarm.mp3")
                break
            else:
                print("Not drowsy")


off = OfflineTraining()
# audio of close eyes
off.audio_listen('Closedframe.mp3')

# snapshot of closed frame
result, image = off.snapshot()
if result:
    off.save_snap(image, "Closedframe")
else:
    print("Picture not captured")

# audio of open eyes
off.audio_listen('Openframe.mp3')

# snapshot of open frame
result, image = off.snapshot()
if result:
    off.save_snap(image, "Openframe")
else:
    print("Picture not captured")

# face detecting
detectors = MTCNN()
face_detector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# open frame
img = cv2.imread("Openframe.png")
out = off.face_detect(img, detectors)
if len(out) != 0:
    x, y, width, height = out[0]['box']
    dlibrect = off.boundingbox(img, x, y, width, height, "Openframe")
    gray = off.grayScale(img)
    #  off.show(gray, "Gray scale")
    face_landmarks = face_detector(gray, dlibrect)
    eyepoints = off.landmarks(face_landmarks, img)
    print(eyepoints)
    x1 = off.euclideanDistance(eyepoints[1], eyepoints[5])
    x2 = off.euclideanDistance(eyepoints[2], eyepoints[4])
    x3 = off.euclideanDistance(eyepoints[0], eyepoints[3])
    EARLO = off.EARCaluculation(x1, x2, x3)
    print("EAR of left eye opened", EARLO)

    x4 = off.euclideanDistance(eyepoints[7], eyepoints[11])
    x5 = off.euclideanDistance(eyepoints[8], eyepoints[10])
    x6 = off.euclideanDistance(eyepoints[6], eyepoints[9])

    EARRO = off.EARCaluculation(x4, x5, x6)
    print("EAR of right eye opened", EARRO)
    EARO = (EARRO + EARLO) / 2
    print("AVg EAR", EARO)
else:
    print("Face not found") 

# closed frame
img = cv2.imread("Closedframe.png")
out = off.face_detect(img, detectors)
if len(out) != 0:
    x, y, width, height = out[0]['box']
    dlibrect = off.boundingbox(img, x, y, width, height, "Closedfrrame")
    gray = off.grayScale(img)
    # off.show(gray, "Gray scale")
    face_landmarks = face_detector(gray, dlibrect)
    eyepoints = off.landmarks(face_landmarks, img)
    print(eyepoints)
    x1 = off.euclideanDistance(eyepoints[1], eyepoints[5])
    x2 = off.euclideanDistance(eyepoints[2], eyepoints[4])
    x3 = off.euclideanDistance(eyepoints[0], eyepoints[3])
    EARLC = off.EARCaluculation(x1, x2, x3)
    print("EAR of left eye closed", EARLC)

    x4 = off.euclideanDistance(eyepoints[7], eyepoints[11])
    x5 = off.euclideanDistance(eyepoints[8], eyepoints[10])
    x6 = off.euclideanDistance(eyepoints[6], eyepoints[9])

    EARRC = off.EARCaluculation(x4, x5, x6)
    print("EAR of right eye closed", EARRC)
    EARC = (EARRC + EARLC) / 2
    print("AVg EAR", EARC)
    minEar = off.P80Caluculation(EARO)
else:
    print("Face not found")

# Online monitoring
onm = OnlineMonitoring()
cap = onm.VideoCap()
onm.onlineFunc(cap)
