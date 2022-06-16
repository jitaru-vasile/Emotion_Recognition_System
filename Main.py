import pickle
import cv2
from FaceDetector import FaceDetector
from SpecialImage import SpecialImage
import collections

loaded_model = pickle.load(open('knnpickle_file', 'rb'))
X = []
y = []
emotionQueue = []
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 10, (480, 360))
recording = 0
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)
    if ret:
        specialImage = SpecialImage(frame, 1)
        faceDetector = FaceDetector()
        hogImage = faceDetector.detect(specialImage)
        if hogImage is not None:
            X = []
            y = []
            X.append(hogImage.fd)
            emotions = ["angry", "disgusted", "fear", "happy", "neutral", "sad", "surprised"]
            emotionId = loaded_model.predict(X)[0]
            (x, yy, w, h) = hogImage.detectedFace
            cv2.rectangle(frame, (x, yy), (x + w, yy + h), (255, 0, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (x + w, yy + h)
            thickness = 1
            emotionQueue.append(emotions[emotionId])
            filteredEmotion = max(set(emotionQueue), key=emotionQueue.count)
            cv2.putText(frame, filteredEmotion,
                        org, font, 1,
                        (255, 0, 0), thickness,
                        cv2.LINE_AA)

            if emotionQueue.__len__() == 10:
                emotionQueue.pop(0)
    if recording == 1:
        out = cv2.VideoWriter('output.avi', fourcc, 10, (480, 360))
        out.write(frame)
        recording = recording + 1
    if recording == 2:
        out.write(frame)
    if recording == 3:
        out.release()

    cv2.imshow('Input', frame)
    c = cv2.waitKey(1)
    if c == ord('r'):
        recording = 1
        print("recording started")
    if c == ord('x'):
        recording = 3
        print("stop recording... saving the video")
    if c == 27:
        break
cap.release()
out.release()
cv2.destroyAllWindows()
