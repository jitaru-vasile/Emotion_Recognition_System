import pickle
import time
import tkinter

import PIL.Image
import PIL.ImageTk
import cv2

from FaceDetector import FaceDetector
from SpecialImage import SpecialImage
import numpy as np


class VisualInterface:
    def __init__(self, windowTk, title, vidSource):
        self.window = windowTk
        self.window['background'] = '#856ff8'
        self.window.geometry("700x700")
        self.window.title(title)
        self.video_source = vidSource

        self.start_recording = False
        self.stop_recording = False
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')

        self.videoCapture = CustomVideoCapture(self.video_source)
        self.canvas = tkinter.Canvas(self.window,
                                     width=self.videoCapture.width,
                                     height=self.videoCapture.height,
                                     bg='white')
        self.canvas.pack()

        self.out = cv2.VideoWriter('output.avi', self.fourcc, self.videoCapture.getCameraFPS(), (480, 360))

        self.btn_photo = tkinter.Button(self.window,
                                        text="Snapshot",
                                        command=self.take_photo,
                                        bg='white',
                                        width=50, )
        self.btn_photo.pack(anchor=tkinter.CENTER, expand=True)

        self.btn_recording = tkinter.Button(self.window,
                                            text="Start recording",
                                            command=self.start_recording_method,
                                            bg='white',
                                            width=50)
        self.btn_recording.pack(anchor=tkinter.CENTER, expand=True)

        self.btn_stopRecording = tkinter.Button(self.window, text="Stop recording",
                                                command=self.stop_recording_method, bg='white', state='disabled',
                                                width=50)
        self.btn_stopRecording.pack(anchor=tkinter.CENTER, expand=True)

        self.delay = 5
        self.timeDelay = 1000
        self.update()

        self.window.mainloop()

    def start_recording_method(self):
        self.start_recording = True
        self.out = cv2.VideoWriter("media/video-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".avi", self.fourcc, self.videoCapture.getCameraFPS(),
                                   (480, 360))
        self.label = tkinter.Label(text="", font=('Helvetica', 20), fg='#856ff8', bg='white')
        self.label.pack()
        self.btn_stopRecording["state"] = "active"
        self.btn_recording["state"] = "disable"
        self.timeStart = time.time()
        self.update_clock()

    def update_clock(self):
        now = -self.timeStart + time.time()
        local_now = time.localtime(now)
        now = time.strftime('%M:%S', local_now)
        self.label.configure(text='Recording...' + now)
        if self.start_recording:
            self.window.after(1000, self.update_clock)
        else:
            self.label.pack_forget()

    def itsRecording(self):
        if self.start_recording:
            ret, frame = self.videoCapture.get_frame()
            self.out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def stop_recording_method(self):
        self.start_recording = False
        self.out.release()
        self.btn_stopRecording["state"] = "disable"
        self.btn_recording["state"] = "active"

    def take_photo(self):
        returnValue, frame = self.videoCapture.get_frame()
        if returnValue:
            image = np.zeros([360, 480, 3], dtype=np.uint8)
            image.fill(255)
            cv2.imwrite("media/image-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg",
                        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def update(self):
        returnValue, frame = self.videoCapture.get_frame()

        if returnValue:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
            self.itsRecording()
        self.window.after(self.delay, self.update)


class CustomVideoCapture:
    def __init__(self, video_source=0):
        self.loaded_model = pickle.load(open('knnpickle_file', 'rb'))
        self.X = []
        self.y = []
        self.emotionQueue = []
        self.width = 480
        self.height = 360
        self.frameCapture = cv2.VideoCapture(video_source)
        self.fps = 10

        if not self.frameCapture.isOpened():
            raise ValueError("Could not open web camera...", video_source)

    def get_frame(self):
        if self.frameCapture.isOpened():
            returnValue, frame = self.frameCapture.read()
            frame = cv2.resize(frame, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)
            if returnValue:
                frame = self.do_the_work(frame)
                return returnValue, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                return returnValue, None
        else:
            return None, None

    def do_the_work(self, frame):
        specialImage = SpecialImage(frame, 1)
        faceDetector = FaceDetector()
        hogImage = faceDetector.detect(specialImage)
        if hogImage is not None:
            self.X = []
            self.y = []
            self.X.append(hogImage.fd)
            emotions = ["angry", "disgusted", "fear", "happy", "neutral", "sad", "surprised"]
            emotionId = self.loaded_model.predict(self.X)[0]
            (x, yy, w, h) = hogImage.detectedFace
            cv2.rectangle(frame, (x, yy), (x + w, yy + h), (255, 0, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (x + w, yy + h)
            thickness = 1
            self.emotionQueue.append(emotions[emotionId])
            filteredEmotion = max(set(self.emotionQueue), key=self.emotionQueue.count)
            cv2.putText(frame, filteredEmotion,
                        org, font, 1,
                        (255, 0, 0), thickness,
                        cv2.LINE_AA)
            if self.emotionQueue.__len__() == 10:
                self.emotionQueue.pop(0)
        return frame

    def __del__(self):
        if self.frameCapture.isOpened():
            self.frameCapture.release()

    def getCameraFPS(self):
        return self.fps


VisualInterface(tkinter.Tk(), "Sistem de recunoastere al emotiilor in timp real", 0)
