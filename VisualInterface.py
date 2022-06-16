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
    def __init__(self, window, window_title, video_source = 0):
        self.window = window
        self.window.title('Sistem de recunoastere al emotiilor in timp real.')
        self.window['background'] = '#856ff8'
        self.window.geometry("700x700")
        self.window_title = window_title
        self.video_source = video_source

        self.start_recording = False
        self.stop_recording = False
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter('output.avi', self.fourcc, 10, (480, 360))

        self.vid = VideoCapture(self.video_source)
        self.canvas = tkinter.Canvas(window, width=self.vid.width, height=self.vid.height, bg='white')
        self.canvas.pack()

        self.btn_snapshot = tkinter.Button(window, text="Snapshot", width=50, command=self.snapshot, bg='white')
        self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)

        self.btn_recording = tkinter.Button(window, text="Start recording", width=50, command=self.start_recording_method, bg='white')
        self.btn_recording.pack(anchor=tkinter.CENTER, expand=True)

        self.btn_stopRecording = tkinter.Button(window, text="Stop recording", width=50, command=self.stop_recording_method, bg='white', state='disabled')
        self.btn_stopRecording.pack(anchor=tkinter.CENTER, expand=True)

        self.delay = 5
        self.timeDelay = 1000
        self.update()

        self.window.mainloop()

    def start_recording_method(self):
        self.start_recording = True
        self.out = cv2.VideoWriter("media/video-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".avi", self.fourcc, 10, (480, 360))
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
        self.label.configure(text='Recording...'+now)
        if self.start_recording:
            self.window.after(1000, self.update_clock)
        else:
            self.label.pack_forget()

    def itsRecording(self):
        if self.start_recording:
            ret, frame = self.vid.get_frame()
            self.out.write(frame)

    def stop_recording_method(self):
        self.start_recording = False
        self.out.release()
        self.btn_stopRecording["state"] = "disable"
        self.btn_recording["state"] = "active"

    def snapshot(self):
        ret, frame = self.vid.get_frame()
        if ret:
            img_3 = np.zeros([360, 480, 3], dtype=np.uint8)
            img_3.fill(255)
            cv2.imwrite("media/frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def update(self):
        ret, frame = self.vid.get_frame()

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
            self.itsRecording()
        self.window.after(self.delay, self.update)


class VideoCapture:
    def __init__(self, video_source = 0):
        self.loaded_model = pickle.load(open('knnpickle_file', 'rb'))
        self.X = []
        self.y = []
        self.emotionQueue = []

        self.vid = cv2.VideoCapture(video_source)

        if not  self.vid.isOpened():
            raise ValueError("Unable to open video souce", video_source)

        self.width = 480
        self.height = 360

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            frame = cv2.resize(frame, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)
            if ret:
                frame = self.do_the_work(frame)
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                return ret, None
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
        if self.vid.isOpened():
            self.vid.release()


VisualInterface(tkinter.Tk(), "Tkinter and openCV")