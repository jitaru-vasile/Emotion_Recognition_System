import pickle
import cv2
from FaceDetector import FaceDetector
from SpecialImage import SpecialImage
import HogDescriptor

image = cv2.imread('D:\\Facultate\\Licenta\\testImage2.jpg')
faceDetector = FaceDetector()

hogImage = faceDetector.detectFace(image)
hogImage2 = faceDetector.detectLips(image)
