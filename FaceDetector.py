import cv2
import cv2 as cv
from HOGFeatures import HogFeature
from SpecialImage import SpecialImage


class FaceDetector:
    def detect(self, data):
        try:
            special_image = SpecialImage(data.image, data.target)
            face_cascade = cv.CascadeClassifier('FaceDetection.xml')
            # mouth = face_cascade.detectMultiScale(image, 1.05, 38)
            face = face_cascade.detectMultiScale(data.image, scaleFactor=1.05, minNeighbors=5, minSize=(130, 130),
                                                 flags=cv.CASCADE_SCALE_IMAGE)
            (x, y, w, h) = face[0]
            # for (x, y, w, h) in face:
            #     cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            faceImage = data.image[y:y + h, x:x + w]
            (X, Y, Z) = faceImage.shape
            mouth_cascade = cv.CascadeClassifier("MouthDetection.xml")
            mouth = mouth_cascade.detectMultiScale(faceImage, scaleFactor=1.05, minNeighbors=20, minSize=(20, 20),
                                                   flags=cv.CASCADE_DO_ROUGH_SEARCH)
            (XMin, YMin, WMin, HMin) = (0, 0, 0, 0)
            minDif = Y
            for (x, y, w, h) in mouth:
                if minDif > Y - y:
                    (XMin, YMin, WMin, HMin) = (x, y, w, h)
                    minDif = Y - y
            # cv2.rectangle(faceImage, (XMin, YMin), (XMin + WMin, YMin + HMin), (255, 0, 0), 2)

            crop_img = faceImage[YMin:YMin + HMin, XMin:XMin + WMin]
            resized_img = FaceDetector.resize_image(crop_img)
            special_image.set_image(resized_img)
            # cv2.imshow('face_detected1.png', faceImage)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            hogFeatures = HogFeature()
            hog_special_image = hogFeatures.get_hog_features(special_image)
            hog_special_image.set_detected_face(face[0])
            return hog_special_image
        except:
            return None

    def detectLips(self, image):
        try:
            face_cascade = cv.CascadeClassifier('FaceDetection.xml')
            # mouth = face_cascade.detectMultiScale(image, 1.05, 38)
            face = face_cascade.detectMultiScale(image, scaleFactor=1.05, minNeighbors=5, minSize=(130, 130),
                                                 flags=cv.CASCADE_SCALE_IMAGE)
            (x, y, w, h) = face[0]
            # for (x, y, w, h) in face:
            #     cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            faceImage = image[y:y + h, x:x + w]
            mouth_cascade = cv.CascadeClassifier("MouthDetection.xml")
            mouth = mouth_cascade.detectMultiScale(faceImage, scaleFactor=1.05, minNeighbors=20, minSize=(20, 20),
                                                   flags=cv.CASCADE_DO_ROUGH_SEARCH)
            (X, Y, Z) = faceImage.shape
            (XMin, YMin, WMin, HMin) = (0, 0, 0, 0)
            minDif = Y
            for (x, y, w, h) in mouth:
                if minDif > Y - y:
                    (XMin, YMin, WMin, HMin) = (x, y, w, h)
                    minDif = Y - y
            crop_img = faceImage[YMin:YMin + HMin, XMin:XMin + WMin]
            resized_img = FaceDetector.resize_image(crop_img)
            special_image = SpecialImage(image, 0)
            special_image.set_image(resized_img)
            # cv2.imshow('face_detected1.png', faceImage)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            hogFeatures = HogFeature()
            hog_special_image = hogFeatures.get_hog_features(special_image)
            hog_special_image.set_detected_face(face[0])
        except:
            return None

    def detectFace(self, image):
        try:
            face_cascade = cv.CascadeClassifier('FaceDetection.xml')
            # mouth = face_cascade.detectMultiScale(image, 1.05, 38)
            face = face_cascade.detectMultiScale(image, scaleFactor=1.05, minNeighbors=5, minSize=(130, 130),
                                                 flags=cv.CASCADE_SCALE_IMAGE)

            for (x, y, w, h) in face:
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.imshow('face_detected1.png', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            return None

    def resize_image(img):
        return cv.resize(img, (70, 40), interpolation=cv.INTER_LINEAR)
