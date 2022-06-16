class SpecialImage:
    def __init__(self, image, target):
        self.fd = None
        self.image = image
        self.target = target
        self.detectedFace = None

    def set_image(self, image):
        self.image = image
        
    def set_fd(self, fd):
        self.fd = fd

    def set_detected_face(self, face):
        self.detectedFace = face
