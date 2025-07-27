import cv2

class VideoHandler:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        self.set_resolution(640, 480)

    def read_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))
        return ret, frame

    def release(self):
        self.cap.release()

    def set_resolution(self, width, height):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def is_opened(self):
        return self.cap.isOpened()
