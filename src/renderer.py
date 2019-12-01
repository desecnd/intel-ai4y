import cv2
import numpy
import time

class Renderer:
    def __init__(self, config):
        self.config = config

        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, config.video_width)
        self.cap.set(4, config.video_height)
        if not self.cap.isOpened():
            raise "Could not open video device"

        self.calc_tpf(20)

        self.open_windows()

    def render(self, images):
        for win, img in images.items():
            if win in self.config.enabled_windows:
                cv2.imshow(win, img)

    def grab_full_frame(self):
        ret, frame = self.cap.read()
        return frame

    def extract_hand(self, frame):
        hand = frame[self.config.uly:self.config.bry, self.config.ulx:self.config.brx,:]
        return hand

    def open_windows(self):
        blank_img = numpy.zeros((self.config.hand_rows, self.config.hand_cols))
        self.render({'video': numpy.zeros((self.config.video_height, self.config.video_width)),
                     'Hand Gesture': blank_img,
                     'Skeleton on hand': blank_img,
                     'Letter Prediction': blank_img,
                     'Word Suggestions': blank_img})
                     
        cv2.moveWindow('video', 500, 125)
        cv2.moveWindow('Hand Gesture', 0, 0)
        cv2.moveWindow('Skeleton on hand', 0, 450)
        cv2.moveWindow('Letter Prediction', 1200, 0)
        cv2.moveWindow('Word Suggestions', 1200, 450)

    def calc_tpf(self, test_frames):
        s = time.time()
        for i in range(test_frames):
            ret, frame = self.cap.read()
        e = time.time()
        self.tpf = (e - s) / test_frames

    def discard_frames(self, elapsed_time):
        frames_to_discard = int(elapsed_time / self.tpf)
        # print("Frames to discard:", frames_to_discard)
        for i in range(frames_to_discard):
            ret, frame = self.cap.read()

    def draw_hand_frame(self, frame):
        cv2.rectangle(frame, (self.config.ulx, self.config.uly), 
                             (self.config.brx, self.config.bry), 
                             (0, 0, 255), 2)