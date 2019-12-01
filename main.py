import sys
import cv2
import numpy
import asyncio
import time
from threading import Thread, Event


class Config:
    def __init__(self):
        self.training_rows = 28
        self.training_cols = 28
        self.hand_rows = self.training_cols * 10
        self.hand_cols = self.training_cols * 10
        self.video_width = 640
        self.video_height = 480
        self.enabled_windows = ('video', 'Letter Prediction', 'Word Suggestions',
                                'Hand Gesture', 'Skeleton on hand')

class Renderer:
    def __init__(self, config):
        self.config = config

        self.open_windows()
        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, config.video_width)
        self.cap.set(4, config.video_height)
        if not self.cap.isOpened():
            raise "Could not open video device"

    def render(self, images):
        for win, img in images.items():
            if win in self.config.enabled_windows:
                cv2.imshow(win, img)

    def open_windows(self):
        blank_img = numpy.zeros((100, 100))
        self.render({'video': blank_img,
                     'Hand Gesture': blank_img,
                     'Skeleton on hand': blank_img,
                     'Letter Prediction': blank_img,
                     'Word Suggestions': blank_img})
                     
        cv2.moveWindow('video', 500, 125)
        cv2.moveWindow('Hand Gesture', 0, 0)
        cv2.moveWindow('Skeleton on hand', 0, 450)
        cv2.moveWindow('Letter Prediction', 1200, 0)
        cv2.moveWindow('Word Suggestions', 1200, 450)


class Predictor:
    def __init__(self):
        pass

    def predict(self):
        time.sleep(1)
        return numpy.random.randn(100, 100)

class Program(Thread):
    def __init__(self, stop_event, config):
        self.config = config
        self.stop_event = stop_event
        self.character_predictor = Predictor()
        self.renderer = Renderer(config)
        Thread.__init__(self)

    def run(self):
        while not stop_event.is_set():
            img = self.character_predictor.predict()
            self.renderer.render({'video': img})

if __name__ == "__main__":
    stop_event = Event()
    config = Config()

    program = Program(stop_event, config)
    program.daemon = True
    program.start()

    while (True):
        userChoice = chr(cv2.waitKey(0) & 255)

        if userChoice == 'q':
            stop_event.set()
            cv2.destroyAllWindows()
            break;

    sys.exit(0)
