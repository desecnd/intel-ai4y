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

class Renderer:
    def __init__(self, config, windows):
        self.window_names = windows
        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, config.video_width)
        self.cap.set(4, config.video_height)
        if not self.cap.isOpened():
            raise "Could not open video device"

    def render(self, images):
        for win, img in images.items():
            if win in self.window_names:
                cv2.imshow(win, img)

class Predictor:
    def __init__(self):
        pass

    def predict(self):
        print("predicting")
        time.sleep(1)
        return numpy.random.randn(100, 100)

class Program(Thread):
    def __init__(self, stop_event, config):
        self.config = config
        self.stop_event = stop_event
        self.character_predictor = Predictor()
        self.renderer = Renderer(config, ('main_window'))
        Thread.__init__(self)

    def run(self):
        print("starting")
        while not stop_event.is_set():
            img = self.character_predictor.predict()
            self.renderer.render({'main_window': img})

if __name__ == "__main__":
    stop_event = Event()
    config = Config()
    program = Program(stop_event, config)
    cv2.imshow('main_window', numpy.zeros((100, 100)))
    program.daemon = True
    program.start()

    while (True):
        print("waiting for user input")
        userChoice = chr(cv2.waitKey(0) & 255)
        print("key pressed: ", userChoice)

        if userChoice == 'q':
            print("sending stop event")
            stop_event.set()
            print("closing windows")
            cv2.destroyAllWindows()
            break;

    sys.exit(0)
