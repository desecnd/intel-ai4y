import sys
import cv2
import numpy
import asyncio
import time
from threading import Thread, Event
from concurrent.futures import ThreadPoolExecutor, Future

from src.predictor import Predictor

class Config:
    def __init__(self):
        self.training_rows = 28
        self.training_cols = 28
        self.hand_rows = self.training_cols * 10
        self.hand_cols = self.training_cols * 10
        self.video_width = 640
        self.video_height = 480
        self.ulx = 20
        self.uly = 150 
        self.brx = self.ulx + self.hand_cols
        self.bry = self.uly + self.hand_rows
        self.enabled_windows = ('video', 'Letter Prediction', 'Word Suggestions',
                                'Hand Gesture', 'Skeleton on hand')

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

class Program(Thread):
    def __init__(self, stop_event, config):
        self.config = config
        self.stop_event = stop_event
        self.character_predictor = Predictor()
        self.renderer = Renderer(config)

        self.executor = ThreadPoolExecutor(max_workers=4)
        self.prediction_future = Future()
        self.prediction_future.set_result(None)

        self.message = ""

        Thread.__init__(self)
    
    def predict_from_gesture(self, frame):
        # render the current gesture that will be used for the next prediction
        hand = self.renderer.extract_hand(frame)
        self.renderer.render({'Hand Gesture': hand})

        # submit the hand extracted from the current frame for prediction
        self.prediction_future = self.executor.submit(self.character_predictor.predict, hand)

    def process_predicted_character(self, character):
        print("Predicted char: ", character)
    
    def run(self):
        while not stop_event.is_set():
            loop_start = time.time()

            frame = self.renderer.grab_full_frame()
            self.renderer.draw_hand_frame(frame)

            if self.prediction_future.done():
                # get the previous gesture prediction and process it
                char = self.prediction_future.result()
                self.process_predicted_character(char)

                self.predict_from_gesture(frame)

            self.renderer.render({'video': frame})

            loop_end = time.time()
            self.renderer.discard_frames(loop_end - loop_start)

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
