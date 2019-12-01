import sys
import cv2
import numpy
import asyncio
import time
from threading import Thread, Event
from concurrent.futures import ThreadPoolExecutor, Future

from src.predictor import Predictor
from src.renderer import Renderer
from src.config import Config

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

            # render each frame in the video window for live feedback
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
