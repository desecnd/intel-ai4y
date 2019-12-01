import cv2
import time
import hand_processing
from threading import Thread, Event

class InferenceThread(Thread):
    def __init__(self, app_quit_event):
        self.terminate_thread = app_quit_event
        Thread.__init__(self)

    def run(self):
        while not (self.terminate_thread.is_set()):
            print("Inference thread")
            time.sleep(1)


# def predict_character():
#     cap = cv2.VideoCapture(0)
#     cap.set(3, 640)
#     cap.set(4, 480)