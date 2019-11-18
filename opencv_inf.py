from cv2.dnn import readNetFromCaffe

class OpencvInference:
    def __init__(self, model_path, weights_path):
        self.net = readNetFromCaffe(model_path, weights_path)

    def infer(self, input):
        self.net.setInput(input)
        return self.net.forward()
