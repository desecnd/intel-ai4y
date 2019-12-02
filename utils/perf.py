class LatencyCalc:
    def __init__(self):
        self.total_latency = 0
        self.inferences = 0
        self.avg_inference_latency = 0

    def calc_inference_latency(self, inference_start, inference_end):
        last_inference_latency = inference_end - inference_start
        self.total_latency += last_inference_latency

        self.inferences += 1
        avg_inference_latency = self.total_latency / self.inferences

        # return in milliseconds
        return avg_inference_latency * 1000
