import onnx
import time
from utils.perf import LatencyCalc
import ngraph as ng
from ngraph_onnx.onnx_importer.importer import import_onnx_model

class NgraphInference:
    """
    Creates an nGraph based inference engine for a ONNX model.
    Optionally the engine can measure average latency time for all inferences of an instance of this engine.
    """
    def __init__(self, model_path, measure_latency = False):
        self.ng_exe = self.prepare_ngraph_exe(model_path)
        self.measure_latency = measure_latency
        
        if measure_latency:
            self.perf = LatencyCalc()

    def prepare_ngraph_exe(self, model_path):
        onnx_protobuf = onnx.load(model_path)
        ng_function = import_onnx_model(onnx_protobuf)
        runtime = ng.runtime(backend_name='CPU')
        return runtime.computation(ng_function)

    def infer(self, input):
        inference_start = time.perf_counter()
        output = self.ng_exe(input)[0]
        inference_end = time.perf_counter()

        if self.measure_latency:
            self.calculate_avg_latency(inference_start, inference_end)
        
        return output

    def calculate_avg_latency(self, inference_start, inference_end):
        avg_inference_latency = self.perf.calc_inference_latency(inference_start, inference_end)
        print("AVG nGraph latency: %.2f ms" % avg_inference_latency)
