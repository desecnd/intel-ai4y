import onnx
import ngraph as ng
from ngraph_onnx.onnx_importer.importer import import_onnx_model

class NgraphInference:
    def __init__(self, model_path):
        self.ng_exe = self.prepare_ngraph_exe(model_path)

    def prepare_ngraph_exe(self, model_path):
        onnx_protobuf = onnx.load(model_path)
        ng_function = import_onnx_model(onnx_protobuf)
        runtime = ng.runtime(backend_name='CPU')
        return runtime.computation(ng_function)

    def infer(self, input):
        return self.ng_exe(input)[0]
