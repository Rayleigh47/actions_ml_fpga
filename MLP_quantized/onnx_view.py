import onnx
import netron
# Load the ONNX model
onnx_model = onnx.load('models/model.onnx')
# View the ONNX model on Netron
netron.start('models/model.onnx')
