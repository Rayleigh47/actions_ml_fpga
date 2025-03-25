import onnx
import netron
# Load the ONNX model
onnx_model = onnx.load('model.onnx')
# View the ONNX model on Netron
netron.start('model.onnx')
