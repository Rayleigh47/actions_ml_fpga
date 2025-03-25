from pynq import Overlay, allocate, PL
import numpy as np

PL.reset()
# Load your bitstream (adjust the filename to your design's bitstream)
overlay = Overlay("your_design.bit")

# Get handles to the DMA and HLS IP
# Adjust the instance names to those in your overlay (e.g., "axi_dma", "hls_ip")
dma = overlay.axi_dma
hls_ip = overlay.hls_ip

# Prompt user to choose the mode:
#   1: load weights and biases (from model_params.csv)
#   0: send input values layer by layer (from input_data.csv)
mode_input = input("Enter mode (1 for loading weights, 0 for inference input): ").strip()
try:
    mode = int(mode_input)
except ValueError:
    print("Invalid input; defaulting mode to 0 (inference input).")
    mode = 0

# Write the mode bit into the HLS IP via its AXI-lite register interface.
# Here we assume that the mode register is at offset 0x10.
# Adjust the offset if your design uses a different one.
hls_ip.write(0x10, mode)

if mode == 1:
    # Mode 1: load weights and biases
    filename = "model_params.csv"
    print("Loading weights from {}...".format(filename))
    
    # Read the CSV file as bytes.
    # (Assumes that your CSV file contains a continuous list of comma-separated values.)
    with open(filename, "rb") as f:
        weight_data = f.read()
    
    # Allocate a buffer for DMA transfer.
    # Use np.uint8 for a byte-wise transfer.
    buf = allocate(shape=(len(weight_data),), dtype=np.uint8)
    buf[:] = np.frombuffer(weight_data, dtype=np.uint8)
    
    # Transfer the data via DMA. The HLS IP should read the weights from its AXI stream.
    dma.sendchannel.transfer(buf)
    dma.sendchannel.wait()
    print("Weights and biases loaded into HLS IP.")
    
elif mode == 0:
    # Mode 0: send input data layer-by-layer.
    filename = "input_data.csv"
    print("Loading input data from {}...".format(filename))
    
    # Read the CSV file.
    # This example assumes the CSV contains comma-separated float values.
    # (If your inputs are organized per layer, ensure that the ordering matches your design.)
    input_data = np.loadtxt(filename, delimiter=",", dtype=np.float32)
    
    # Allocate a buffer for DMA transfer.
    buf = allocate(shape=(input_data.size,), dtype=np.float32)
    buf[:] = input_data.flatten()
    
    # Transfer input data via DMA.
    dma.sendchannel.transfer(buf)
    dma.sendchannel.wait()
    print("Input data sent to HLS IP.")
    
    # After inference, retrieve the result from the DMA output channel.
    # Adjust the shape and data type based on your expected output.
    # For example, if the output is a single label (int32) or a vector.
    out_shape = (10,)  # Example: 10 classes
    out_buf = allocate(shape=out_shape, dtype=np.int32)
    dma.recvchannel.transfer(out_buf)
    dma.recvchannel.wait()
    print("Inference result:", out_buf)
    
else:
    print("Invalid mode. Please enter either 1 or 0.")
