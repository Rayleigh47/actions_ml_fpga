import os
import onnx
import numpy as np
from onnx import numpy_helper

def numpy_array_to_c_array_str(arr, integer=True):
    """
    Recursively converts a NumPy array into a C-style nested initializer list string.
    If integer is True, values are formatted as integers.
    """
    if arr.ndim == 0:
        return str(int(arr.item())) if integer else f"{arr.item():.8f}"
    elif arr.ndim == 1:
        if integer:
            return "{ " + ", ".join(str(int(x)) for x in arr) + " }"
        else:
            return "{ " + ", ".join(f"{x:.8f}" for x in arr) + " }"
    else:
        inner = ",\n    ".join(numpy_array_to_c_array_str(sub, integer=integer) for sub in arr)
        return "{ " + inner + " }"

def export_initializers_to_hpp(onnx_file, hpp_filename="model_params.hpp"):
    # Load the ONNX model
    model = onnx.load(onnx_file)
    graph = model.graph

    weight_layer = 0
    bias_layer = 0

    with open(hpp_filename, "w") as f:
        f.write("// Auto-generated model parameters for Vitis HLS\n")
        f.write(f"// Exported from ONNX model: {onnx_file}\n\n")

        # Iterate over all initializers
        for init in graph.initializer:
            arr = numpy_helper.to_array(init)
            name_lower = init.name.lower()
            # Check for weight initializers (and ignore scalars)
            if "weight" in name_lower and arr.shape != ():
                layer_name = f"weight_layer_{weight_layer}"
                print(f"Exporting {layer_name} with shape {arr.shape}")
                # Quantize: round and cast to int8
                arr_int = np.rint(arr).astype(np.int8)
                c_type = "int8_t"
                dims = "".join(f"[{dim}]" for dim in arr_int.shape)
                f.write(f"// {layer_name} shape: {arr_int.shape}\n")
                f.write(f"static const {c_type} {layer_name}{dims} = \n")
                c_array_str = numpy_array_to_c_array_str(arr_int, integer=True)
                f.write(c_array_str + ";\n\n")
                weight_layer += 1

            # Check for bias initializers
            elif "bias" in name_lower:
                layer_name = f"bias_layer_{bias_layer}"
                print(f"Exporting {layer_name} with shape {arr.shape}")
                # Quantize: round and cast to int32
                arr_int = np.rint(arr).astype(np.int32)
                c_type = "int32_t"
                dims = "".join(f"[{dim}]" for dim in arr_int.shape)
                f.write(f"// {layer_name} shape: {arr_int.shape}\n")
                f.write(f"static const {c_type} {layer_name}{dims} = \n")
                c_array_str = numpy_array_to_c_array_str(arr_int, integer=True)
                f.write(c_array_str + ";\n\n")
                bias_layer += 1

        if weight_layer == 0 and bias_layer == 0:
            f.write("// No weight or bias initializers found.\n")
    print(f"Model parameters exported to {hpp_filename}")

if __name__ == '__main__':
    # Path to onnx file and model_params.hpp
    current_dir = os.path.dirname(__file__)
    onnx_file = os.path.join(current_dir, 'model.onnx')
    hpp = os.path.join(current_dir, 'model_params.hpp')

    export_initializers_to_hpp(onnx_file, hpp)
