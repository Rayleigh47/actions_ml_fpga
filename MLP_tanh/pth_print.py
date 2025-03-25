import torch
import numpy as np
import torch.nn as nn

# Define the MLP model (same architecture as in your original script)
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=2):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        return self.network(x)

# Load the saved PyTorch model
model = MLP(72, 64, 10)  # Define your model structure
state_dict = torch.load("model.pth", map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()

# Extract state dictionary (weights and biases)
state_dict = model.state_dict()

# Define the C++ header file
header_filename = "model_params.hpp"

def array_to_brace_str(arr):
    """
    Recursively convert a NumPy array into a string with nested braces.
    """
    # If it's 1-D, output with braces.
    if arr.ndim == 1:
        return "{" + ", ".join(str(x) for x in arr.tolist()) + "}"
    
    # For multi-dimensional arrays, process each sub-array recursively.
    inner = ", ".join(array_to_brace_str(sub_arr) for sub_arr in arr)
    return "{" + inner + "}"

# Open the file for writing
with open(header_filename, "w") as f:
    f.write("// Auto-generated model parameters for Vitis HLS\n")
    f.write("// Exported from PTH model\n\n")

    # Iterate over model parameters
    for name, param in state_dict.items():
        # Convert PyTorch tensor to NumPy array
        param_np = param.cpu().numpy()
        param_shape = param_np.shape

        # Convert tensor name (e.g., 'network.0.weight') to a valid C++ variable name
        var_name = name.replace(".", "_")

        # Determine the type based on the variable name.
        type_name = "weights_t" if "weight" in var_name.lower() else "fixed_t"

        # Create the dimensions string (e.g. [64][10] for a 64x10 matrix)
        dims_str = "".join(f"[{dim}]" for dim in param_shape)

        # Write shape information as a comment (optional)
        f.write(f"// Shape: {param_shape}\n")

        # Write the C++ array declaration with preserved dimensions.
        initializer = array_to_brace_str(param_np)
        f.write(f"static const {type_name} {var_name}{dims_str} = {initializer};\n\n")

print(f"Weights and biases exported to {header_filename}")
