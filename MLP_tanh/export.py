import torch
import numpy as np
import torch.nn as nn

# Define the MLP model (same architecture as before)
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
model = MLP(72, 64, 10)
state_dict = torch.load("models/model.pth", map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()

# Get the state dictionary (OrderedDict) of parameters
state_dict = model.state_dict()

# Create a list to hold all values from each layer in the desired order:
# layer 0 weight, then bias, then layer 1 weight, then bias, etc.
all_params = []

# Iterate over the state dictionary in order.
# (For a Sequential model, the ordering is typically as desired.)
for name, param in state_dict.items():
    # Convert the tensor to a NumPy array and flatten it to 1D
    param_flat = param.cpu().numpy().flatten()
    # Append the values to the list
    all_params.extend(param_flat.tolist())

print(f"length of CSV: {len(all_params)}")

# Write the continuous list of values to a CSV file
csv_filename = "weights_bias.csv"
with open(csv_filename, "w") as f:
    # Join all values with commas and write as one continuous line
    f.write(",".join(str(val) for val in all_params))

print(f"Weights and biases exported to {csv_filename}")
