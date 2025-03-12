import torch
import torch.nn as nn
import numpy as np
import os
import sys

# Define the minimal MLP model (should match the training architecture)
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=2):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

def load_model(model_path, input_dim, num_classes=2):
    # Instantiate the model and load saved weights
    model = MLP(input_dim=input_dim, hidden_dim=64, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict(model, input_data):
    """
    Run inference on input_data (a list or numpy array of features)
    and return the predicted class.
    """
    if not isinstance(input_data, np.ndarray):
        input_data = np.array(input_data, dtype=np.float32)
    # Ensure the input has a batch dimension
    if input_data.ndim == 1:
        input_data = input_data.reshape(1, -1)
    with torch.no_grad():
        tensor_data = torch.from_numpy(input_data)
        outputs = model(tensor_data)
        _, predicted = torch.max(outputs, 1)
    return predicted.numpy()

def main():
    # Usage: python inference.py feature1 feature2 ... featureN
    if len(sys.argv) < 2:
        print("Usage: python inference.py feature1 feature2 ...")
        sys.exit(1)
    
    # Check if the argument is comma separated; if so, split it.
    if ',' in sys.argv[1]:
        input_features = [float(x.strip()) for x in sys.argv[1].split(',')]
    else:
        # Otherwise, treat each argument as a separate feature
        input_features = [float(arg) for arg in sys.argv[1:]]
    
    input_dim = len(input_features)
    print(f"Input dimensions: {input_dim}")

    # Load the model (assumes model.pth is in the same directory)
    model_path = os.path.join(os.path.dirname(__file__), 'model.pth')
    model = load_model(model_path, input_dim=input_dim, num_classes=7) # hard code num_classes
    
    # Run inference and print the prediction
    prediction = predict(model, input_features)
    print("Predicted class:", prediction[0])

if __name__ == '__main__':
    main()
