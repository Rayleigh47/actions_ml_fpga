
import torch
import torch.nn as nn
import joblib
import numpy as np

# Define the same model architecture
class ImprovedMLP(nn.Module):
    def __init__(self, input_dim=36, hidden_dims=[128, 64], dropout_rate=0.3, num_classes=7):
        super(ImprovedMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = dim
            
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def predict(input_data):
    """
    Make predictions using the trained model
    
    Args:
        input_data: Input features as numpy array or pandas DataFrame
        
    Returns:
        Predicted class label
    """
    # Load preprocessing components
    scaler = joblib.load('scaler.joblib')
    le = joblib.load('label_encoder.joblib')
    
    # Preprocess input
    if hasattr(input_data, 'values'):  # Check if it's a DataFrame
        input_data = input_data.values
    
    input_scaled = scaler.transform(input_data.reshape(1, -1))
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
    
    # Load model
    model = ImprovedMLP()
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        
    # Convert to original label
    predicted_label = le.inverse_transform(predicted.numpy())
    
    return predicted_label[0]

if __name__ == '__main__':
    # Example usage
    print("Load your input data and call the predict function")
    # Example: predicted_label = predict(your_input_data)
