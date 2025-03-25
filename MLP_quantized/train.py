import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Import Brevitas quantized modules and quantization functions
from brevitas.nn import QuantIdentity, QuantLinear, QuantReLU
from brevitas.quant import (
    Int8ActPerTensorFloat, 
    Int8WeightPerTensorFixedPoint, 
    Int32Bias,
    Int8ActPerTensorFixedPoint
)
from brevitas.export import export_onnx_qcdq

# configure device to use
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print(f"Using device: {device}")

# Define the quantized MLP model using Brevitas modules
class QuantizedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=2):
        super(QuantizedMLP, self).__init__()
        # Input quantization: set threshold=1.0 since inputs are in [0, 1]
        self.QuantIdentity1 = QuantIdentity(bit_width=8, return_quant_tensor=True, act_quant=Int8ActPerTensorFloat, threshold=1.0)

        # Layer 1: Linear -> ReLU with threshold for quantization
        self.QuantLinear1 = QuantLinear(input_dim, hidden_dim,
                                        weight_bit_width=8,
                                        weight_quant=Int8WeightPerTensorFixedPoint,
                                        return_quant_tensor=True,
                                        # bias=False,
                                        bias_quant=Int32Bias)
        self.QuantReLU1 = QuantReLU(bit_width=8, act_quant=Int8ActPerTensorFixedPoint, return_quant_tensor=True)

        # Layer 2: Linear -> ReLU
        self.QuantLinear2 = QuantLinear(hidden_dim, hidden_dim,
                                        weight_bit_width=8,
                                        weight_quant=Int8WeightPerTensorFixedPoint,
                                        return_quant_tensor=True,
                                        # bias=False,
                                        bias_quant=Int32Bias)
        self.QuantReLU2 = QuantReLU(bit_width=8, act_quant=Int8ActPerTensorFixedPoint, return_quant_tensor=True)

        # Output Layer: Linear (typically no activation quantization needed here)
        self.QuantLinear3 = QuantLinear(hidden_dim, num_classes,
                                        weight_bit_width=8,
                                        weight_quant=Int8WeightPerTensorFixedPoint,
                                        # bias=False,
                                        bias_quant=Int32Bias)
        
    def forward(self, x):
        out = self.QuantIdentity1(x)
        out = self.QuantLinear1(out)
        out = self.QuantReLU1(out)
        out = self.QuantLinear2(out)
        out = self.QuantReLU2(out)
        out = self.QuantLinear3(out)
        return out

def main():
    # Load Processed Data (assumed to be already scaled to [0, 1])
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, 'data', 'processed_data_scaled.csv')
    df = pd.read_csv(data_path)
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_np = X.values.astype(np.float32)
    y_np = y_encoded.astype(np.int64)
    
    # Create Train-Test Splits
    X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.2, random_state=42)
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    
    batch_size = 16
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    # Instantiate the quantized model
    input_dim = X_np.shape[1]
    print(f"Input dimension: {input_dim}")
    num_classes = len(np.unique(y_np))
    model = QuantizedMLP(input_dim=input_dim, hidden_dim=128, num_classes=num_classes)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)  # L2 regularization

    # L1 Regularization Parameter
    l1_lambda = 1e-5

    # Training Loop
    num_epochs = 300
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Apply L1 Regularization Manually
            l1_penalty = sum(param.abs().sum() for param in model.parameters())
            loss += l1_lambda * l1_penalty

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_X.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    
    # data directory
    data_dir = os.path.join(current_dir, 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Save the model pth
    model_path = os.path.join(data_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Export as onnx
    dummy_input = torch.randn(1, input_dim)
    onnx_path = os.path.join(data_dir, 'model.onnx')
    export_onnx_qcdq(model, dummy_input, onnx_path)
    print(f"Model exported to {onnx_path}")

if __name__ == '__main__':
    main()
    def forward(self, x):
        out = self.QuantIdentity1(x)
        out = self.QuantLinear1(out)
        out = self.QuantReLU1(out)
        out = self.QuantLinear2(out)
        out = self.QuantReLU2(out)
        out = self.QuantLinear3(out)
        return out