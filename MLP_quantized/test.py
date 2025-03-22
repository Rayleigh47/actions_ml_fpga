import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import Brevitas quantized modules and quantization functions
from brevitas.nn import QuantIdentity, QuantLinear, QuantReLU
from brevitas.quant import (
    Int8ActPerTensorFloat, 
    Int8WeightPerTensorFixedPoint, 
    Int8Bias, 
    Int8ActPerTensorFixedPoint
)

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
                                        cache_inference_quant_bias=True,
                                        bias_quant=Int8Bias)
        self.QuantReLU1 = QuantReLU(bit_width=8, act_quant=Int8ActPerTensorFixedPoint, return_quant_tensor=True)

        # Layer 2: Linear -> ReLU
        self.QuantLinear2 = QuantLinear(hidden_dim, hidden_dim,
                                        weight_bit_width=8,
                                        weight_quant=Int8WeightPerTensorFixedPoint,
                                        return_quant_tensor=True,
                                        # bias=False,
                                        cache_inference_quant_bias=True,
                                        bias_quant=Int8Bias)
        self.QuantReLU2 = QuantReLU(bit_width=8, act_quant=Int8ActPerTensorFixedPoint, return_quant_tensor=True)

        # Output Layer: Linear (typically no activation quantization needed here)
        self.QuantLinear3 = QuantLinear(hidden_dim, num_classes,
                                        weight_bit_width=8,
                                        weight_quant=Int8WeightPerTensorFixedPoint,
                                        # bias=False,
                                        cache_inference_quant_bias=True,
                                        bias_quant=Int8Bias)
        
    def forward(self, x):
        out = self.QuantIdentity1(x)
        out = self.QuantLinear1(out)
        out = self.QuantReLU1(out)
        out = self.QuantLinear2(out)
        out = self.QuantReLU2(out)
        out = self.QuantLinear3(out)
        return out

def main():
    # Load Processed Data
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, 'data', 'processed_data_scaled.csv')
    df = pd.read_csv(data_path)
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Encode labels (make sure the mapping is consistent)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_np = X.values.astype(np.float32)
    y_np = y_encoded.astype(np.int64)
    
    # Create Train-Test Splits (using the same random_state ensures the same split)
    _, X_test, _, y_test = train_test_split(X_np, y_np, test_size=0.2, random_state=42)
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    
    batch_size = 16
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # Define model architecture and load the saved model weights
    input_dim = X_np.shape[1]
    num_classes = len(np.unique(y_np))
    model = QuantizedMLP(input_dim=input_dim, hidden_dim=64, num_classes=num_classes)
    
    model_path = os.path.join(current_dir, 'model.pth')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Model loaded from {model_path}")
    
    # Evaluate on test data
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = (all_preds == all_labels).mean() * 100
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Compute and Plot Confusion Matrix with Totals
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)
    
    # Calculate row (actual) and column (predicted) totals
    row_totals = cm.sum(axis=1)
    col_totals = cm.sum(axis=0)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    
    # Annotate each row with its total
    for i in range(len(cm)):
        ax.text(len(cm[0]) + 0.3, i + 0.5, f'{row_totals[i]}',
                va='center', ha='center', fontweight='bold')
    
    # Annotate each column with its total
    for j in range(len(cm[0])):
        ax.text(j + 0.5, -0.3, f'{col_totals[j]}',
                va='center', ha='center', fontweight='bold')
    
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    
    cm_path = os.path.join(current_dir, "confusion_matrix_with_totals.png")
    plt.savefig(cm_path)
    print(f"Confusion matrix saved as '{cm_path}'")

if __name__ == '__main__':
    main()
