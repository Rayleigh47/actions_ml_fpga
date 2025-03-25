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

# Define the same MLP model (this must match the training architecture)
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

def export_model_to_header(model, filename="model_params.hpp"):
    # Get the state dict from the model
    state_dict = model.state_dict()
    
    with open(filename, "w") as f:
        f.write("// Auto-generated model parameters\n\n")
        f.write("// Parameter shapes (for reference):\n")
        layer_idx = 0
        # Print the shapes of the weight matrices
        for layer in model.network:
            if isinstance(layer, nn.Linear):
                # Retrieve weight and bias tensors
                weight_tensor = layer.weight.detach().cpu().numpy()
                bias_tensor = layer.bias.detach().cpu().numpy()
                
                # Get the shape of the weight matrix
                rows, cols = weight_tensor.shape
                f.write(f"// layer_{layer_idx}_weight[{rows}][{cols}]\n")

                # Get the shape of the bias vector
                bias_len = bias_tensor.shape[0]
                f.write(f"// layer_{layer_idx}_bias[{bias_len}]\n\n")

                layer_idx += 1

        layer_idx = 0
        # Iterate over each layer in the sequential container
        for layer in model.network:
            if isinstance(layer, nn.Linear):
                # Retrieve weight and bias tensors
                weight_tensor = layer.weight.detach().cpu().numpy()
                bias_tensor = layer.bias.detach().cpu().numpy()
                
                # Get the shape of the weight matrix: [rows, cols]
                rows, cols = weight_tensor.shape
                
                # Write the weight matrix as a multi-dimensional array
                f.write(f"static const float layer_{layer_idx}_weight[{rows}][{cols}] = {{\n")
                for i, row in enumerate(weight_tensor):
                    row_vals = ", ".join([f"{v:.8f}" for v in row])
                    # Separate rows with commas
                    if i < rows - 1:
                        f.write(f"    {{ {row_vals} }},\n")
                    else:
                        f.write(f"    {{ {row_vals} }}\n")
                f.write("};\n\n")
                
                # Write the bias vector as a 1D array
                bias_len = bias_tensor.shape[0]
                bias_vals = ", ".join([f"{v:.8f}" for v in bias_tensor])
                f.write(f"static const float layer_{layer_idx}_bias[{bias_len}] = {{ {bias_vals} }};\n\n")
                
                layer_idx += 1

    print(f"Model parameters exported to {filename}")

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
    model = MLP(input_dim=input_dim, hidden_dim=64, num_classes=num_classes)
    
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

    # Export model parameters to a C++ header file
    export_model_to_header(model, filename="model_params.hpp")

if __name__ == '__main__':
    main()
