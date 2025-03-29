import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm

# Import Brevitas quantized modules and quantization functions
from brevitas.nn import QuantIdentity, QuantLinear, QuantTanh
from brevitas.quant import (
    Int8ActPerTensorFloat,
    Int8WeightPerTensorFixedPoint, 
    Int32Bias,
    Int8ActPerTensorFixedPoint
)
from brevitas.export import export_onnx_qcdq

# Configure device to use
device = torch.device('cpu')
print(f"Using device: {device}")

# Define the quantized MLP model using Brevitas modules
class QuantizedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, num_classes=10):
        super(QuantizedMLP, self).__init__()
        self.QuantIdentity1 = QuantIdentity(bit_width=8, return_quant_tensor=True, act_quant=Int8ActPerTensorFloat, threshold=1.0)
        self.QuantLinear1 = QuantLinear(input_dim, hidden_dim_1,
                                        weight_bit_width=8,
                                        weight_quant=Int8WeightPerTensorFixedPoint,
                                        return_quant_tensor=True,
                                        bias_quant=Int32Bias)
        self.QuantTanh1 = QuantTanh(bit_width=8, act_quant=Int8ActPerTensorFixedPoint, return_quant_tensor=True)
        self.QuantLinear2 = QuantLinear(hidden_dim_1, hidden_dim_2,
                                        weight_bit_width=8,
                                        weight_quant=Int8WeightPerTensorFixedPoint,
                                        return_quant_tensor=True,
                                        bias_quant=Int32Bias)
        self.QuantTanh2 = QuantTanh(bit_width=8, act_quant=Int8ActPerTensorFixedPoint, return_quant_tensor=True)
        self.QuantLinear3 = QuantLinear(hidden_dim_2, num_classes,
                                        weight_bit_width=8,
                                        weight_quant=Int8WeightPerTensorFixedPoint,
                                        bias_quant=Int32Bias)
        
    def forward(self, x):
        out = self.QuantIdentity1(x)
        out = self.QuantLinear1(out)
        out = self.QuantTanh1(out)
        out = self.QuantLinear2(out)
        out = self.QuantTanh2(out)
        out = self.QuantLinear3(out)
        return out

# Save the train and test splits as CSV files
def save_dataset_as_csv(csv_path, dataset, csv_filename):
    images = []
    labels = []
    for img, label in dataset:
        images.append(img.numpy())
        labels.append(label)
    images_np = np.vstack(images)
    columns = [f'pixel{i}' for i in range(images_np.shape[1])]
    df = pd.DataFrame(images_np, columns=columns)
    df['label'] = labels
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_filename} with shape {df.shape}")

# EarlyStopping class that monitors validation loss and saves the best model
class EarlyStopping:
    def __init__(self, patience=3, delta=0, path='models/model.pth', path2='models/model.onnx'):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            delta (float): Minimum change in monitored value to qualify as improvement.
            path (str): Path to save the best model.
        """
        self.patience = patience
        self.delta = delta
        self.path = path
        self.path2 = path2
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        # Check if the validation loss improved
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            # TORCH
            torch.save(model.state_dict(), self.path)
            print(f"Validation loss improved, saving model to {self.path}")
            # ONNX
            dummy_input = torch.randn(1, 784).to(device)
            export_onnx_qcdq(model, dummy_input, self.path2)
            print(f"Best model exported to {self.path2}")
        else:
            self.counter += 1
            print(f"No improvement in validation loss for {self.counter} epoch(s)")
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stopping triggered!")

# Combined training and testing routine with early stopping
def train_and_test(model, train_loader, test_loader, criterion, optimizer, num_epochs=50, l1_lambda=1e-5):
    early_stopping = EarlyStopping(patience=5, delta=0.001, path='models/model.pth')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_start_time = time.time()
        
        # Training loop
        for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            # Apply L1 Regularization
            l1_penalty = sum(param.abs().sum() for param in model.parameters())
            loss += l1_lambda * l1_penalty
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        accuracy = (correct / total) * 100
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f}, Train Accuracy: {accuracy:.2f}%, Time: {epoch_time:.2f}s")
        
        # Testing phase (used as validation for early stopping)
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs, 1)
                test_correct += (predicted == batch_y).sum().item()
                test_total += batch_y.size(0)
        val_loss = test_loss / len(test_loader.dataset)
        val_accuracy = (test_correct / test_total) * 100
        print(f"Epoch {epoch+1} - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        
        # Check early stopping condition
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            break

    # Load the best model saved during early stopping
    model.load_state_dict(torch.load('models/best_model.pth'))
    print("Best model loaded after training.")
    return model

def main():
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Define a transform to convert MNIST images to tensors and flatten them
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    
    # Load MNIST datasets
    train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)
    
    # Save as CSV if files don't exist already
    train_csv_path = os.path.join('data', 'train.csv')
    test_csv_path = os.path.join('data', 'test.csv')
    if not os.path.exists(train_csv_path):
        save_dataset_as_csv(train_csv_path, train_dataset, 'train.csv')
    else:
        print(f"{train_csv_path} exists, skipping saving train dataset.")
    if not os.path.exists(test_csv_path):
        save_dataset_as_csv(test_csv_path, test_dataset, 'test.csv')
    else:
        print(f"{test_csv_path} exists, skipping saving test dataset.")
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    input_dim = 28 * 28
    num_classes = 10
    model = QuantizedMLP(input_dim=input_dim, hidden_dim_1=512, hidden_dim_2=256, num_classes=num_classes).to(device)
    print(f"Model created with input_dim={input_dim} and num_classes={num_classes}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    
    # Train with early stopping and test after each epoch.
    best_model = train_and_test(model, train_loader, test_loader, criterion, optimizer, num_epochs=50, l1_lambda=1e-5)

if __name__ == '__main__':
    main()
