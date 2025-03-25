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

# Configure device to use
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print(f"Using device: {device}")

class MNISTMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, num_classes=10):
        super(MNISTMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Identity(),
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, num_classes)
        )
    def forward(self, x):
        return self.network(x)

# Save the train and test splits as CSV files
def save_dataset_as_csv(csv_path, dataset, csv_filename):
    # Preallocate lists to hold flattened image data and labels
    images = []
    labels = []
    for img, label in dataset:
        images.append(img.numpy())
        labels.append(label)
    # Convert list of arrays to a single 2D numpy array
    images_np = np.vstack(images)
    # Create column names for each pixel feature
    columns = [f'pixel{i}' for i in range(images_np.shape[1])]
    df = pd.DataFrame(images_np, columns=columns)
    df['label'] = labels
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_filename} with shape {df.shape}")

def train(model, train_loader, criterion, optimizer, num_epochs=10):
    # L1 Regularization Parameter
    l1_lambda = 1e-5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_start_time = time.time()
        
        # Use tqdm for a progress bar over the batches
        for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Apply L1 Regularization Manually
            l1_penalty = sum(param.abs().sum() for param in model.parameters())
            loss += l1_lambda * l1_penalty

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_X.size(0)
            
            # Update training accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        accuracy = (correct / total) * 100
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%, Time: {epoch_time:.2f}s")

def test(model, test_loader):
    model.eval()
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
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)
    
    # Calculate row (actual) and column (predicted) totals
    row_totals = cm.sum(axis=1)
    col_totals = cm.sum(axis=0)
    
    # Determine class labels based on unique values in the test labels
    classes = np.unique(all_labels)
    
    # Create heatmap plot of the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    
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
    
    # Save the plot
    current_dir = os.getcwd()
    cm_path = os.path.join(current_dir, "confusion_matrix_with_totals.png")
    plt.savefig(cm_path)
    print(f"Confusion matrix saved as '{cm_path}'")

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
    # Ensure the data folder exists
    os.makedirs('data', exist_ok=True)
    
    # Define a transform to convert MNIST images to tensors and flatten them
    transform = transforms.Compose([
        transforms.ToTensor(),                     # Converts images to [0, 1] tensors
        transforms.Lambda(lambda x: x.view(-1))      # Flatten the 28x28 image into a 784-dim vector
    ])
    
    # Load MNIST training and test datasets. The data will be stored in the 'data/' folder.
    train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)
    
    # Check if CSV files exist; if not, save the datasets as CSV files
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
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Set input dimension to 28x28 and adjust number of classes to 10 (digits 0-9)
    input_dim = 28 * 28
    num_classes = 10
    
    # Instantiate the quantized model
    model = MNISTMLP(input_dim=input_dim, hidden_dim_1=256, hidden_dim_2=128, num_classes=num_classes).to(device)
    print(f"Model created with input_dim={input_dim} and num_classes={num_classes}")
    
    # Train the model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)  # L2 regularization
    train(model, train_loader, criterion, optimizer, num_epochs=10)

    # Evaluate the model on the test set
    test(model, test_loader)

    # Create 'models' directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the trained model's state_dict to 'model.pth'
    model_path = os.path.join('models', 'model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Export model parameters to a header file
    export_model_to_header(model, filename="model_params.hpp")

if __name__ == '__main__':
    main()
