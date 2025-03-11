import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # -------------------------------
    # Load Processed Data
    # -------------------------------]
    current_dir = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(current_dir, 'data', 'processed_data.csv'))
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_np = X.values.astype(np.float32)
    y_np = y_encoded.astype(np.int64)
    
    # -------------------------------
    # Create Train-Test Splits
    # -------------------------------
    X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.2, random_state=42)
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    
    batch_size = 16
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # -------------------------------
    # Define the MLP Model
    # -------------------------------
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
    
    input_dim = X_np.shape[1]
    num_classes = len(np.unique(y_np))
    model = MLP(input_dim=input_dim, hidden_dim=64, num_classes=num_classes)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # -------------------------------
    # Training Loop
    # -------------------------------
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_X.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    
    # -------------------------------
    # Evaluation on Test Data
    # -------------------------------
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
    
    # -------------------------------
    # Compute and Plot Confusion Matrix with Totals
    # -------------------------------
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)
    
    # Calculate row totals (actual counts) and column totals (predicted counts)
    row_totals = cm.sum(axis=1)
    col_totals = cm.sum(axis=0)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    
    # Annotate each row with its total (actual count)
    for i in range(len(cm)):
        ax.text(len(cm[0]) + 0.3, i + 0.5, f'Total: {row_totals[i]}',
                va='center', ha='center', fontweight='bold')
    
    # Annotate each column with its total (predicted count)
    for j in range(len(cm[0])):
        ax.text(j + 0.5, -0.3, f'Total: {col_totals[j]}',
                va='center', ha='center', fontweight='bold')
    
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    
    # Save the plot to a file
    plt.savefig("confusion_matrix_with_totals.png")
    print("Confusion matrix saved as 'confusion_matrix_with_totals.png'")

if __name__ == '__main__':
    main()
