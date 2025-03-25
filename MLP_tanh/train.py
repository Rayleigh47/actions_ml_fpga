import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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
    
    # Instantiate the model
    input_dim = X_np.shape[1]
    print(f"Input dimension: {input_dim}")
    num_classes = len(np.unique(y_np))
    model = MLP(input_dim=input_dim, hidden_dim=64, num_classes=num_classes)
    
    criterion = nn.CrossEntropyLoss()

    # Use L2 Regularization (Weight Decay)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # L2 regularization

    # L1 Regularization Parameter
    l1_lambda = 1e-5

    # Training Loop
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Apply L1 Regularization Manually
            l1_penalty = sum(param.abs().sum() for param in model.parameters())
            loss += l1_lambda * l1_penalty  # Adding L1 penalty to loss

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_X.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    
    # Save the trained model's state_dict to 'model.pth'
    model_dir = os.path.join(current_dir, 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    main()
