import numpy as np
import os
from core.tensor import Tensor, memory_manager
from nn.modules import Module, Linear, Sequential, Dropout
from optim.optimizers import Adam
from optim.schedulers import ReduceLROnPlateau
from losses.losses import cross_entropy_loss
from data.data_loaders import TensorDataset, DataLoader
from utils.trainer import train_model

# ----------------------------
# Example Usage
# ----------------------------
if __name__ == "__main__":
    # Create a simple neural network
    class SimpleNN(Module):
        def __init__(self, device='cpu'):
            super().__init__()
            self.layers = Sequential(
                Linear(784, 256, bias=True, device=device),
                Tensor.relu,
                Dropout(0.2, device=device),
                Linear(256, 128, bias=True, device=device),
                Tensor.relu,
                Dropout(0.2, device=device),
                Linear(128, 10, bias=True, device=device)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    # Create model, optimizer and data
    device = 'cpu'
    model = SimpleNN(device=device)
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    # Generate dummy data
    X_train = Tensor(np.random.randn(1000, 784), device=device)
    y_train = Tensor(np.random.randint(0, 10, size=(1000,)), device=device)
    X_val = Tensor(np.random.randn(200, 784), device=device)
    y_val = Tensor(np.random.randint(0, 10, size=(200,)), device=device)
    
    # Create datasets and data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs("checkpoints", exist_ok=True)
    
    # Training loop
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=cross_entropy_loss,
        epochs=10,
        device=device,
        scheduler=scheduler,
        checkpoint_path='checkpoints/model',
        early_stopping_patience=10
    )
    
    print("Training completed!")
    print(f"Final Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"Final Val Loss: {history['val_loss'][-1]:.4f}")
    print(f"Memory Usage: {memory_manager.get_memory_info()}")

