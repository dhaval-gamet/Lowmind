# Zaroori modules import karein
import numpy as np
import os
from core.tensor import Tensor, memory_manager
from nn.modules import Module, Linear, Sequential, Dropout
from optim.optimizers import Adam
from optim.schedulers import ReduceLROnPlateau
from losses.losses import cross_entropy_loss
from data.data_loaders import TensorDataset, DataLoader
from utils.trainer import train_model

# Dummy data ki jagah par MNIST data load karne ke liye placeholder
def load_mnist_dummy(num_samples=1000, img_size=28, num_classes=10):
    # Asal mein, aap yahan asli MNIST data load karenge
    # Lekin demonstration ke liye, hum dummy data use kar rahe hain
    X = np.random.randn(num_samples, img_size * img_size).astype(np.float32)
    y = np.random.randint(0, num_classes, size=(num_samples,))
    return X, y

# Model Class (wohi code jo aapne diya tha)
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

if __name__ == "__main__":
    device = 'cpu'
    
    # Dummy MNIST data load karein
    X_train_data, y_train_data = load_mnist_dummy(num_samples=1000)
    X_val_data, y_val_data = load_mnist_dummy(num_samples=200)

    # Tensors banayein
    X_train = Tensor(X_train_data, device=device)
    y_train = Tensor(y_train_data, device=device)
    X_val = Tensor(X_val_data, device=device)
    y_val = Tensor(y_val_data, device=device)
    
    # Model, Optimizer aur DataLoader set karein
    model = SimpleNN(device=device)
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Checkpoint directory banayein
    os.makedirs("checkpoints", exist_ok=True)
    
    # Model ko train karein
    print("MNIST Image Classifier Training Shuru ho raha hai...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=cross_entropy_loss,
        epochs=10,
        device=device,
        scheduler=scheduler,
        checkpoint_path='checkpoints/mnist_model',
        early_stopping_patience=10
    )
    
    print("Training poora hua!")
    print(f"Final Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"Final Val Loss: {history['val_loss'][-1]:.4f}")
    
    # Prediction ka example
    model.eval()
    test_input = Tensor(X_val_data[0:1], device=device)
    predicted_output = model(test_input)
    predicted_class = np.argmax(predicted_output.data)
    print(f"\nPehli validation image ke liye, predicted class: {predicted_class}")
    print(f"Actual class: {y_val_data[0]}")

