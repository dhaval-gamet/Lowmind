# Zaroori modules import karein
import numpy as np
import os
from core.tensor import Tensor, memory_manager
from nn.modules import Module, Linear, Sequential, Dropout, Conv2d
from optim.optimizers import Adam
from optim.schedulers import ReduceLROnPlateau
from losses.losses import cross_entropy_loss
from data.data_loaders import TensorDataset, DataLoader
from utils.trainer import train_model

# Dummy data ki jagah par dog images ke liye placeholder
def load_dog_breeds_dummy(num_samples=100, img_size=64, num_classes=5):
    # Dummy image data: (batch, channels, height, width)
    X = np.random.randn(num_samples, 3, img_size, img_size).astype(np.float32)
    y = np.random.randint(0, num_classes, size=(num_samples,))
    return X, y

# CNN Model Class
class DogBreedClassifier(Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.features = Sequential(
            Conv2d(3, 16, kernel_size=3, padding=1, device=device),
            Tensor.relu,
            Conv2d(16, 32, kernel_size=3, padding=1, device=device),
            Tensor.relu
        )
        # Calculate flat features size after convolutions
        # (batch_size, 32, 64, 64) -> 32 * 64 * 64 = 131072
        # isko dynamic tareeke se calculate karne ka code bhi likh sakte hain
        self.classifier = Sequential(
            Linear(32 * 64 * 64, 128, device=device), # Example input size
            Tensor.relu,
            Linear(128, 5, device=device) # 5 dog breeds
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.shape[0], -1) # Flatten the output
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    device = 'cpu'
    
    # Dummy Dog Breed data load karein
    X_train_data, y_train_data = load_dog_breeds_dummy(num_samples=500)
    X_val_data, y_val_data = load_dog_breeds_dummy(num_samples=100)

    # Tensors banayein
    X_train = Tensor(X_train_data, device=device)
    y_train = Tensor(y_train_data, device=device)
    X_val = Tensor(X_val_data, device=device)
    y_val = Tensor(y_val_data, device=device)
    
    # Model, Optimizer aur DataLoader set karein
    model = DogBreedClassifier(device=device)
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    # Checkpoint directory banayein
    os.makedirs("checkpoints", exist_ok=True)
    
    # Model ko train karein
    print("Dog Breed Classifier Training Shuru ho raha hai...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=cross_entropy_loss,
        epochs=5,
        device=device,
        scheduler=scheduler,
        checkpoint_path='checkpoints/dog_classifier',
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
