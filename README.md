<b>Custom Deep Learning Framework</b>
A toy deep learning framework built from scratch using Python and NumPy, designed to demystify the core concepts behind modern deep learning libraries like PyTorch and TensorFlow.
🚀 Introduction
Ever wondered what goes on under the hood of a deep learning framework? This project is an educational deep dive into the fundamentals. It's not just another model; it's a complete, albeit simplified, system that handles everything from data representation to automatic differentiation and model optimization.
Our goal is to provide a clean, readable, and functional implementation of key deep learning components. Whether you're a student, a curious developer, or an aspiring researcher, exploring this codebase will help solidify your understanding of how neural networks truly learn.
✨ Features
 * Custom Tensor Class: A robust Tensor object with built-in automatic differentiation (autograd) for effortless gradient computation.
 * Modular Layers: A collection of essential neural network building blocks, including Linear, Conv2d, BatchNorm2d, and Dropout layers, all designed to be easily combined.
 * Plug-and-Play Optimizers: State-of-the-art optimizers like Adam and SGD to efficiently update model parameters and minimize loss.
 * Rich Loss Functions: Includes common loss functions like Cross-Entropy Loss and Mean Squared Error (MSE).
 * Efficient Utilities: Simple yet effective tools for data management (DataLoader) and a comprehensive training loop (train_model).
⚙️ Installation
To get started, clone the repository and install the required dependencies.
git clone https://github.com/yourusername/custom_deep_learning_framework.git
cd custom_deep_learning_framework
pip install -r requirements.txt

📖 Usage
1. Building a Neural Network
You can create a custom model by combining the modular layers provided in the framework. Here's how to build a simple classifier:
from nn.modules import Module, Linear, Sequential, Dropout
from core.tensor import Tensor

class SimpleNN(Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.layers = Sequential(
            Linear(784, 256, device=device),
            Tensor.relu,
            Dropout(0.2, device=device),
            Linear(256, 128, device=device),
            Tensor.relu,
            Dropout(0.2, device=device),
            Linear(128, 10, device=device)
        )
    
    def forward(self, x):
        return self.layers(x)

# Instantiate the model
model = SimpleNN()

2. Training Your Model
The framework includes a utility function to handle the entire training and validation process.
import numpy as np
from optim.optimizers import Adam
from losses.losses import cross_entropy_loss
from data.data_loaders import TensorDataset, DataLoader
from utils.trainer import train_model

# Generate dummy data
X_train = Tensor(np.random.randn(1000, 784))
y_train = Tensor(np.random.randint(0, 10, size=(1000,)))
X_val = Tensor(np.random.randn(200, 784))
y_val = Tensor(np.random.randint(0, 10, size=(200,)))

# Create data loaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Set up optimizer and loss function
optimizer = Adam(model.parameters(), lr=0.001)
criterion = cross_entropy_loss

# Run the training loop
train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    epochs=10
)

📂 Project Structure
The codebase is organized into a clean, modular structure for easy navigation and extension.
.
├── core/                  # Foundational components
│   ├── tensor.py          # The core Tensor class with autograd
│   └── __init__.py
├── nn/                    # Neural Network modules
│   ├── modules.py         # Base Module and various layers (Linear, Conv2d)
│   └── __init__.py
├── optim/                 # Optimization algorithms
│   ├── optimizers.py      # SGD, Adam, and base Optimizer class
│   └── schedulers.py      # Learning rate schedulers
│   └── __init__.py
├── losses/                # Loss functions
│   ├── losses.py          # Cross-Entropy, MSE, etc.
│   └── __init__.py
├── data/                  # Data handling
│   ├── data_loaders.py    # Dataset and DataLoader classes
│   └── __init__.py
├── utils/                 # Training utilities
│   ├── trainer.py         # The main training function
│   └── __init__.py
├── main.py                # Example usage script
├── requirements.txt       # Project dependencies
└── README.md              # You are here!

🤝 Contributing
We welcome contributions! Whether it's adding a new layer, improving an optimizer, or fixing a bug, your help is valuable. Feel free to fork the repository, open an issue, or submit a pull request.
📄 License
This project is licensed under the MIT License.
