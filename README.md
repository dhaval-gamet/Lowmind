
📌 README.md

# 🧠 Custom Deep Learning Framework

A **toy deep learning framework** built from scratch using **Python** and **NumPy**, designed to demystify the core concepts behind modern libraries like **PyTorch** and **TensorFlow**.

---

## 🚀 Introduction

Ever wondered what goes on *under the hood* of a deep learning framework?  
This project is an **educational deep dive** into the fundamentals.  

It’s not just another model; it’s a complete (yet simplified) system that handles everything from:

- Data representation  
- Automatic differentiation  
- Model optimization  

Our goal is to provide a **clean, readable, and functional** implementation of key deep learning components.  

Whether you're a **student**, a **curious developer**, or an **aspiring researcher**, exploring this codebase will help solidify your understanding of how neural networks truly learn.  

---

## ✨ Features

- **Custom Tensor Class** → With built-in automatic differentiation (autograd).  
- **Modular Layers** → Linear, Conv2d, BatchNorm2d, Dropout, and more.  
- **Plug-and-Play Optimizers** → State-of-the-art optimizers like **Adam** and **SGD**.  
- **Rich Loss Functions** → CrossEntropyLoss, Mean Squared Error (MSE).  
- **Efficient Utilities** → DataLoader and a full training loop (`train_model`).  

---

## ⚙️ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/dhaval-gamet/lowmind.git
cd custom_deep_learning_framework
pip install -r requirements.txt```


---

📖 Usage

1️⃣ Building a Neural Network

```from nn.modules import Module, Linear, Sequential, Dropout
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
```

---

2️⃣ Training Your Model

import numpy as np
from optim.optimizers import Adam
from losses.losses import cross_entropy_loss
from data.data_loaders import TensorDataset, DataLoader
from utils.trainer import train_model
from core.tensor import Tensor

# Dummy data
X_train = Tensor(np.random.randn(1000, 784))
y_train = Tensor(np.random.randint(0, 10, size=(1000,)))
X_val = Tensor(np.random.randn(200, 784))
y_val = Tensor(np.random.randint(0, 10, size=(200,)))

# Data loaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Optimizer + Loss
optimizer = Adam(model.parameters(), lr=0.001)
criterion = cross_entropy_loss

# Training loop
train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    epochs=10
)

---

```📂 Project Structure

.
├── core/                  # Foundational components
│   ├── tensor.py          # Tensor class with autograd
│   └── __init__.py
├── example/
    ├── dog_breed_identifier.py
    └── simple_image_classifier.py 
├── nn/                    # Neural Network modules
│   ├── modules.py         # Base Module, Linear, Conv2d, etc.
│   └── __init__.py
├── optim/                 # Optimizers + Schedulers
│   ├── optimizers.py
│   ├── schedulers.py
│   └── __init__.py
├── losses/                # Loss functions
│   ├── losses.py
│   └── __init__.py
├── data/                  # Data utilities
│   ├── data_loaders.py
│   └── __init__.py
├── utils/                 # Training utilities
│   ├── trainer.py
│   └── __init__.py
├── main.py                # Example script
├── requirements.txt       # Dependencies
└── README.md              # You are here!

```
---

```🤝 Contributing

We welcome contributions! 🎉

Add a new layer

Improve an optimizer

Fix a bug


Feel free to fork the repo, open an issue, or submit a pull request.


---

📄 License

This project is licensed under the MIT License.
See LICENSE for more details.


