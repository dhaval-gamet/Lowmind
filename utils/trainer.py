import time
from nn.modules import Module
from core.tensor import Tensor, memory_manager

def train_model(model: Module, train_loader, val_loader, optimizer, criterion, epochs: int, 
                device='cpu', scheduler=None, checkpoint_path=None, early_stopping_patience=None):
    model.train()
    model.to(device)
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        start_time = time.time()
        
        for i, batch in enumerate(train_loader):
            inputs, targets = batch if len(batch) == 2 else (batch[0], None)
            inputs = inputs.to(device)
            targets = targets.to(device) if targets is not None else None
            
            outputs = model(inputs)
            loss = criterion(outputs, targets) if targets is not None else criterion(outputs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.data
            
            if (i + 1) % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {i+1}, Loss: {loss.data:.4f}')
        
        model.eval()
        total_val_loss = 0
        
        # Deactivating gradient computation during validation
        original_requires_grad = [p.requires_grad for p in model.parameters()]
        for p in model.parameters():
            p.requires_grad = False
            
        for batch in val_loader:
            inputs, targets = batch if len(batch) == 2 else (batch[0], None)
            inputs = inputs.to(device)
            targets = targets.to(device) if targets is not None else None
            
            outputs = model(inputs)
            loss = criterion(outputs, targets) if targets is not None else criterion(outputs)
            total_val_loss += loss.data
        
        # Reactivating gradient computation
        for i, p in enumerate(model.parameters()):
            p.requires_grad = original_requires_grad[i]
            
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        
        if scheduler:
            if hasattr(scheduler, 'step') and 'ReduceLROnPlateau' in str(scheduler.__class__):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
        
        if checkpoint_path and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save(f"{checkpoint_path}_best.pth")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if early_stopping_patience and patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Time: {epoch_time:.2f}s, LR: {optimizer.lr:.6f}')
    
    return history

