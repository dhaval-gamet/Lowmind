import numpy as np

def mse_loss(output, target, reduction='mean'):
    diff = output - target
    loss = (diff * diff)
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

def cross_entropy_loss(output, target, reduction='mean', label_smoothing=0.0):
    max_vals = output.max(axis=1, keepdims=True)
    exp_output = (output - max_vals).exp()
    softmax_output = exp_output / exp_output.sum(axis=1, keepdims=True)
    
    if label_smoothing > 0:
        num_classes = output.shape[1]
        target_one_hot = np.eye(num_classes)[target.data.astype(int)]
        target_one_hot = (1 - label_smoothing) * target_one_hot + label_smoothing / num_classes
        log_probs = (softmax_output.log() * target_one_hot).sum(axis=1)
    else:
        batch_size = output.shape[0]
        log_probs = -softmax_output.log()[range(batch_size), target.data.astype(int)]
    
    if reduction == 'mean':
        return log_probs.mean()
    elif reduction == 'sum':
        return log_probs.sum()
    else:
        return log_probs

def nll_loss(output, target, reduction='mean'):
    batch_size = output.shape[0]
    loss = -output[range(batch_size), target.data.astype(int)]
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

def binary_cross_entropy(output, target, reduction='mean'):
    output_clamped = np.clip(output.data, 1e-7, 1 - 1e-7)
    loss = -(target.data * np.log(output_clamped) + (1 - target.data) * np.log(1 - output_clamped))
    loss = Tensor(loss, requires_grad=output.requires_grad, device=output.device)
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

