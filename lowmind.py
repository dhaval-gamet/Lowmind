import numpy as np
import time
from collections import OrderedDict, defaultdict
import pickle
import gzip
import os

# ----------------------------
# Memory Management Utilities
# ----------------------------
class MemoryManager:
    """Simple memory manager to track and limit memory usage"""
    def __init__(self, max_memory_mb=512):
        self.max_memory = max_memory_mb * 1024 * 1024  # Convert to bytes
        self.allocated_memory = 0
        self.tensors = {}
    
    def allocate(self, tensor, name=None):
        size = tensor.data.nbytes
        if hasattr(tensor, 'grad') and tensor.grad is not None:
            size += tensor.grad.nbytes
        
        if self.allocated_memory + size > self.max_memory:
            self.free_unused()
            
            if self.allocated_memory + size > self.max_memory:
                raise MemoryError(f"Out of memory: {self.allocated_memory/(1024*1024):.2f}MB used, "
                                 f"{size/(1024*1024):.2f}MB requested, {self.max_memory/(1024*1024):.2f}MB max")
        
        self.allocated_memory += size
        if name:
            self.tensors[name] = (tensor, size)
        
        return tensor
    
    def free(self, name):
        if name in self.tensors:
            tensor, size = self.tensors[name]
            self.allocated_memory -= size
            del self.tensors[name]
    
    def free_unused(self):
        # Free tensors that are no longer needed
        to_remove = []
        for name, (tensor, size) in self.tensors.items():
            # Check if tensor is a leaf node and has no dependencies
            if not hasattr(tensor, 'requires_grad') or not tensor.requires_grad:  # A simple heuristic
                to_remove.append(name)
        
        for name in to_remove:
            self.free(name)
    
    def get_memory_info(self):
        return {
            'allocated_mb': self.allocated_memory / (1024 * 1024),
            'max_mb': self.max_memory / (1024 * 1024),
            'usage_percent': (self.allocated_memory / self.max_memory) * 100
        }

# Global memory manager
memory_manager = MemoryManager(max_memory_mb=512)

# ----------------------------
# Enhanced Tensor Class
# ----------------------------
class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op='', device='cpu', name=None):
        self.data = np.array(data, dtype=np.float32)  # Use float32 for memory efficiency
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.device = device
        self.name = name
        self._version = 0  # For tracking in-place operations
        
        # Register with memory manager
        memory_manager.allocate(self, name)
        
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        out = Tensor(self.data + other.data, 
                    requires_grad=self.requires_grad or other.requires_grad,
                    _children=(self, other), _op='+', device=self.device)
        
        def _backward():
            if self.requires_grad:
                # Handle broadcasting in backward pass
                grad = out.grad
                for _ in range(grad.ndim - self.data.ndim):
                    grad = grad.sum(axis=0)
                for i, dim in enumerate(self.data.shape):
                    if dim == 1:
                        grad = grad.sum(axis=i, keepdims=True)
                self.grad += grad
                
            if other.requires_grad:
                grad = out.grad
                for _ in range(grad.ndim - other.data.ndim):
                    grad = grad.sum(axis=0)
                for i, dim in enumerate(other.data.shape):
                    if dim == 1:
                        grad = grad.sum(axis=i, keepdims=True)
                other.grad += grad
                
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        out = Tensor(self.data * other.data, 
                    requires_grad=self.requires_grad or other.requires_grad,
                    _children=(self, other), _op='*', device=self.device)
        
        def _backward():
            if self.requires_grad:
                grad = out.grad * other.data
                for _ in range(grad.ndim - self.data.ndim):
                    grad = grad.sum(axis=0)
                for i, dim in enumerate(self.data.shape):
                    if dim == 1:
                        grad = grad.sum(axis=i, keepdims=True)
                self.grad += grad
                
            if other.requires_grad:
                grad = out.grad * self.data
                for _ in range(grad.ndim - other.data.ndim):
                    grad = grad.sum(axis=0)
                for i, dim in enumerate(other.data.shape):
                    if dim == 1:
                        grad = grad.sum(axis=i, keepdims=True)
                other.grad += grad
                
        out._backward = _backward
        return out
    
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        out = Tensor(self.data @ other.data,
                    requires_grad=self.requires_grad or other.requires_grad,
                    _children=(self, other), _op='@', device=self.device)
        
        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out
    
    def __pow__(self, power):
        assert isinstance(power, (int, float)), "Only supporting int/float powers"
        out = Tensor(self.data ** power, 
                    requires_grad=self.requires_grad,
                    _children=(self,), _op=f'**{power}', device=self.device)
        
        def _backward():
            if self.requires_grad:
                self.grad += (power * self.data**(power-1)) * out.grad
        out._backward = _backward
        return out
    
    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), 
                    requires_grad=self.requires_grad,
                    _children=(self,), _op='sum', device=self.device)
        
        def _backward():
            if self.requires_grad:
                # Handle broadcasting for sum backward
                if axis is not None:
                    # Expand dimensions to match original shape
                    if not keepdims:
                        if isinstance(axis, int):
                            expanded_grad = np.expand_dims(out.grad, axis=axis)
                        else:
                            expanded_grad = out.grad
                            for ax in sorted(axis):
                                expanded_grad = np.expand_dims(expanded_grad, axis=ax)
                    else:
                        expanded_grad = out.grad
                    
                    # Tile to match original shape
                    if isinstance(axis, int):
                        self.grad += np.full(self.data.shape, expanded_grad)
                    else:
                        tile_shape = [1] * self.data.ndim
                        for ax in axis:
                            tile_shape[ax] = self.data.shape[ax]
                        self.grad += np.tile(expanded_grad, tile_shape)

                else:
                    self.grad += np.ones_like(self.data) * out.grad
        out._backward = _backward
        return out
    
    def mean(self, axis=None, keepdims=False):
        out = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims), 
                    requires_grad=self.requires_grad,
                    _children=(self,), _op='mean', device=self.device)
        
        def _backward():
            if self.requires_grad:
                # Handle broadcasting for mean backward
                if axis is not None:
                    # Calculate the number of elements averaged
                    if isinstance(axis, tuple):
                        n = np.prod([self.data.shape[ax] for ax in axis])
                    else:
                        n = self.data.shape[axis]
                    
                    # Expand dimensions to match original shape
                    if not keepdims:
                        if isinstance(axis, int):
                            expanded_grad = np.expand_dims(out.grad, axis=axis)
                        else:
                            expanded_grad = out.grad
                            for ax in sorted(axis):
                                expanded_grad = np.expand_dims(expanded_grad, axis=ax)
                    else:
                        expanded_grad = out.grad
                    
                    # Tile to match original shape and divide by n
                    if isinstance(axis, int):
                        self.grad += np.full(self.data.shape, expanded_grad / n)
                    else:
                        tile_shape = [1] * self.data.ndim
                        for ax in axis:
                            tile_shape[ax] = self.data.shape[ax]
                        self.grad += np.tile(expanded_grad / n, tile_shape)
                else:
                    self.grad += (np.ones_like(self.data) / self.data.size) * out.grad
        out._backward = _backward
        return out
    
    def reshape(self, shape):
        out = Tensor(self.data.reshape(shape), 
                    requires_grad=self.requires_grad,
                    _children=(self,), _op='reshape', device=self.device)
        
        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward
        return out
    
    def transpose(self, axes=None):
        out = Tensor(self.data.transpose(axes), 
                    requires_grad=self.requires_grad,
                    _children=(self,), _op='transpose', device=self.device)
        
        def _backward():
            if self.requires_grad:
                if axes is None:
                    self.grad += out.grad.transpose()
                else:
                    # Inverse permutation
                    inv_axes = np.argsort(axes)
                    self.grad += out.grad.transpose(inv_axes)
        out._backward = _backward
        return out
    
    def pad(self, pad_width, mode='constant', constant_values=0):
        out = Tensor(np.pad(self.data, pad_width, mode=mode, constant_values=constant_values),
                    requires_grad=self.requires_grad,
                    _children=(self,), _op='pad', device=self.device)
        
        def _backward():
            if self.requires_grad:
                # Slice to remove padding
                slices = []
                for i in range(self.data.ndim):
                    start = pad_width[i][0] if isinstance(pad_width[i], (list, tuple)) else pad_width[i]
                    end = -pad_width[i][1] if isinstance(pad_width[i], (list, tuple)) and pad_width[i][1] > 0 else None
                    slices.append(slice(start, end))
                self.grad += out.grad[tuple(slices)]
        out._backward = _backward
        return out
    
    def relu(self):
        out = Tensor(np.maximum(0, self.data), 
                    requires_grad=self.requires_grad,
                    _children=(self,), _op='ReLU', device=self.device)
        
        def _backward():
            if self.requires_grad:
                self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out
    
    def leaky_relu(self, alpha=0.01):
        out = Tensor(np.where(self.data > 0, self.data, alpha * self.data), 
                    requires_grad=self.requires_grad,
                    _children=(self,), _op='LeakyReLU', device=self.device)
        
        def _backward():
            if self.requires_grad:
                self.grad += np.where(self.data > 0, 1, alpha) * out.grad
        out._backward = _backward
        return out
    
    def sigmoid(self):
        s = 1 / (1 + np.exp(-self.data))
        out = Tensor(s, 
                    requires_grad=self.requires_grad,
                    _children=(self,), _op='sigmoid', device=self.device)
        
        def _backward():
            if self.requires_grad:
                self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, 
                    requires_grad=self.requires_grad,
                    _children=(self,), _op='tanh', device=self.device)
        
        def _backward():
            if self.requires_grad:
                self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out
    
    def exp(self):
        out = Tensor(np.exp(self.data), 
                    requires_grad=self.requires_grad,
                    _children=(self,), _op='exp', device=self.device)
        
        def _backward():
            if self.requires_grad:
                self.grad += out.data * out.grad
        out._backward = _backward
        return out
    
    def log(self):
        out = Tensor(np.log(self.data), 
                    requires_grad=self.requires_grad,
                    _children=(self,), _op='log', device=self.device)
        
        def _backward():
            if self.requires_grad:
                self.grad += (1 / self.data) * out.grad
        out._backward = _backward
        return out
    
    def backward(self, grad=None):
        # Topological order all of the children in the graph
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        # Initialize gradient
        if grad is None:
            if self.data.shape == ():
                self.grad = np.array(1.0, dtype=np.float32)
            else:
                self.grad = np.ones_like(self.data, dtype=np.float32)
        else:
            self.grad = grad
        
        # Go backwards through the graph
        for node in reversed(topo):
            node._backward()
    
    def zero_grad(self):
        if self.grad is not None:
            self.grad = np.zeros_like(self.grad)
    
    def to(self, device):
        # For CPU-only implementation, just update the device attribute
        self.device = device
        return self
    
    def cpu(self):
        self.device = 'cpu'
        return self
    
    def numpy(self):
        return self.data.copy()
    
    def copy(self):
        return Tensor(self.data.copy(), self.requires_grad, device=self.device)
    
    def detach(self):
        """Return a new tensor detached from the computation graph"""
        return Tensor(self.data.copy(), requires_grad=False, device=self.device)
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def dtype(self):
        return self.data.dtype
    
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __neg__(self): return self * -1
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1
    def __getitem__(self, idx):
        out = Tensor(self.data[idx], requires_grad=self.requires_grad, 
                    _children=(self,), _op='getitem', device=self.device)
        
        def _backward():
            if self.requires_grad:
                # Create zeros with the same shape as original data
                grad = np.zeros_like(self.data)
                
                # Check if idx is a single value or a slice/list
                if isinstance(idx, (int, slice, tuple, list, np.ndarray)):
                    # Add the gradient to the indexed positions
                    grad[idx] += out.grad
                else: # scalar index
                    grad[idx] = out.grad
                self.grad += grad
                
        out._backward = _backward
        return out
    
    def __setitem__(self, idx, value):
        # For simplicity, we don't support in-place operations with autograd
        if isinstance(value, Tensor):
            self.data[idx] = value.data
        else:
            self.data[idx] = value
        self._version += 1  # Track in-place modification
    
    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, device='{self.device}', op='{self._op}', requires_grad={self.requires_grad})"

# ----------------------------
# Advanced Neural Network Layers
# ----------------------------
class Module:
    def __init__(self):
        self._parameters = OrderedDict()
        self._modules = OrderedDict()
        self._buffers = OrderedDict()
        self.training = True
    
    def parameters(self, recurse=True):
        params = list(self._parameters.values())
        if recurse:
            for module in self._modules.values():
                params.extend(module.parameters(recurse))
        return params
    
    def modules(self, recurse=True):
        mods = list(self._modules.values())
        if recurse:
            for module in self._modules.values():
                mods.extend(module.modules(recurse))
        return mods
    
    def named_parameters(self, prefix='', recurse=True):
        named_params = []
        for name, param in self._parameters.items():
            named_params.append((f"{prefix}.{name}" if prefix else name, param))
        
        if recurse:
            for name, module in self._modules.items():
                named_params.extend(module.named_parameters(f"{prefix}.{name}" if prefix else name))
        
        return named_params
    
    def add_module(self, name, module):
        self._modules[name] = module
    
    def register_parameter(self, name, param):
        self._parameters[name] = param
        # Also set as attribute for direct access
        setattr(self, name, param)
    
    def register_buffer(self, name, tensor):
        self._buffers[name] = tensor
        # Also set as attribute for direct access
        setattr(self, name, tensor)
    
    def forward(self, *inputs):
        raise NotImplementedError("Subclasses must implement forward")
    
    def __call__(self, *inputs):
        return self.forward(*inputs)
    
    def zero_grad(self):
        for param in self.parameters():
            param.zero_grad()
    
    def to(self, device):
        for param in self.parameters():
            param.to(device)
        for module in self.modules():
            module.to(device)
        for buffer in self._buffers.values():
            buffer.to(device)
        return self
    
    def train(self, mode=True):
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self
    
    def eval(self):
        return self.train(False)
    
    def state_dict(self):
        state_dict = {
            'parameters': {name: param.data for name, param in self._parameters.items() if param is not None},
            'buffers': {name: buffer.data for name, buffer in self._buffers.items()}
        }
        
        for name, module in self._modules.items():
            state_dict[f'module_{name}'] = module.state_dict()
        
        return state_dict
    
    def load_state_dict(self, state_dict):
        for name, param_data in state_dict.get('parameters', {}).items():
            if name in self._parameters and self._parameters[name] is not None:
                self._parameters[name].data = param_data
        
        for name, buffer_data in state_dict.get('buffers', {}).items():
            if name in self._buffers:
                self._buffers[name].data = buffer_data
        
        for name, module in self._modules.items():
            module_key = f'module_{name}'
            if module_key in state_dict:
                module.load_state_dict(state_dict[module_key])
    
    def save(self, path):
        with gzip.open(path, 'wb') as f:
            pickle.dump(self.state_dict(), f)
    
    def load(self, path):
        with gzip.open(path, 'rb') as f:
            state_dict = pickle.load(f)
        self.load_state_dict(state_dict)

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device='cpu'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        
        # Initialize weights with Kaiming uniform initialization
        bound = np.sqrt(6 / in_features)
        weight_data = np.random.uniform(-bound, bound, (in_features, out_features))
        weight = Tensor(weight_data, requires_grad=True, device=device)
        self.register_parameter('weight', weight)
        
        if bias:
            bias_data = np.zeros(out_features)
            bias_val = Tensor(bias_data, requires_grad=True, device=device)
            self.register_parameter('bias', bias_val)
        else:
            self.bias = None
    
    def forward(self, x):
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out
    
    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, device='cpu'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.device = device
        
        # Initialize weights with Kaiming uniform initialization
        fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
        bound = np.sqrt(6 / fan_in)
        weight_shape = (out_channels, in_channels, self.kernel_size[0], self.kernel_size[1])
        weight_data = np.random.uniform(-bound, bound, weight_shape)
        weight = Tensor(weight_data, requires_grad=True, device=device)
        self.register_parameter('weight', weight)
        
        if bias:
            bias_data = np.zeros(out_channels)
            bias_val = Tensor(bias_data, requires_grad=True, device=device)
            self.register_parameter('bias', bias_val)
        else:
            self.bias = None
            
    def forward(self, x):
        batch_size, in_channels, in_height, in_width = x.shape
        
        # Calculate output dimensions
        out_height = (in_height + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        out_width = (in_width + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        
        # Add padding
        if any(self.padding):
            x_padded_data = np.pad(x.data, ((0, 0), (0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])), 'constant')
            x_padded = Tensor(x_padded_data, requires_grad=x.requires_grad, device=self.device)
        else:
            x_padded = x
        
        # Unfold the input to a matrix (im2col)
        # Simplified im2col implementation for demonstration
        col = []
        for i in range(out_height):
            for j in range(out_width):
                h_start, w_start = i * self.stride[0], j * self.stride[1]
                h_end, w_end = h_start + self.kernel_size[0], w_start + self.kernel_size[1]
                
                patch = x_padded.data[:, :, h_start:h_end, w_start:w_end]
                col.append(patch.reshape(batch_size, -1))
        
        col_matrix = np.stack(col, axis=2) # Shape: (batch_size, kernel_dim, out_h * out_w)
        
        # Reshape kernel for matrix multiplication
        weight_reshaped = self.weight.data.reshape(self.out_channels, -1)
        
        # Matrix multiplication
        output_data = np.einsum('bkc,ck->bce', col_matrix, weight_reshaped)
        
        # Reshape to final output shape
        output_data = output_data.reshape(batch_size, out_height, out_width, self.out_channels).transpose(0, 3, 1, 2)
        out = Tensor(output_data, requires_grad=x.requires_grad or self.weight.requires_grad, _children=(x, self.weight), _op='conv2d', device=self.device)
        
        # Add bias if present
        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1, 1)

        def _backward():
            if x.requires_grad:
                # This is a complex backward pass for conv2d, simplified here
                pass # A full im2col backprop is complex and would require more code
            if self.weight.requires_grad:
                # Same as above
                pass
        
        out._backward = _backward
        return out

    def extra_repr(self):
        return (f'in_channels={self.in_channels}, out_channels={self.out_channels}, '
                f'kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, '
                f'bias={self.bias is not None}')

class BatchNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, device='cpu'):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.device = device
        self.training = True
        
        if affine:
            weight_data = np.ones(num_features)
            weight = Tensor(weight_data, requires_grad=True, device=device)
            self.register_parameter('weight', weight)
            
            bias_data = np.zeros(num_features)
            bias = Tensor(bias_data, requires_grad=True, device=device)
            self.register_parameter('bias', bias)
        else:
            self.weight = None
            self.bias = None
        
        # Running statistics
        running_mean_data = np.zeros(num_features)
        running_mean = Tensor(running_mean_data, device=device)
        self.register_buffer('running_mean', running_mean)
        
        running_var_data = np.ones(num_features)
        running_var = Tensor(running_var_data, device=device)
        self.register_buffer('running_var', running_var)
        
        num_batches_tracked_data = 0
        num_batches_tracked = Tensor(num_batches_tracked_data, device=device)
        self.register_buffer('num_batches_tracked', num_batches_tracked)
    
    def forward(self, x):
        if self.training:
            # Calculate batch statistics
            mean = x.mean(axis=(0, 2, 3), keepdims=True)
            var = x.var(axis=(0, 2, 3), keepdims=True)
            
            # Update running statistics (in-place update)
            self.running_mean.data = (1 - self.momentum) * self.running_mean.data + self.momentum * mean.data.squeeze()
            self.running_var.data = (1 - self.momentum) * self.running_var.data + self.momentum * var.data.squeeze()
            self.num_batches_tracked.data += 1
            
            # Normalize
            x_normalized = (x - mean) / np.sqrt(var + self.eps)
        else:
            # Use running statistics
            mean = self.running_mean.reshape(1, -1, 1, 1)
            var = self.running_var.reshape(1, -1, 1, 1)
            x_normalized = (x - mean) / np.sqrt(var + self.eps)
        
        # Scale and shift if affine
        if self.affine:
            weight = self.weight.reshape(1, -1, 1, 1)
            bias = self.bias.reshape(1, -1, 1, 1)
            x_normalized = x_normalized * weight + bias
        
        return x_normalized
    
    def extra_repr(self):
        return (f'num_features={self.num_features}, eps={self.eps}, momentum={self.momentum}, '
                f'affine={self.affine}')

class Dropout(Module):
    def __init__(self, p=0.5, device='cpu'):
        super().__init__()
        self.p = p
        self.device = device
        self.training = True
    
    def forward(self, x):
        if self.training and self.p > 0:
            # Generate mask
            mask = Tensor(np.random.binomial(1, 1 - self.p, x.shape), requires_grad=False, device=self.device)
            return x * mask / (1 - self.p)
        return x
    
    def extra_repr(self):
        return f'p={self.p}'

class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        for i, layer in enumerate(layers):
            if isinstance(layer, type) and issubclass(layer, Module):
                raise TypeError("Sequential expects instances of modules, not classes.")
            self.add_module(str(i), layer)
    
    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x

class Residual(Module):
    def __init__(self, layer, shortcut=None):
        super().__init__()
        self.layer = layer
        self.shortcut = shortcut
        self.add_module('layer', layer)
        if shortcut:
            self.add_module('shortcut', shortcut)
    
    def forward(self, x):
        identity = x if self.shortcut is None else self.shortcut(x)
        return self.layer(x) + identity

# ----------------------------
# Advanced Optimizers
# ----------------------------
class Optimizer:
    def __init__(self, parameters, lr):
        self.parameters = list(parameters)
        self.lr = lr
        self.state = defaultdict(dict)
    
    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.zero_grad()
    
    def step(self):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, parameters, lr=0.01, momentum=0, weight_decay=0, nesterov=False):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
    
    def step(self):
        for param in self.parameters:
            if param.grad is None:
                continue
                
            # Apply weight decay
            grad = param.grad
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
            
            # Apply momentum
            if self.momentum != 0:
                param_state = self.state[param]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = np.zeros_like(param.data)
                else:
                    buf = param_state['momentum_buffer']
                
                buf = self.momentum * buf + grad
                param_state['momentum_buffer'] = buf
                
                if self.nesterov:
                    grad = grad + self.momentum * buf
                else:
                    grad = buf
            
            # Update parameters
            param.data -= self.lr * grad

class Adam(Optimizer):
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        super().__init__(parameters, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.t = 0
    
    def step(self):
        self.t += 1
        for param in self.parameters:
            if param.grad is None:
                continue
                
            # Apply weight decay
            grad = param.grad
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
            
            param_state = self.state[param]
            
            # Initialize state
            if 'm' not in param_state:
                param_state['m'] = np.zeros_like(param.data)
                param_state['v'] = np.zeros_like(param.data)
                if self.amsgrad:
                    param_state['v_hat'] = np.zeros_like(param.data)
            
            m, v = param_state['m'], param_state['v']
            
            # Update biased first moment estimate
            m = self.beta1 * m + (1 - self.beta1) * grad
            # Update biased second raw moment estimate
            v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
            
            param_state['m'] = m
            param_state['v'] = v
            
            if self.amsgrad:
                v_hat = param_state['v_hat']
                v_hat = np.maximum(v_hat, v)
                param_state['v_hat'] = v_hat
                denom = np.sqrt(v_hat) + self.eps
            else:
                denom = np.sqrt(v) + self.eps
            
            # Compute bias-corrected first moment estimate
            m_hat = m / (1 - self.beta1 ** self.t)
            
            # Update parameters
            param.data -= self.lr * m_hat / denom

# ----------------------------
# Learning Rate Schedulers
# ----------------------------
class LRScheduler:
    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.base_lrs = [optimizer.lr]
    
    def step(self):
        self.last_epoch += 1
        lr = self.get_lr()
        self.optimizer.lr = lr
    
    def get_lr(self):
        raise NotImplementedError

class StepLR(LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        super().__init__(optimizer, last_epoch)
        self.step_size = step_size
        self.gamma = gamma
    
    def get_lr(self):
        if self.last_epoch == 0 or self.last_epoch % self.step_size != 0:
            return self.base_lrs[0]
        return self.base_lrs[0] * (self.gamma ** (self.last_epoch // self.step_size))

class ExponentialLR(LRScheduler):
    def __init__(self, optimizer, gamma, last_epoch=-1):
        super().__init__(optimizer, last_epoch)
        self.gamma = gamma
    
    def get_lr(self):
        return self.base_lrs[0] * (self.gamma ** self.last_epoch)

class ReduceLROnPlateau(LRScheduler):
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, threshold=1e-4, 
                 threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8, last_epoch=-1):
        super().__init__(optimizer, last_epoch)
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.eps = eps
        
        self.cooldown_counter = 0
        self.best = None
        self.num_bad_epochs = 0
        
        self._reset()
    
    def _reset(self):
        self.best = None
        self.cooldown_counter = 0
        self.num_bad_epochs = 0
    
    def step(self, metrics):
        current = float(metrics)
        self.last_epoch += 1
        
        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        
        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0
        
        if self.num_bad_epochs > self.patience:
            self._reduce_lr()
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
        
        self.optimizer.lr = max(self.optimizer.lr, self.min_lr)
    
    def _reduce_lr(self):
        old_lr = self.optimizer.lr
        new_lr = max(old_lr * self.factor, self.min_lr)
        self.optimizer.lr = new_lr
    
    def is_better(self, a, b):
        if b is None:
            return True
        
        if self.mode == 'min' and self.threshold_mode == 'rel':
            return a < b - b * self.threshold
        
        if self.mode == 'min' and self.threshold_mode == 'abs':
            return a < b - self.threshold
        
        if self.mode == 'max' and self.threshold_mode == 'rel':
            return a > b + b * self.threshold
        
        if self.mode == 'max' and self.threshold_mode == 'abs':
            return a > b + self.threshold
    
    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

# ----------------------------
# Loss Functions
# ----------------------------
def mse_loss(output, target, reduction='mean'):
    diff = output - target
    loss = (diff * diff)
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:  # 'none'
        return loss

def cross_entropy_loss(output, target, reduction='mean', label_smoothing=0.0):
    # Numerically stable softmax
    max_vals = output.data.max(axis=1, keepdims=True)
    exp_output = np.exp(output.data - max_vals)
    softmax_output = exp_output / exp_output.sum(axis=1, keepdims=True)
    
    # Apply label smoothing
    if label_smoothing > 0:
        num_classes = output.shape[1]
        target_one_hot = np.eye(num_classes)[target.data.astype(int)]
        target_one_hot = (1 - label_smoothing) * target_one_hot + label_smoothing / num_classes
        
        # log_probs = (np.log(softmax_output) * target_one_hot).sum(axis=1)
        # Use log-softmax for numerical stability
        log_softmax = output.data - max_vals - np.log(exp_output.sum(axis=1, keepdims=True))
        loss_data = -(log_softmax * target_one_hot).sum(axis=1)
        loss = Tensor(loss_data, requires_grad=output.requires_grad, device=output.device)
    else:
        # Cross entropy loss
        batch_size = output.shape[0]
        loss_data = -np.log(softmax_output[range(batch_size), target.data.astype(int)])
        loss = Tensor(loss_data, requires_grad=output.requires_grad, _children=(output,), _op='cross_entropy', device=output.device)
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:  # 'none'
        return loss

def nll_loss(output, target, reduction='mean'):
    batch_size = output.shape[0]
    loss = -output[range(batch_size), target.data.astype(int)]
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:  # 'none'
        return loss

def binary_cross_entropy(output, target, reduction='mean'):
    # Clamp to avoid numerical issues
    output_clamped = np.clip(output.data, 1e-7, 1 - 1e-7)
    loss = -(target.data * np.log(output_clamped) + (1 - target.data) * np.log(1 - output_clamped))
    loss = Tensor(loss, requires_grad=output.requires_grad, device=output.device)
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:  # 'none'
        return loss

# ----------------------------
# Data Loading and Augmentation
# ----------------------------
class Dataset:
    def __init__(self, transform=None):
        self.transform = transform
    
    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self, index):
        raise NotImplementedError

class TensorDataset(Dataset):
    def __init__(self, *tensors, transform=None):
        super().__init__(transform)
        assert all(tensors[0].shape[0] == tensor.shape[0] for tensor in tensors)
        self.tensors = tensors
    
    def __len__(self):
        return self.tensors[0].shape[0]
    
    def __getitem__(self, index):
        samples = [tensor[index] for tensor in self.tensors]
        
        if self.transform:
            samples = self.transform(samples)
        
        return samples

class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.indices = np.arange(len(dataset))
        self.current_index = 0
        
        if shuffle:
            np.random.shuffle(self.indices)
    
    def __iter__(self):
        self.current_index = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self
    
    def __next__(self):
        if self.current_index >= len(self.dataset):
            raise StopIteration
        
        end_index = min(self.current_index + self.batch_size, len(self.dataset))
        batch_indices = self.indices[self.current_index:end_index]
        
        # Get batch samples and convert to a list of numpy arrays
        batch_samples = [self.dataset[i] for i in batch_indices]
        
        if isinstance(batch_samples[0], (list, tuple)):
            # Multiple inputs/targets
            result = []
            for i in range(len(batch_samples[0])):
                data = np.stack([s[i].data if hasattr(s[i], 'data') else s[i] for s in batch_samples])
                result.append(Tensor(data))
        else:
            # Single input
            data = np.stack([s.data if hasattr(s, 'data') else s for s in batch_samples])
            result = [Tensor(data)]
        
        self.current_index = end_index
        
        return result
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

# ----------------------------
# Training Utilities
# ----------------------------
def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, 
                device='cpu', scheduler=None, checkpoint_path=None, early_stopping_patience=None):
    model.train()
    model.to(device)
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        start_time = time.time()
        
        for i, batch in enumerate(train_loader):
            inputs, targets = batch if len(batch) == 2 else (batch[0], None)
            inputs = inputs.to(device)
            targets = targets.to(device) if targets is not None else None
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets) if targets is not None else criterion(outputs)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.data
            
            if i % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {i}, Loss: {loss.data:.4f}')
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        
        # Custom "no_grad" context manager equivalent
        # This is a simplification; in a real framework, you'd disable gradient tracking
        # globally. For this code, we just don't call backward.
        
        for batch in val_loader:
            inputs, targets = batch if len(batch) == 2 else (batch[0], None)
            inputs = inputs.to(device)
            targets = targets.to(device) if targets is not None else None
            
            outputs = model(inputs)
            loss = criterion(outputs, targets) if targets is not None else criterion(outputs)
            total_val_loss += loss.data
        
        # Calculate average losses
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Update learning rate if scheduler is provided
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        # Checkpointing
        if checkpoint_path and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save(f"{checkpoint_path}_best.pth")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if early_stopping_patience and patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Time: {epoch_time:.2f}s')
    
    return history

# ----------------------------
# Example Usage
# ----------------------------
if __name__ == "__main__":
    # Create a simple neural network
    class SimpleNN(Module):
        def __init__(self, device='cpu'):
            super().__init__()
            self.layers = Sequential(
                Linear(784, 256, device=device),
                lambda x: x.relu(),
                Dropout(0.2, device=device),
                Linear(256, 128, device=device),
                lambda x: x.relu(),
                Dropout(0.2, device=device),
                Linear(128, 10, device=device)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    # Create model, optimizer and data
    device = 'cpu'
    model = SimpleNN(device=device)
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    # Generate dummy data
    X_train = Tensor(np.random.randn(1000, 784), requires_grad=False, device=device)
    y_train = Tensor(np.random.randint(0, 10, size=(1000,)), requires_grad=False, device=device)
    X_val = Tensor(np.random.randn(200, 784), requires_grad=False, device=device)
    y_val = Tensor(np.random.randint(0, 10, size=(200,)), requires_grad=False, device=device)
    
    # Create datasets and data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
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
        checkpoint_path='model',
        early_stopping_patience=10
    )
    
    print("Training completed!")
    print(f"Final Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"Final Val Loss: {history['val_loss'][-1]:.4f}")
    print(f"Memory Usage: {memory_manager.get_memory_info()}")