import numpy as np
from collections import defaultdict

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
        size = tensor.data.nbytes + (tensor.grad.nbytes if tensor.grad is not None else 0)
        
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
        to_remove = []
        for name, (tensor, size) in self.tensors.items():
            if len(tensor._prev) == 0 and not tensor.requires_grad:
                to_remove.append(name)
        
        for name in to_remove:
            self.free(name)
    
    def get_memory_info(self):
        return {
            'allocated_mb': self.allocated_memory / (1024 * 1024),
            'max_mb': self.max_memory / (1024 * 1024),
            'usage_percent': (self.allocated_memory / self.max_memory) * 100
        }

memory_manager = MemoryManager(max_memory_mb=512)

# ----------------------------
# Enhanced Tensor Class
# ----------------------------
class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op='', device='cpu', name=None):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.device = device
        self.name = name
        self._version = 0
        
        memory_manager.allocate(self, name)
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        out = Tensor(self.data + other.data, 
                    requires_grad=self.requires_grad or other.requires_grad,
                    _children=(self, other), _op='+', device=self.device)
        
        def _backward():
            if self.requires_grad:
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
                if axis is not None:
                    expanded_grad = out.grad
                    if not keepdims:
                        expanded_grad = np.expand_dims(expanded_grad, axis=axis if isinstance(axis, int) else None)
                    grad_shape = np.ones_like(self.data.shape)
                    if isinstance(axis, int):
                        grad_shape[axis] = self.data.shape[axis]
                    elif isinstance(axis, tuple):
                        for ax in axis:
                            grad_shape[ax] = self.data.shape[ax]
                    self.grad += expanded_grad * grad_shape
                else:
                    self.grad += np.ones_like(self.data) * out.grad
        out._backward = _backward
        return out
    
    def mean(self, axis=None, keepdims=False):
        n = np.prod(self.data.shape if axis is None else [self.data.shape[ax] for ax in (axis if isinstance(axis, tuple) else (axis,))])
        out = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims), 
                    requires_grad=self.requires_grad,
                    _children=(self,), _op='mean', device=self.device)
        
        def _backward():
            if self.requires_grad:
                expanded_grad = out.grad
                if not keepdims:
                    expanded_grad = np.expand_dims(expanded_grad, axis=axis if isinstance(axis, int) else None)
                grad_shape = np.ones_like(self.data.shape)
                if isinstance(axis, int):
                    grad_shape[axis] = self.data.shape[axis]
                elif isinstance(axis, tuple):
                    for ax in axis:
                        grad_shape[ax] = self.data.shape[ax]
                self.grad += (expanded_grad * grad_shape) / n
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
        if not self.requires_grad:
            return
            
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        if grad is None:
            self.grad = np.array(1.0, dtype=np.float32)
        else:
            self.grad = grad
        
        for node in reversed(topo):
            node._backward()
    
    def zero_grad(self):
        if self.grad is not None:
            self.grad = np.zeros_like(self.grad)
    
    def to(self, device):
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
                grad = np.zeros_like(self.data)
                grad[idx] += out.grad
                self.grad += grad
        out._backward = _backward
        return out
    
    def __setitem__(self, idx, value):
        if isinstance(value, Tensor):
            self.data[idx] = value.data
        else:
            self.data[idx] = value
        self._version += 1
    
    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, device='{self.device}', op='{self._op}', requires_grad={self.requires_grad})"

