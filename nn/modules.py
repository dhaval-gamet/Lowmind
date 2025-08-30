from collections import OrderedDict
import pickle
import gzip
import numpy as np
from core.tensor import Tensor

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
    
    def register_buffer(self, name, tensor):
        self._buffers[name] = tensor
    
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
        for module in self.modules():
            module.train(mode)
        return self
    
    def eval(self):
        return self.train(False)
    
    def state_dict(self):
        state_dict = {
            'parameters': {name: param.data for name, param in self._parameters.items()},
            'buffers': {name: buffer.data for name, buffer in self._buffers.items()}
        }
        
        for name, module in self._modules.items():
            state_dict[f'module_{name}'] = module.state_dict()
        
        return state_dict
    
    def load_state_dict(self, state_dict):
        for name, param_data in state_dict.get('parameters', {}).items():
            if name in self._parameters:
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
        
        bound = np.sqrt(6 / in_features)
        weight = Tensor(np.random.uniform(-bound, bound, (in_features, out_features)), 
                       requires_grad=True, device=device)
        self.register_parameter('weight', weight)
        
        if bias:
            bias_val = Tensor(np.zeros(out_features), requires_grad=True, device=device)
            self.register_parameter('bias', bias_val)
        else:
            self.register_parameter('bias', None)
    
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
        
        fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
        bound = np.sqrt(6 / fan_in)
        weight_shape = (out_channels, in_channels, self.kernel_size[0], self.kernel_size[1])
        weight = Tensor(np.random.uniform(-bound, bound, weight_shape), 
                       requires_grad=True, device=device)
        self.register_parameter('weight', weight)
        
        if bias:
            bias_val = Tensor(np.zeros(out_channels), requires_grad=True, device=device)
            self.register_parameter('bias', bias_val)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        batch_size, in_channels, in_height, in_width = x.shape
        
        out_height = (in_height + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        out_width = (in_width + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        
        if any(self.padding):
            x_padded = x.pad(((0, 0), (0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])))
        else:
            x_padded = x
        
        out = Tensor(np.zeros((batch_size, self.out_channels, out_height, out_width)), 
                    requires_grad=x.requires_grad, device=self.device)
        
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride[0]
                h_end = h_start + self.kernel_size[0]
                w_start = j * self.stride[1]
                w_end = w_start + self.kernel_size[1]
                
                region = x_padded[:, :, h_start:h_end, w_start:w_end]
                
                region_reshaped = region.reshape(batch_size, -1)
                weight_reshaped = self.weight.reshape(self.out_channels, -1)
                
                out.data[:, :, i, j] = region_reshaped.data @ weight_reshaped.data.T
                
                if self.bias is not None:
                    out.data[:, :, i, j] = out.data[:, :, i, j] + self.bias.data
        
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
        
        if affine:
            weight = Tensor(np.ones(num_features), requires_grad=True, device=device)
            bias = Tensor(np.zeros(num_features), requires_grad=True, device=device)
            self.register_parameter('weight', weight)
            self.register_parameter('bias', bias)
        else:
            self.weight = None
            self.bias = None
        
        self.register_buffer('running_mean', Tensor(np.zeros(num_features), device=device))
        self.register_buffer('running_var', Tensor(np.ones(num_features), device=device))
        self.register_buffer('num_batches_tracked', Tensor(0, device=device))
    
    def forward(self, x):
        if self.training:
            mean = x.data.mean(axis=(0, 2, 3), keepdims=True)
            var = x.data.var(axis=(0, 2, 3), keepdims=True)
            
            self.running_mean.data = (1 - self.momentum) * self.running_mean.data + self.momentum * mean.squeeze()
            self.running_var.data = (1 - self.momentum) * self.running_var.data + self.momentum * var.squeeze()
            self.num_batches_tracked.data += 1
            
            x_normalized = (x.data - mean) / np.sqrt(var + self.eps)
        else:
            mean = self.running_mean.data.reshape(1, -1, 1, 1)
            var = self.running_var.data.reshape(1, -1, 1, 1)
            x_normalized = (x.data - mean) / np.sqrt(var + self.eps)
        
        out = Tensor(x_normalized, requires_grad=x.requires_grad, _children=(x,), _op='batchnorm')
        
        if self.affine:
            weight_tensor = self._parameters['weight'].data.reshape(1, -1, 1, 1)
            bias_tensor = self._parameters['bias'].data.reshape(1, -1, 1, 1)
            out.data = out.data * weight_tensor + bias_tensor
        
        return out
    
    def extra_repr(self):
        return (f'num_features={self.num_features}, eps={self.eps}, momentum={self.momentum}, '
                f'affine={self.affine}')

class Dropout(Module):
    def __init__(self, p=0.5, device='cpu'):
        super().__init__()
        self.p = p
        self.device = device
    
    def forward(self, x):
        if self.training and self.p > 0:
            mask = Tensor(np.random.binomial(1, 1 - self.p, x.shape), device=self.device)
            return x * mask / (1 - self.p)
        return x
    
    def extra_repr(self):
        return f'p={self.p}'

class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        for i, layer in enumerate(layers):
            self.add_module(str(i), layer)
    
    def forward(self, x):
        for module in self._modules.values():
            if callable(module):  # Handles activation functions passed directly
                x = module(x)
            else:
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

