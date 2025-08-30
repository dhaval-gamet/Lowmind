from collections import defaultdict
import numpy as np

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
                
            grad = param.grad
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
            
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
                
            grad = param.grad
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
            
            param_state = self.state[param]
            
            if 'm' not in param_state:
                param_state['m'] = np.zeros_like(param.data)
                param_state['v'] = np.zeros_like(param.data)
                if self.amsgrad:
                    param_state['v_hat'] = np.zeros_like(param.data)
            
            m, v = param_state['m'], param_state['v']
            
            m = self.beta1 * m + (1 - self.beta1) * grad
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
            
            m_hat = m / (1 - self.beta1 ** self.t)
            
            param.data -= self.lr * m_hat / denom
