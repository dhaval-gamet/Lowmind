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

