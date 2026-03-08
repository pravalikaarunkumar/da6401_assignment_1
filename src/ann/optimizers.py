"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""

import numpy as np

class BaseOptimizer:
    def __init__(self, lr: float):
        self.lr = lr

    def update(self, layers):
        raise NotImplementedError


#SGD  (simple mini-batch gradient descent)
class SGD(BaseOptimizer):
    def update(self, layers):
        for layer in layers:
            layer.W -= self.lr * layer.grad_W
            layer.b -= self.lr * layer.grad_b

#SGD with momentum
class Momentum(BaseOptimizer):
    def __init__(self, lr: float, beta: float = 0.9):
        super().__init__(lr)
        self.beta = beta
        self._vW  = None
        self._vb  = None

    def _init_state(self, layers):
        self._vW = [np.zeros_like(l.W) for l in layers]
        self._vb = [np.zeros_like(l.b) for l in layers]

    def update(self, layers):
        if self._vW is None:
            self._init_state(layers)

        for i, layer in enumerate(layers):
            self._vW[i] = self.beta * self._vW[i] + layer.grad_W
            self._vb[i] = self.beta * self._vb[i] + layer.grad_b
            layer.W    -= self.lr * self._vW[i]
            layer.b    -= self.lr * self._vb[i]

#Nesterov Accelerated Gradient (NAG)
class NAG(BaseOptimizer):
    def __init__(self, lr: float, beta: float = 0.9):
        super().__init__(lr)
        self.beta = beta
        self._vW  = None
        self._vb  = None

    def _init_state(self, layers):
        self._vW = [np.zeros_like(l.W) for l in layers]
        self._vb = [np.zeros_like(l.b) for l in layers]

    def update(self, layers):
        if self._vW is None:
            self._init_state(layers)

        for i, layer in enumerate(layers):
            self._vW[i] = self.beta * self._vW[i] + layer.grad_W
            self._vb[i] = self.beta * self._vb[i] + layer.grad_b

            layer.W -= self.lr * (self.beta * self._vW[i] + layer.grad_W)
            layer.b -= self.lr * (self.beta * self._vb[i] + layer.grad_b)

# RMSProp
class RMSProp(BaseOptimizer):
    def __init__(self, lr: float, beta: float = 0.9, eps: float = 1e-8):
        super().__init__(lr)
        self.beta = beta
        self.eps  = eps
        self._sW  = None
        self._sb  = None

    def _init_state(self, layers):
        self._sW = [np.zeros_like(l.W) for l in layers]
        self._sb = [np.zeros_like(l.b) for l in layers]

    def update(self, layers):
        if self._sW is None:
            self._init_state(layers)

        for i, layer in enumerate(layers):
            self._sW[i] = (self.beta * self._sW[i]
                           + (1.0 - self.beta) * layer.grad_W ** 2)
            self._sb[i] = (self.beta * self._sb[i]
                           + (1.0 - self.beta) * layer.grad_b ** 2)

            layer.W -= self.lr * layer.grad_W / (np.sqrt(self._sW[i]) + self.eps)
            layer.b -= self.lr * layer.grad_b / (np.sqrt(self._sb[i]) + self.eps)

_OPTIMIZER_REGISTRY = {
    "sgd":      SGD,
    "momentum": Momentum,
    "nag":      NAG,
    "rmsprop":  RMSProp,
}

def get_optimizer(name: str, lr: float, **kwargs) -> BaseOptimizer:
    name = name.lower()
    if name not in _OPTIMIZER_REGISTRY:
        raise ValueError(
            f"Unknown optimizer '{name}'. "
            f"Choose from: {list(_OPTIMIZER_REGISTRY.keys())}"
        )
    return _OPTIMIZER_REGISTRY[name](lr, **kwargs)
