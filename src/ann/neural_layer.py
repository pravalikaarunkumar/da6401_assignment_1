"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""

import numpy as np
from ann.activations import get_activation

class NeuralLayer:
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = "relu",
        weight_init: str = "xavier",
        is_output: bool = False,
    ):
        self.in_features  = in_features
        self.out_features = out_features
        self.is_output    = is_output

        #weight initialisation
        if weight_init == "xavier":
            limit = np.sqrt(6.0 / (in_features + out_features))
            self.W = np.random.uniform(-limit, limit,
                                       (in_features, out_features))
        else:
            self.W = np.random.randn(in_features, out_features) * 0.01

        self.b = np.zeros((1, out_features))

        #activation 
        if not is_output:
            self._act_fn, self._act_deriv = get_activation(activation)
        else:
            self._act_fn = self._act_deriv = None

        self._input = None  
        self._z     = None  

        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

    #forward pass
    def forward(self, a: np.ndarray) -> np.ndarray:
        self._input = a
        self._z     = a @ self.W + self.b        

        if self.is_output:
            return self._z                     
        return self._act_fn(self._z)

    #backward pass
    def backward(self, delta: np.ndarray, weight_decay: float = 0.0) -> np.ndarray:
        batch = self._input.shape[0]

        if not self.is_output:
            delta = delta * self._act_deriv(self._z)   

        self.grad_W = (self._input.T @ delta) / batch + weight_decay * self.W
        self.grad_b = np.mean(delta, axis=0, keepdims=True)

        return delta @ self.W.T
