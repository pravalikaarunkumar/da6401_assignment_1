"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""

import numpy as np
#individual functions
def sigmoid(z: np.ndarray) -> np.ndarray: #sigmoid= 1 / (1 + exp(-z))
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z))
    )

def sigmoid_derivative(z: np.ndarray) -> np.ndarray: #d(sigmoid)/dz= sigmoid(z) * (1 - sigmoid(z))
    s = sigmoid(z)
    return s * (1.0 - s)

def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)

def tanh_derivative(z: np.ndarray) -> np.ndarray: #d(tanh)/dz= 1 - tanh(z)^2
    return 1.0 - np.tanh(z) ** 2

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)

def relu_derivative(z: np.ndarray) -> np.ndarray: #sub-gradient: 1 if z > 0, else 0
    return (z > 0).astype(np.float64)

def softmax(z: np.ndarray) -> np.ndarray: #z: (batch_size, num_classes)
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


_ACTIVATION_REGISTRY = {
    "sigmoid": (sigmoid, sigmoid_derivative),
    "tanh":    (tanh,    tanh_derivative),
    "relu":    (relu,    relu_derivative),
}


def get_activation(name: str):
    name = name.lower()
    if name not in _ACTIVATION_REGISTRY:
        raise ValueError(
            f"Unknown activation '{name}'. "
            f"Choose from: {list(_ACTIVATION_REGISTRY.keys())}"
        )
    return _ACTIVATION_REGISTRY[name]
