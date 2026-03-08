"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""
import numpy as np
from ann.activations import softmax

#cross-entropy Loss
def cross_entropy_loss(logits: np.ndarray, labels: np.ndarray): 
    batch       = logits.shape[0]
    probs       = softmax(logits)                         
    one_hot     = np.zeros_like(probs)
    one_hot[np.arange(batch), labels] = 1.0
    loss        = -np.sum(one_hot * np.log(probs + 1e-12)) / batch
    dlogits     = (probs - one_hot) / batch
    return loss, dlogits


#mean-squared-error 
def mse_loss(logits: np.ndarray, labels: np.ndarray):
    batch       = logits.shape[0]
    probs       = softmax(logits)                          
    one_hot     = np.zeros_like(probs)
    one_hot[np.arange(batch), labels] = 1.0
    diff        = probs - one_hot                         
    loss        = np.mean(np.sum(diff ** 2, axis=1))
    dL_dp       = 2.0 * diff / batch
    dot         = np.sum(dL_dp * probs, axis=1, keepdims=True)   
    dlogits     = probs * (dL_dp - dot)
    return loss, dlogits

_LOSS_REGISTRY = {
    "cross_entropy": cross_entropy_loss,
    "ce":            cross_entropy_loss,
    "mse":           mse_loss,
    "mean_squared_error": mse_loss,
}

def get_loss(name: str):
    name = name.lower()
    if name not in _LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss '{name}'. "
            f"Choose from: cross_entropy, mse"
        )
    return _LOSS_REGISTRY[name]
