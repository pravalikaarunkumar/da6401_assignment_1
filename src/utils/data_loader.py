"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""

import numpy as np
from sklearn.model_selection import train_test_split


def load_dataset(dataset: str = "fashion_mnist"):
    if dataset == "mnist":
        from keras.datasets import mnist as loader
    elif dataset == "fashion_mnist":
        from keras.datasets import fashion_mnist as loader
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose 'mnist' or 'fashion_mnist'.")

    (X_full, y_full), (X_test, y_test) = loader.load_data()

    X_full = X_full.reshape(-1, 784).astype(np.float64) / 255.0
    X_test = X_test.reshape(-1, 784).astype(np.float64) / 255.0

    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full,
        test_size=0.1,
        random_state=42,
        stratify=y_full,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def batch_iterator(X: np.ndarray, y: np.ndarray,
                   batch_size: int, shuffle: bool = True):
    n   = X.shape[0]
    idx = np.random.permutation(n) if shuffle else np.arange(n)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        b   = idx[start:end]
        yield X[b], y[b]

FASHION_MNIST_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal",      "Shirt",   "Sneaker",  "Bag",   "Ankle boot",
]
MNIST_CLASSES = [str(i) for i in range(10)]
