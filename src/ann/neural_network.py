"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np
from ann.neural_layer import NeuralLayer
from ann.objective_functions import get_loss
from ann.activations import get_activation


class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """

    def __init__(self, cli_args):
        cfg = vars(cli_args) if hasattr(cli_args, "__dict__") else dict(cli_args)

        self.activation_name = cfg.get("activation", "relu")
        self.loss_name       = cfg.get("loss", "cross_entropy")
        weight_init          = cfg.get("weight_init", "xavier")

        hidden_size = cfg.get("hidden_size", [128])
        if isinstance(hidden_size, int):
            num_layers  = cfg.get("num_layers", 3)
            hidden_size = [hidden_size] * num_layers

        self.layers = []
        n = 784  

        for h in hidden_size:
            self.layers.append(
                NeuralLayer(n, h,
                            activation=self.activation_name,
                            weight_init=weight_init,
                            is_output=False)
            )
            n = h

        self.layers.append(
            NeuralLayer(n, 10,
                        weight_init=weight_init,
                        is_output=True)
        )

        self.grad_W = None
        self.grad_b = None

    def forward(self, X):
        """
        Forward propagation through all layers.
        Returns logits (no softmax applied)
        X is shape (b, D_in) and output is shape (b, D_out).
        b is batch size, D_in is input dimension, D_out is output dimension.
        """
        a = X
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def backward(self, y_true_or_dlogits, y_pred=None, weight_decay=0.0):
        """
        Backward propagation to compute gradients.
        Returns two numpy arrays: grad_Ws, grad_bs.
        - `grad_Ws[0]` is gradient for the last (output) layer weights,
          `grad_bs[0]` is gradient for the last layer biases, and so on.

        Accepts two calling conventions:
          backward(y_true, y_pred)   - computes loss gradient internally
          backward(dlogits)          - uses provided gradient directly
        """
        if y_pred is None:
            dlogits = y_true_or_dlogits
        else:
            loss_fn    = get_loss(self.loss_name)
            _, dlogits = loss_fn(y_pred, y_true_or_dlogits)

        grad_W_list = []
        grad_b_list = []

        delta = dlogits
        for layer in reversed(self.layers):
            delta = layer.backward(delta, weight_decay=weight_decay)
            grad_W_list.append(layer.grad_W.copy())
            grad_b_list.append(layer.grad_b.copy())

        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        print("Shape of grad_Ws:", self.grad_W.shape, self.grad_W[0].shape)
        print("Shape of grad_bs:", self.grad_b.shape, self.grad_b[0].shape)

        return self.grad_W, self.grad_b

    def update_weights(self, optimizer=None):
        if optimizer:
            optimizer.update(self.layers)

    def train(self, X_train, y_train, epochs=1, batch_size=32):
        from utils.data_loader import batch_iterator
        loss_fn = get_loss(self.loss_name)
        for epoch in range(epochs):
            losses = []
            for Xb, yb in batch_iterator(X_train, y_train, batch_size):
                logits  = self.forward(Xb)
                loss, _ = loss_fn(logits, yb)
                self.backward(yb, logits)
                losses.append(loss)
            print(f"Epoch {epoch+1}/{epochs} loss={np.mean(losses):.4f}")

    def evaluate(self, X, y):
        preds = self.predict(X)
        return float(np.mean(preds == y))

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):
        num_w = sum(1 for k in weight_dict
                    if k.startswith('W') and k[1:].lstrip('_').isdigit())

        if num_w != len(self.layers):
            from ann.neural_layer import NeuralLayer
            if '_config' in weight_dict:
                act = weight_dict['_config'].get('activation', 'relu')
                wi  = weight_dict['_config'].get('weight_init', 'xavier')
            else:
                act = self.activation_name
                wi  = 'xavier'

            self.layers = []
            for i in range(num_w):
                wk = f'W{i}' if f'W{i}' in weight_dict else f'W_{i}'
                bk = f'b{i}' if f'b{i}' in weight_dict else f'b_{i}'
                W  = weight_dict[wk]
                b  = weight_dict[bk]
                is_out = (i == num_w - 1)
                layer = NeuralLayer(W.shape[0], W.shape[1],
                                    activation=act,
                                    weight_init=wi,
                                    is_output=is_out)
                layer.W      = W.copy()
                layer.b      = b.copy()
                layer.grad_W = np.zeros_like(W)
                layer.grad_b = np.zeros_like(b)
                self.layers.append(layer)
        else:
            for i, layer in enumerate(self.layers):
                wk = f'W{i}' if f'W{i}' in weight_dict else f'W_{i}'
                bk = f'b{i}' if f'b{i}' in weight_dict else f'b_{i}'
                if wk in weight_dict:
                    layer.W = weight_dict[wk].copy()
                if bk in weight_dict:
                    layer.b = weight_dict[bk].copy()

    @classmethod
    def from_weights(cls, weight_dict):
        import argparse
        if '_config' in weight_dict:
            c = weight_dict['_config']
            hidden_size = [c['hidden_size']] * c['num_layers']
            activation  = c.get('activation', 'relu')
        else:
            num_total   = sum(1 for k in weight_dict if k.startswith('W') and k[1:].isdigit())
            hidden_size = [weight_dict[f'W{i}'].shape[1] for i in range(num_total - 1)]
            activation  = 'relu'

        cfg = argparse.Namespace(
            hidden_size = hidden_size,
            activation  = activation,
            weight_init = 'xavier',
            loss        = 'cross_entropy',
        )
        model = cls(cfg)
        model.set_weights(weight_dict)
        return model
