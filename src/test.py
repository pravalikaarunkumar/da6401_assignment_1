import numpy as np
import argparse
from ann.neural_network import NeuralNetwork

best_config = argparse.Namespace(
    dataset="fashion_mnist", epochs=20, batch_size=32,
    loss="cross_entropy", optimizer="rmsprop", weight_decay=0.0,
    learning_rate=0.00045024653096303305, num_layers=3,
    hidden_size=[128, 128, 128], activation="relu",
    weight_init="xavier", input_size=784, num_classes=10
)

model = NeuralNetwork(best_config)
weights = np.load("best_model.npy", allow_pickle=True).item()
model.set_weights(weights)
print("Model loaded successfully!")

# test prediction
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.data_loader import load_dataset
_, _, X_test, _, _, y_test = load_dataset("fashion_mnist")
preds = model.predict(X_test)
from sklearn.metrics import f1_score
f1 = f1_score(y_test, preds, average="macro")
print(f"F1 Score: {f1:.4f}")