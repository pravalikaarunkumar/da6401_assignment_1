"""
Inference Script
Evaluate trained models on test sets
"""
import argparse
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ann.neural_network import NeuralNetwork
from utils.data_loader  import load_dataset
from sklearn.metrics    import (accuracy_score, precision_score,
                                recall_score, f1_score, confusion_matrix)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run inference on test set')
    parser.add_argument("-d",   "--dataset",       type=str,   choices=["mnist", "fashion_mnist"], default="fashion_mnist")
    parser.add_argument("-e",   "--epochs",         type=int,   default=20)
    parser.add_argument("-b",   "--batch_size",     type=int,   default=32)
    parser.add_argument("-lr",  "--learning_rate",  type=float, default=0.00045024653096303305)
    parser.add_argument("-wd",  "--weight_decay",   type=float, default=0.0)
    parser.add_argument("-l",   "--loss",           type=str,   choices=["mse", "cross_entropy"], default="cross_entropy")
    parser.add_argument("-o",   "--optimizer",      type=str,   choices=["sgd", "momentum", "nag", "rmsprop"], default="rmsprop")
    parser.add_argument("-nhl", "--num_layers",     type=int,   default=3)
    parser.add_argument("-sz",  "--hidden_size",    nargs="+",  type=int, default=[128, 128, 128])
    parser.add_argument("-a",   "--activation",     type=str,   choices=["sigmoid", "tanh", "relu"], default="tanh")
    parser.add_argument("-w_i", "--weight_init",    type=str,   choices=["random", "xavier"], default="xavier")
    parser.add_argument("-w_p", "--wandb_project",  type=str,   default="da6401_assignment1")
    parser.add_argument("-m",   "--model_path",     type=str,   default="best_model.npy",
                        help="Path to saved model weights (relative path)")
    return parser.parse_args()


def load_model(model_path):
    data = np.load(model_path, allow_pickle=True).item()
    return data


def evaluate_model(model, X_test, y_test):
    logits = model.forward(X_test)
    preds  = np.argmax(logits, axis=1)
    return {
        "logits":           logits,
        "loss":             None,
        "accuracy":         float(accuracy_score(y_test, preds)),
        "precision":        float(precision_score(y_test, preds, average="macro", zero_division=0)),
        "recall":           float(recall_score(y_test, preds,    average="macro", zero_division=0)),
        "f1":               float(f1_score(y_test, preds,        average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, preds),
    }


def main():
    args = parse_arguments()
    _, _, X_test, _, _, y_test = load_dataset(args.dataset)
    model = NeuralNetwork(args)
    weights = load_model(args.model_path)
    model.set_weights(weights)
    print(f"Loaded weights from '{args.model_path}'")
    results = evaluate_model(model, X_test, y_test)
    print("\n── Evaluation Results ─────────────────────────────────")
    print(f"  Accuracy  : {results['accuracy']:.4f}")
    print(f"  Precision : {results['precision']:.4f}  (macro)")
    print(f"  Recall    : {results['recall']:.4f}  (macro)")
    print(f"  F1-Score  : {results['f1']:.4f}  (macro)")
    print("\nConfusion Matrix:")
    print(results["confusion_matrix"])
    print("\nEvaluation complete!")
    return results

if __name__ == "__main__":
    main()
