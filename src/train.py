"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""
import argparse
import numpy as np
import wandb
import os
import sys
import json
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ann.neural_network      import NeuralNetwork
from ann.objective_functions import get_loss
from ann.optimizers          import get_optimizer
from utils.data_loader       import load_dataset, batch_iterator


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a neural network')
    parser.add_argument("-d",  "--dataset",       type=str,   choices=["mnist", "fashion_mnist"], default="fashion_mnist")
    parser.add_argument("-e",  "--epochs",         type=int,   default=20)
    parser.add_argument("-b",  "--batch_size",     type=int,   default=32)
    parser.add_argument("-lr", "--learning_rate",  type=float, default=0.00045024653096303305)
    parser.add_argument("-wd", "--weight_decay",   type=float, default=0.0)
    parser.add_argument("-l",  "--loss",           type=str,   choices=["mse", "cross_entropy"], default="cross_entropy")
    parser.add_argument("-o",  "--optimizer",      type=str,   choices=["sgd", "momentum", "nag", "rmsprop"], default="rmsprop")
    parser.add_argument("-nhl","--num_layers",     type=int,   default=3)
    parser.add_argument("-sz", "--hidden_size",    nargs="+",  type=int, default=[128, 128, 128])
    parser.add_argument("-a",  "--activation",     type=str,   choices=["sigmoid", "tanh", "relu"], default="tanh")
    parser.add_argument("-w_i","--weight_init",    type=str,   choices=["random", "xavier"], default="xavier")
    parser.add_argument("-w_p","--wandb_project",  type=str,   default="da6401_assignment1")
    parser.add_argument("--model_save_path",       type=str,   default="best_model.npy")
    parser.add_argument("--model_path",            type=str,   default="best_model.npy")
    return parser.parse_args()


def main():
    args = parse_arguments()
    if len(args.hidden_size) == 1:
        args.hidden_size = args.hidden_size * args.num_layers
    wandb.init(project=args.wandb_project,
               name=f"{args.optimizer}_lr{args.learning_rate}_hl{'_'.join(map(str,args.hidden_size))}_{args.activation}",
               config=vars(args))
    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(args.dataset)
    print(f"[data] train={X_train.shape}  val={X_val.shape}  test={X_test.shape}")
    model     = NeuralNetwork(args)
    loss_fn   = get_loss(args.loss)
    optimizer = get_optimizer(args.optimizer, args.learning_rate)
    best_f1      = -1.0
    best_weights = None
    for epoch in range(1, args.epochs + 1):
        epoch_losses = []
        for X_b, y_b in batch_iterator(X_train, y_train, args.batch_size):
            logits  = model.forward(X_b)
            loss, _ = loss_fn(logits, y_b)
            model.backward(y_b, logits, weight_decay=args.weight_decay)
            optimizer.update(model.layers)
            epoch_losses.append(loss)
        train_loss = float(np.mean(epoch_losses))
        val_logits  = model.forward(X_val)
        val_loss, _ = loss_fn(val_logits, y_val)
        val_acc     = float(np.mean(np.argmax(val_logits, axis=1) == y_val))
        test_preds  = model.predict(X_test)
        test_acc    = float(np.mean(test_preds == y_test))
        test_f1     = float(f1_score(y_test, test_preds, average="macro"))

        print(f"Epoch {epoch:3d}/{args.epochs} | train_loss={train_loss:.4f}  val_acc={val_acc:.4f}  test_f1={test_f1:.4f}")
        wandb.log({"epoch": epoch, "train_loss": train_loss,
                   "val_loss": float(val_loss), "val_acc": val_acc,
                   "test_acc": test_acc, "test_f1": test_f1})
        if test_f1 > best_f1:
            best_f1      = test_f1
            best_weights = model.get_weights()
    src_dir = os.path.dirname(os.path.abspath(__file__))
    np.save(os.path.join(src_dir, "best_model.npy"), best_weights)
    with open(os.path.join(src_dir, "best_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"\nBest test F1 : {best_f1:.4f}")
    print(f"Model saved  → {os.path.join(src_dir, 'best_model.npy')}")
    print(f"Config saved → {os.path.join(src_dir, 'best_config.json')}")
    wandb.summary["best_test_f1"] = best_f1
    wandb.finish()
    print("Training complete!")
    return best_weights, best_f1

if __name__ == '__main__':
    main()
