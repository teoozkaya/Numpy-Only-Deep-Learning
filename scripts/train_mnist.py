"""MNIST baseline: 784 -> hidden -> 10 with a softmax cross-entropy loss.

This script is wiring only. It expects the following from `core`, which is
where you write the actual maths:

    core.model.Sequential(layers)   .forward(x), .backward(dout),
                                    .params -> dict, .grads -> dict
    core.layers.affine.Affine(in_features, out_features)
    core.layers.relu.ReLU()
    core.losses.softmax_cross_entropy.softmax_cross_entropy(scores, labels)
        -> (loss, dscores)
    core.optim.adam.Adam(lr=...)    .step(params, grads)

If you name any of those differently, fix the import block below (and the
matching entry in tests/registry.py).

Usage:
    python scripts/train_mnist.py --epochs 10 --batch-size 128 --lr 1e-3
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.batching import DataLoader  # noqa: E402
from data.mnist import load_mnist  # noqa: E402


def import_core():
    """Import the pieces of `core` this script needs, with a clear error."""
    missing: List[str] = []
    try:
        from core.model import Sequential
    except ImportError:
        Sequential = None
        missing.append("core.model.Sequential")
    try:
        from core.layers.affine import Affine
    except ImportError:
        Affine = None
        missing.append("core.layers.affine.Affine")
    try:
        from core.layers.relu import ReLU
    except ImportError:
        ReLU = None
        missing.append("core.layers.relu.ReLU")
    try:
        from core.losses.softmax_cross_entropy import softmax_cross_entropy
    except ImportError:
        softmax_cross_entropy = None
        missing.append("core.losses.softmax_cross_entropy.softmax_cross_entropy")
    try:
        from core.optim.adam import Adam
    except ImportError:
        Adam = None
        missing.append("core.optim.adam.Adam")

    if missing:
        raise SystemExit(
            "not implemented yet:\n  " + "\n  ".join(missing) + "\n"
            "Write these in core/, then run this script again."
        )
    return Sequential, Affine, ReLU, softmax_cross_entropy, Adam


def accuracy(model, loss_fn, X: np.ndarray, y: np.ndarray, batch_size: int = 512):
    """Mean loss and accuracy over a dataset, evaluated in batches.

    Args:
        X: inputs of shape (N, 784).
        y: labels of shape (N,).

    Returns:
        (mean_loss, accuracy) as floats.
    """
    total_loss, correct = 0.0, 0
    loader = DataLoader(X, y, batch_size=batch_size, shuffle=False)
    for batch_x, batch_y in loader:
        scores = model.forward(batch_x)
        loss, _ = loss_fn(scores, batch_y)
        total_loss += float(loss) * batch_x.shape[0]
        correct += int((np.argmax(scores, axis=1) == batch_y).sum())
    return total_loss / X.shape[0], correct / X.shape[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, nargs="+", default=[256])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="use only the first N training images (quick smoke run)",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="path to save the loss/accuracy curves, e.g. runs/mnist.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Sequential, Affine, ReLU, softmax_cross_entropy, Adam = import_core()

    np.random.seed(args.seed)
    data = load_mnist(flatten=True, normalize=True, seed=args.seed)
    X_train, y_train = data["X_train"], data["y_train"]
    if args.limit is not None:
        X_train, y_train = X_train[: args.limit], y_train[: args.limit]

    sizes = [X_train.shape[1]] + list(args.hidden) + [10]
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(Affine(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(ReLU())

    model = Sequential(layers)
    optimizer = Adam(lr=args.lr)
    loader = DataLoader(
        X_train, y_train, batch_size=args.batch_size, shuffle=True, seed=args.seed
    )

    print("train {}  val {}  architecture {}".format(
        X_train.shape[0], data["X_val"].shape[0], " -> ".join(map(str, sizes))
    ))

    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []
    }

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        running_loss, seen = 0.0, 0

        for batch_x, batch_y in loader:
            scores = model.forward(batch_x)
            loss, dscores = softmax_cross_entropy(scores, batch_y)
            model.backward(dscores)
            optimizer.step(model.params, model.grads)

            running_loss += float(loss) * batch_x.shape[0]
            seen += batch_x.shape[0]

        train_loss = running_loss / seen
        _, train_acc = accuracy(model, softmax_cross_entropy, X_train, y_train)
        val_loss, val_acc = accuracy(
            model, softmax_cross_entropy, data["X_val"], data["y_val"]
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            "epoch {:>3}  train loss {:.4f}  train acc {:.4f}  "
            "val loss {:.4f}  val acc {:.4f}  ({:.1f}s)".format(
                epoch, train_loss, train_acc, val_loss, val_acc, time.time() - start
            )
        )

    test_loss, test_acc = accuracy(
        model, softmax_cross_entropy, data["X_test"], data["y_test"]
    )
    print("test loss {:.4f}  test acc {:.4f}".format(test_loss, test_acc))

    if args.plot is not None:
        from viz.curves import plot_history

        directory = os.path.dirname(args.plot)
        if directory:
            os.makedirs(directory, exist_ok=True)
        plot_history(history, save_path=args.plot)
        print("wrote {}".format(args.plot))


if __name__ == "__main__":
    main()
