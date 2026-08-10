"""Decision boundaries for two-dimensional toy datasets."""

from __future__ import annotations

from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np

from viz.curves import _finish


def plot_decision_boundary(
    predict: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    resolution: int = 200,
    margin: float = 0.5,
    title: str = "decision boundary",
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Shade the predicted class over a grid and scatter the data on top.

    Args:
        predict: maps a batch of shape (N, 2) to labels (N,) or scores
            (N, C); scores are reduced with argmax.
        X: points of shape (N, 2).
        y: labels of shape (N,).
        resolution: number of grid steps per axis.
        margin: padding added around the data range.

    Returns:
        The matplotlib Figure.
    """
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("X must have shape (N, 2), got {}".format(X.shape))

    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()]).astype(X.dtype)

    predictions = np.asarray(predict(grid))
    if predictions.ndim == 2 and predictions.shape[1] > 1:
        predictions = predictions.argmax(axis=1)
    predictions = predictions.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(xx, yy, predictions, alpha=0.25, cmap="coolwarm")
    ax.scatter(X[:, 0], X[:, 1], c=y, s=18, cmap="coolwarm", edgecolors="k", linewidths=0.3)
    ax.set_title(title)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    return _finish(fig, save_path, show)
