"""Weight histograms and first-layer filter grids."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from viz.curves import _finish


def plot_weight_histogram(
    params: Dict[str, np.ndarray],
    bins: int = 50,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Histogram every parameter array, one subplot each.

    Dead or exploding layers show up here first: a spike at zero means dead
    units, a spread that keeps widening across epochs means the updates are
    too large.

    Args:
        params: mapping from parameter name to array of any shape.

    Returns:
        The matplotlib Figure.
    """
    if not params:
        raise ValueError("params is empty, nothing to plot")

    names = sorted(params)
    cols = min(3, len(names))
    rows = -(-len(names) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)

    for ax, name in zip(axes.ravel(), names):
        values = np.asarray(params[name]).ravel()
        ax.hist(values, bins=bins)
        ax.set_title("{}  std={:.3f}".format(name, values.std()))
        ax.grid(alpha=0.3)

    for ax in axes.ravel()[len(names) :]:
        ax.axis("off")

    return _finish(fig, save_path, show)


def plot_filters(
    W: np.ndarray,
    image_shape: Tuple[int, int] = (28, 28),
    max_filters: int = 64,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Show the rows or channels of a weight matrix as images.

    Accepts either an affine weight of shape (D, M) -- each of the M columns is
    reshaped to ``image_shape`` -- or a conv weight of shape (F, C, H, W), in
    which case the first input channel of each filter is shown.

    Args:
        W: weight array, shape (D, M) or (F, C, H, W).
        image_shape: (H, W) used to reshape affine columns.
        max_filters: cap on how many filters are drawn.

    Returns:
        The matplotlib Figure.
    """
    W = np.asarray(W)
    if W.ndim == 2:
        count = min(W.shape[1], max_filters)
        images = [W[:, i].reshape(image_shape) for i in range(count)]
    elif W.ndim == 4:
        count = min(W.shape[0], max_filters)
        images = [W[i, 0] for i in range(count)]
    else:
        raise ValueError("expected a 2D or 4D weight array, got shape {}".format(W.shape))

    cols = int(np.ceil(np.sqrt(count)))
    rows = -(-count // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows), squeeze=False)

    for ax, image in zip(axes.ravel(), images):
        ax.imshow(image, cmap="gray", interpolation="nearest")
        ax.axis("off")
    for ax in axes.ravel()[count:]:
        ax.axis("off")

    return _finish(fig, save_path, show)
