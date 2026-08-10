"""Loss and accuracy curves."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt


def _finish(fig, save_path: Optional[str], show: bool):
    """Save and/or show a figure, then return it."""
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


def plot_loss_curves(
    train_loss: Sequence[float],
    val_loss: Optional[Sequence[float]] = None,
    title: str = "loss",
    xlabel: str = "epoch",
    log_scale: bool = False,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Plot training (and optionally validation) loss against epochs.

    Args:
        train_loss: one value per epoch.
        val_loss: one value per epoch, or None.
        log_scale: use a logarithmic y axis, useful once the loss spans orders
            of magnitude.
        save_path: write the figure here when given.
        show: call plt.show() at the end.

    Returns:
        The matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(train_loss) + 1), train_loss, label="train")
    if val_loss is not None and len(val_loss) > 0:
        ax.plot(range(1, len(val_loss) + 1), val_loss, label="val")
        ax.legend()
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return _finish(fig, save_path, show)


def plot_history(
    history: Dict[str, Sequence[float]],
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Plot a training history dict as loss and accuracy side by side.

    Args:
        history: dict that may contain the keys ``train_loss``, ``val_loss``,
            ``train_acc`` and ``val_acc``, each a sequence with one value per
            epoch.

    Returns:
        The matplotlib Figure.
    """
    has_acc = any(key in history for key in ("train_acc", "val_acc"))
    fig, axes = plt.subplots(1, 2 if has_acc else 1, figsize=(11 if has_acc else 6, 4))
    axes = axes if has_acc else [axes]

    for key, label in (("train_loss", "train"), ("val_loss", "val")):
        if key in history:
            axes[0].plot(range(1, len(history[key]) + 1), history[key], label=label)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    if has_acc:
        for key, label in (("train_acc", "train"), ("val_acc", "val")):
            if key in history:
                axes[1].plot(range(1, len(history[key]) + 1), history[key], label=label)
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("accuracy")
        axes[1].set_title("accuracy")
        axes[1].set_ylim(0.0, 1.0)
        axes[1].grid(alpha=0.3)
        axes[1].legend()

    return _finish(fig, save_path, show)
