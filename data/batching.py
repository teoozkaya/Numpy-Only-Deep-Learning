"""Mini-batch iteration and label helpers."""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np


class DataLoader:
    """Iterate over (X, y) in mini-batches, batch dimension first.

    Args:
        X: inputs of shape (N, ...).
        y: labels of shape (N,).
        batch_size: number of samples per batch.
        shuffle: reshuffle the order at the start of every epoch.
        drop_last: drop the final batch when it is smaller than ``batch_size``.
        seed: seed of the internal shuffling generator.

    Yields:
        (X_batch, y_batch) with shapes (B, ...) and (B,).
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 64,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "X and y disagree on the batch dimension: {} vs {}".format(
                    X.shape[0], y.shape[0]
                )
            )
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1, got {}".format(batch_size))

        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        """Number of batches per epoch."""
        n = self.X.shape[0]
        if self.drop_last:
            return n // self.batch_size
        return -(-n // self.batch_size)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n = self.X.shape[0]
        order = self.rng.permutation(n) if self.shuffle else np.arange(n)
        for start in range(0, n, self.batch_size):
            idx = order[start : start + self.batch_size]
            if self.drop_last and idx.shape[0] < self.batch_size:
                break
            yield self.X[idx], self.y[idx]


def one_hot(y: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray:
    """Encode integer labels as one-hot rows.

    Args:
        y: labels of shape (N,) with values in [0, num_classes).
        num_classes: width of the encoding, inferred from ``y`` when None.

    Returns:
        Array of shape (N, num_classes), float64, one 1.0 per row.
    """
    y = np.asarray(y).ravel()
    if num_classes is None:
        num_classes = int(y.max()) + 1
    encoded = np.zeros((y.shape[0], num_classes))
    encoded[np.arange(y.shape[0]), y] = 1.0
    return encoded


def train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    val_fraction: float = 0.1,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split (X, y) into a training and a validation part.

    Args:
        X: inputs of shape (N, ...).
        y: labels of shape (N,).
        val_fraction: share of the data held out for validation, in [0, 1).

    Returns:
        (X_train, y_train, X_val, y_val).
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1), got {}".format(val_fraction))

    n = X.shape[0]
    n_val = int(round(n * val_fraction))
    perm = np.random.default_rng(seed).permutation(n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]
