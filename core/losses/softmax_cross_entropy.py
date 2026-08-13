"""Softmax cross-entropy loss"""

import numpy as np

from typing import Tuple


def softmax_cross_entropy(logits: np.ndarray, target: np.ndarray) -> Tuple[float, np.ndarray]:
    """Softmax followed by cross-entropy, fused into one function.

    Args:
        logits: raw unnormalised class scores of shape (N, C). Not probabilities
            -- the softmax happens in here.
        target: correct class index per sample, shape (N,), integers in [0, C).

    Returns:
        loss: scalar (0-dimensional), averaged over the batch.
        dx: gradient of the loss with respect to logits, shape (N, C).
    """
    n = logits.shape[0]
    rows = np.arange(n)

    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    log_denom = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    log_probs = shifted - log_denom

    loss = -np.mean(log_probs[rows, target])

    dx = np.exp(log_probs)
    dx[rows, target] -= 1.0
    dx /= n

    return (loss, dx)
