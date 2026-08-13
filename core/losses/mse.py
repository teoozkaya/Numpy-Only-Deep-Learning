"""Mean squared error loss"""

import numpy as np

from typing import Tuple


def mse(prediction: np.ndarray, target: np.ndarray) -> Tuple[float, np.ndarray]:
    """Mean squared error between a prediction and a target.

    Args:
        prediction: model output of shape (N, D).
        target: ground truth of shape (N, D), same shape as prediction.

    Returns:
        loss: scalar (0-dimensional).
        dx: gradient of the loss with respect to prediction, shape (N, D).
    """
    n = prediction.shape[0]
    diff = prediction - target
    loss = np.sum(diff**2) / n

    dx = 2 * diff / n
    return (loss, dx)
