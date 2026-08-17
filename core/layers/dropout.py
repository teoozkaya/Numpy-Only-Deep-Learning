"""Inverted dropout layer"""

import numpy as np

from typing import Dict, Optional, Tuple


class Dropout:
    """Zeroes a random fraction of the activations while training.

    At evaluation time the layer is a no-op: inverted dropout does the
    rescaling during training instead, so nothing has to happen at inference.

    Attributes:
        p: probability of dropping any single activation, in [0, 1).
        training: when False, forward returns its input unchanged.
        cache: what backward needs from the last forward call.
    """

    def __init__(self, p: float = 0.5, seed: int = None) -> None:
        self.p = p
        self.rng = np.random.default_rng(seed)
        self.training = True
        self.params: Dict[str, np.ndarray] = {}
        self.grads: Dict[str, np.ndarray] = {}
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply the dropout mask.

        Args:
            x: input of shape (N, ...).

        Returns:
            Output of the same shape as x. Unchanged when self.training is
            False.
        """
        if not self.training or self.p == 0.0:
            self.cache = None
            return x
        
        mask = (self.rng.random(x.shape) >= self.p) / (1.0 - self.p)
        self.cache = mask
        return x * mask

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.cache is None:
            return dout
        
        return dout * self.cache

    def train(self) -> None:
        """Switch to training mode: the mask is applied."""
        self.training = True

    def eval(self) -> None:
        """Switch to evaluation mode: forward becomes a no-op."""
        self.training = False

