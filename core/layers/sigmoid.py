"""Sigmoid activation layer"""

import numpy as np

from typing import Dict, Optional, Tuple

class Sigmoid:

    def __init__(self):
        self.params: Dict[str, np.ndarray] = {}
        self.grads: Dict[str, np.ndarray] = {}
        self.cache: Optional[Tuple[np.ndarray, ...]] = None

    def forward(self, x):
        self.cache = 1 / (1 + np.exp(x))
        return self.cache
    
    def backward(self, dout):
        grad = self.cache*(1 - self.cache)
        return dout * grad