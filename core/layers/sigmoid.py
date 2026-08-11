"""Sigmoid activation layer"""

import numpy as np

from typing import Dict, Optional, Tuple

class Sigmoid:

    def __init__(self):
        self.params: Dict[str, np.ndarray] = {}
        self.grads: Dict[str, np.ndarray] = {}
        self.cache: Optional[Tuple[np.ndarray, ...]] = None

    def forward(self, x):
        sigmo = 1 / (1 + np.exp(-x))
        self.cache = (sigmo, )
        return sigmo
    
    def backward(self, dout):
        (out, ) = self.cache
        grad = out * (1 - out)
        return dout * grad