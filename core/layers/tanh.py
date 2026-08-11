"""Tanh activation layer"""

import numpy as np

from typing import Dict, Optional, Tuple


class Tanh:

    def __init__(self):
        self.params: Dict[str, np.ndarray] = {}
        self.grads: Dict[str, np.ndarray] = {}
        self.cache: Optional[Tuple[np.ndarray, ...]] = None
 
    def forward(self, x):
        output = np.tanh(x)
        self.cache = (output, )
        return output
    
    def backward(self, dout):
        (out, ) = self.cache
        grad = 1 - out * out
        return dout * grad