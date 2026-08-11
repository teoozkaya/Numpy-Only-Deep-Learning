""" Rectified Linear Unit - ReLU """

from typing import Dict, Optional, Tuple

import numpy as np

class ReLU:

    def __init__(self):
        self.params: Dict[str, np.ndarray] = {}
        self.grads: Dict[str, np.ndarray] = {}
        self.cache: Optional[Tuple[np.ndarray, ...]] = None
    
    def forward(self, x):
        self.cache = (x, )
        return np.maximum(0, x)

    def backward(self, dout):
        (x,) = self.cache
        boolean_x = x > 0
        dx = dout * boolean_x

        return dx