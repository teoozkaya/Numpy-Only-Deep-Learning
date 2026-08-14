from typing import Dict

import numpy as np

class Sequential:

    def __init__(self, layers):
        self.layers = layers

    @property
    def params(self) -> Dict[str, np.ndarray]:
        out = {}
        for i, l in enumerate(self.layers):
            for name, p in getattr(l, "params", {}).items():
                out[f"{i}.{name}"] = p
        return out


    @property
    def grads(self):
        out = {}
        for i, l in enumerate(self.layers):
            for name, g in getattr(l, "grads", {}).items():
                out[f"{i}.{name}"] = g
        return out
    
    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)

        return x

    def backward(self, dout):
        for l in self.layers[::-1]:
            dout = l.backward(dout)

        return dout
