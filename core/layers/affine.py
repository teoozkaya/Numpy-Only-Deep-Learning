"""Fully connected (affine) layer."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


class Affine:

    def __init__(self, in_features, out_features, seed: int = None):
        rng = np.random.default_rng(seed)

        W = rng.standard_normal((in_features, out_features)) * np.sqrt(2.0 / in_features)
        b = np.zeros(out_features)
        
        self.params: Dict[str, np.ndarray] = {"W": W, "b": b}
        self.grads: Dict[str, np.ndarray] = {}
        self.cache: Optional[Tuple[np.ndarray, ...]] = None

    def forward(self, x):
        self.cache = (x,)

        out = np.dot(x, self.params["W"]) + self.params["b"]
        return out
    

    def backward(self, dout):
        (x,) = self.cache
        W = self.params["W"]

        self.grads["W"] = np.dot(x.T, dout)
        self.grads["b"] = dout.sum(axis=0)

        dx = np.dot(dout, W.T)

        return dx