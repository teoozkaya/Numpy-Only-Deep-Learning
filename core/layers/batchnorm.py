
from typing import Dict, Optional, Tuple

import numpy as np

class BatchNorm:

    def __init__(self, num_features=None, eps=1e-5, momentum=0.9):
        self.eps = eps
        self.training = True
        self.momentum = momentum

        self.params: Dict[str, np.ndarray] = {}
        self.grads: Dict[str, np.ndarray] = {}
        self.cache = None

        if num_features is not None:
            self._init_parameters(num_features)

    def _init_parameters(self, d):
        self.params["gamma"] = np.ones(d)
        self.params["beta"] = np.zeros(d)
        self.running_mean = np.zeros(d)
        self.running_var = np.ones(d)

    def forward(self, x):
        if "gamma" not in self.params:
            self._init_parameters(x.shape[1])
        gamma, beta = self.params["gamma"], self.params["beta"]

        if self.training:
            mu = x.mean(axis=0)
            xc = x - mu
            var = (xc ** 2).mean(axis=0)
            std = np.sqrt(var + self.eps)
            xhat = xc / std

            m = x.shape[0]
            unbiased = var * m / (m - 1) if m > 1 else var
            self.running_mean = (self.momentum * self.running_mean + (1 - self.momentum) * mu)
            self.running_var = (self.momentum * self.running_var + (1 - self.momentum) * unbiased)

            self.cache = (xhat, std, gamma)
        else:
            xhat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            self.cache = None

        return gamma * xhat + beta


    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.cache is None:
            raise RuntimeError("backward() requires a forward() pass in train mode")
        xhat, std, gamma = self.cache
        n = dout.shape[0]

        self.grads["beta"] = dout.sum(axis=0)
        self.grads["gamma"] = (dout * xhat).sum(axis=0)

        dxhat = dout * gamma
        dx = (dxhat - dxhat.mean(axis=0) - xhat * (dxhat * xhat).mean(axis=0)) / std
        return dx

    def train(self):
        self.training = True

    def eval(self):
        self.training = False
