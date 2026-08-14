
import numpy as np

class Momentum:

    def __init__(self, lr=1e-2, beta=0.9):
        self.lr = lr
        self.m = None

        self.beta = beta

    def step(self, params, grads):
        if self.m is None:
            self.m = {k: np.zeros_like(p) for k, p in params.items()}
        
        for k, p in params.items():
            g = grads[k]

            self.m[k] = self.beta * self.m[k] - self.lr * g

            p += self.m[k]