
import numpy as np

class RMSProp:

    def __init__(self, lr=1e-2, beta=0.99, eps=1e-8):
        self.lr = lr
        self.v = None
        
        self.eps = eps
        self.beta = beta

    def step(self, params, grads):
        if self.v is None:
            self.v = {k: np.zeros_like(p) for k, p in params.items()}
        
        for k, p in params.items():
            g = grads[k]

            self.v[k] = self.beta * self.v[k] + (1 - self.beta) * (g**2)

            p -=  self.lr * g / (np.sqrt(self.v[k]) + self.eps)