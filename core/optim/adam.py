
import numpy as np

class Adam:

    def __init__(self, lr=1e-2, beta1 = 0.9, beta2 = 0.999, eps = 1e-8):
        self.lr = lr
        self.m = None
        self.v = None
        
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = {k: np.zeros_like(p) for k, p in params.items()}
            self.v = {k: np.zeros_like(p) for k, p in params.items()}

        self.t += 1

        for k, p in params.items():
            g = grads[k]

            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (g ** 2)

            m_correction = self.m[k] / (1 - self.beta1 ** self.t)
            v_correction = self.v[k] / (1 - self.beta2 ** self.t)

            p -= self.lr * m_correction / (np.sqrt(v_correction) + self.eps)

        
