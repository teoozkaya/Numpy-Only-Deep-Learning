class SGD:

    def __init__(self, lr=1e-2):
        self.lr = lr


    def step(self, params, grads):
        for k in params:
            params[k] -= grads[k] * self.lr
    