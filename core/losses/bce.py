
import numpy as np

def bce(prediction, target):
    
    n = prediction.shape[0]
    eps = 1e-12

    p = np.clip(prediction, eps, 1.0 - eps)

    loss = -np.sum(target * np.log(p) + (1 - target) * np.log(1 - p)) / n
    dx = (p - target) / (p * (1 - p) * n)

    return (loss, dx)