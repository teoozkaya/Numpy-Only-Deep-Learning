"""Max pooling layer"""

import numpy as np

from typing import Dict, Optional, Tuple


class MaxPool:
    """Takes the maximum over each spatial window.

    No learnable parameters: the layer only routes values forward and
    gradients back.

    Attributes:
        kernel_size: side length of the square pooling window.
        stride: step between windows. Defaults to kernel_size (non-overlapping).
        cache: what backward needs from the last forward call.
    """

    def __init__(self, kernel_size: int = 2, stride: Optional[int] = None) -> None:
        """
        Args:
            kernel_size: side length of the square window, k.
            stride: step between windows. None means stride == kernel_size.
        """
        # TODO: store kernel_size and stride (defaulting stride to
        # kernel_size), plus empty params/grads dicts and cache = None.
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride

        self.params: Dict[str, np.ndarray] = {}
        self.grads: Dict[str, np.ndarray] = {}
        self.cache = None



    def forward(self, x: np.ndarray) -> np.ndarray:
        N, C, H, W = x.shape
        k, s = self.kernel_size, self.stride
        H_out = (H - k) // s + 1
        W_out = (W - k) // s + 1

        windows = np.lib.stride_tricks.sliding_window_view(x, (k, k), axis=(2, 3))
        windows = windows[:, :, ::s, ::s]

        flat = windows.reshape(N, C, H_out, W_out, k * k)
        idx = flat.argmax(axis=-1)
        out = np.take_along_axis(flat, idx[..., None], axis=-1).squeeze(-1)

        self.cache = (x.shape, idx)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        (N, C, H, W), idx = self.cache
        k, s = self.kernel_size, self.stride
        H_out, W_out = dout.shape[2], dout.shape[3]

        dx = np.zeros((N, C, H, W), dtype=dout.dtype)

        di, dj = np.divmod(idx, k)

        rows = np.arange(H_out)[None, None, :, None] * s + di
        cols = np.arange(W_out)[None, None, None, :] * s + dj

        n = np.arange(N)[:, None, None, None]
        c = np.arange(C)[None, :, None, None]

        np.add.at(dx, (n, c, rows, cols), dout)
        return dx
