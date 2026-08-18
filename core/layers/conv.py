"""2D convolution layer"""

import numpy as np
from typing import Dict, Optional, Tuple


class Conv2d:

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 0) -> None:
        self.k = kernel_size
        self.stride = stride
        self.padding = padding

        fan_in = in_channels * kernel_size * kernel_size
        self.params: Dict[str, np.ndarray] = {
            "W": np.random.randn(out_channels, in_channels,
                                 kernel_size, kernel_size) * np.sqrt(2.0 / fan_in),
            "b": np.zeros(out_channels),
        }
        self.grads: Dict[str, np.ndarray] = {}
        self.cache = None

    def _window_index(self, H_out: int, W_out: int) -> Tuple[np.ndarray, np.ndarray]:
        k, s = self.k, self.stride
        rows = (np.arange(H_out) * s)[:, None, None, None] + np.arange(k)[None, None, :, None]
        cols = (np.arange(W_out) * s)[None, :, None, None] + np.arange(k)[None, None, None, :]
        return rows, cols

    def forward(self, x: np.ndarray) -> np.ndarray:

        W_, b = self.params["W"], self.params["b"]
        k, s, p = self.k, self.stride, self.padding
        N, C_in, H, W = x.shape

        xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p))) if p else x
        H_out = (H + 2 * p - k) // s + 1
        W_out = (W + 2 * p - k) // s + 1

        windows = np.lib.stride_tricks.sliding_window_view(xp, (k, k), axis=(2, 3))
        windows = windows[:, :, ::s, ::s]

        out = np.einsum("nchwij,ocij->nohw", windows, W_, optimize=True)
        out += b[None, :, None, None]

        self.cache = (x.shape, xp.shape, windows)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:

        (N, C_in, H, W), xp_shape, windows = self.cache
        W_ = self.params["W"]
        p = self.padding
        H_out, W_out = dout.shape[2], dout.shape[3]

        self.grads["b"] = dout.sum(axis=(0, 2, 3))
        self.grads["W"] = np.einsum("nchwij,nohw->ocij", windows, dout, optimize=True)

        dwin = np.einsum("nohw,ocij->nchwij", dout, W_, optimize=True)

        dxp = np.zeros(xp_shape, dtype=dout.dtype)
        rows, cols = self._window_index(H_out, W_out)
        np.add.at(dxp, (slice(None), slice(None), rows, cols), dwin)

        return dxp[:, :, p:p + H, p:p + W] if p else dxp