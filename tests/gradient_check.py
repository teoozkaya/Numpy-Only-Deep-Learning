"""Numerical gradient checking utilities.

Central differences everywhere: the forward-difference error is O(h) while the
central-difference error is O(h^2), which is what makes 1e-7 relative errors
achievable in float64.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np


def rel_error(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Max relative error between two arrays.

    Args:
        a: array of any shape.
        b: array broadcastable to the shape of ``a``.

    Returns:
        max |a - b| / max(eps, |a| + |b|) over all entries.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("shape mismatch: {} vs {}".format(a.shape, b.shape))
    denom = np.maximum(eps, np.abs(a) + np.abs(b))
    return float(np.max(np.abs(a - b) / denom))


def numerical_gradient(
    f: Callable[[np.ndarray], float], x: np.ndarray, h: float = 1e-5
) -> np.ndarray:
    """Gradient of a scalar function ``f`` at ``x`` by central differences.

    Args:
        f: maps an array of shape (...) to a scalar.
        x: point of evaluation, shape (...). Modified in place and restored.
        h: step size.

    Returns:
        Array of shape ``x.shape`` holding df/dx.
    """
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        original = x[idx]

        x[idx] = original + h
        fx_plus = f(x)
        x[idx] = original - h
        fx_minus = f(x)
        x[idx] = original

        grad[idx] = (fx_plus - fx_minus) / (2.0 * h)
        it.iternext()
    return grad


def numerical_gradient_array(
    f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    dout: np.ndarray,
    h: float = 1e-5,
) -> np.ndarray:
    """Backprop-style numerical gradient of an array-valued function.

    Computes ``d(sum(f(x) * dout))/dx``, which is exactly what a correct
    ``backward(dout)`` must return.

    Args:
        f: maps an array of shape (...) to an array of shape (...).
        x: point of evaluation. Modified in place and restored.
        dout: upstream gradient, same shape as ``f(x)``.
        h: step size.

    Returns:
        Array of shape ``x.shape``.
    """
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        original = x[idx]

        x[idx] = original + h
        pos = np.array(f(x), dtype=np.float64, copy=True)
        x[idx] = original - h
        neg = np.array(f(x), dtype=np.float64, copy=True)
        x[idx] = original

        grad[idx] = np.sum((pos - neg) * dout) / (2.0 * h)
        it.iternext()
    return grad


def check_layer(
    layer,
    x: np.ndarray,
    dout: Optional[np.ndarray] = None,
    h: float = 1e-5,
    seed: int = 0,
) -> Dict[str, float]:
    """Compare a layer's analytic gradients against numerical ones.

    The input gradient is always checked. Parameter gradients are checked when
    the layer exposes ``params`` and ``grads`` dicts with matching keys.

    Args:
        layer: object with ``forward(x)`` and ``backward(dout)``.
        x: input of shape (N, ...), ideally float64 for a tight tolerance.
        dout: upstream gradient with the shape of ``forward(x)``; random when
            None.
        h: central-difference step size.

    Returns:
        Dict mapping ``"dx"`` and each parameter name to its relative error.
    """
    rng = np.random.default_rng(seed)
    x = np.array(x, dtype=np.float64, copy=True)

    out = layer.forward(x)
    if dout is None:
        dout = rng.standard_normal(np.shape(out))
    dout = np.asarray(dout, dtype=np.float64)

    if np.shape(out) != dout.shape:
        raise ValueError(
            "dout shape {} does not match forward output shape {}".format(
                dout.shape, np.shape(out)
            )
        )

    dx = layer.backward(dout)
    if np.shape(dx) != x.shape:
        raise ValueError(
            "backward returned dx of shape {} but the input has shape {}".format(
                np.shape(dx), x.shape
            )
        )

    errors = {"dx": rel_error(numerical_gradient_array(layer.forward, x, dout, h), dx)}

    params = getattr(layer, "params", None)
    grads = getattr(layer, "grads", None)
    if isinstance(params, dict) and isinstance(grads, dict):
        for name, value in params.items():
            if name not in grads:
                continue

            # `value` is the very array the layer reads from, so perturbing it
            # in place is what makes the numerical parameter gradient work.
            def forward_wrt_param(_perturbed_param):
                return layer.forward(x)

            numeric = numerical_gradient_array(forward_wrt_param, value, dout, h)
            errors[name] = rel_error(numeric, grads[name])

    return errors
