"""Single place where the tests learn how to build things out of `core`.

Nothing in `core` exists yet, so every entry here is a guess at the name and
constructor signature. When a name does not match what you wrote, fix it here
and the whole suite follows -- no test file needs to change.

Each entry is (import path, attribute name, factory, input shape). The factory
receives no arguments and returns a fresh instance. A ``None`` factory means
"call the attribute directly" (used for the losses).
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, Optional, Tuple

BATCH = 4

# name -> (module, attribute, build(cls) -> instance, input shape without N)
LAYERS: Dict[str, Tuple[str, str, Callable[[Any], Any], Tuple[int, ...]]] = {
    "affine": ("core.layers.affine", "Affine", lambda cls: cls(6, 5), (6,)),
    "relu": ("core.layers.relu", "ReLU", lambda cls: cls(), (6,)),
    "sigmoid": ("core.layers.sigmoid", "Sigmoid", lambda cls: cls(), (6,)),
    "tanh": ("core.layers.tanh", "Tanh", lambda cls: cls(), (6,)),
    "batchnorm": ("core.layers.batchnorm", "BatchNorm", lambda cls: cls(6), (6,)),
    "conv": ("core.layers.conv", "Conv2d", lambda cls: cls(2, 3, 3), (2, 6, 6)),
    "pool": ("core.layers.pool", "MaxPool", lambda cls: cls(2, 2), (2, 6, 6)),
}

# Dropout is excluded from the gradient checks above: its mask is random, so a
# numerical check only makes sense once you can pin the mask (fixed seed, or
# p=0.0). Add it here when you decide how to expose that.

# name -> (module, attribute)
LOSSES: Dict[str, Tuple[str, str]] = {
    "mse": ("core.losses.mse", "mse"),
    "bce": ("core.losses.bce", "bce"),
    "softmax_cross_entropy": (
        "core.losses.softmax_cross_entropy",
        "softmax_cross_entropy",
    ),
}

# name -> (module, attribute, build(cls) -> instance)
OPTIMIZERS: Dict[str, Tuple[str, str, Callable[[Any], Any]]] = {
    "sgd": ("core.optim.sgd", "SGD", lambda cls: cls(lr=1e-2)),
    "momentum": ("core.optim.momentum", "Momentum", lambda cls: cls(lr=1e-2)),
    "rmsprop": ("core.optim.rmsprop", "RMSProp", lambda cls: cls(lr=1e-2)),
    "adam": ("core.optim.adam", "Adam", lambda cls: cls(lr=1e-2)),
}

MODEL = ("core.model", "Sequential")


def resolve(module_path: str, attribute: str) -> Optional[Any]:
    """Import ``module_path`` and return ``attribute``, or None if missing."""
    try:
        module = importlib.import_module(module_path)
    except ImportError:
        return None
    return getattr(module, attribute, None)
