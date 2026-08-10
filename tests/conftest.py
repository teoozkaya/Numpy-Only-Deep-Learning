"""Shared fixtures and skip helpers.

Everything in `core` is unwritten, so the suite is built to skip cleanly rather
than error: a test for a layer you have not implemented yet reports as skipped
and turns green the moment the class appears.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests import registry


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded generator, so a failing test fails the same way twice."""
    return np.random.default_rng(0)


def require(module_path: str, attribute: str) -> Any:
    """Return ``module_path.attribute`` or skip the test if it does not exist."""
    obj = registry.resolve(module_path, attribute)
    if obj is None:
        pytest.skip("{}.{} is not implemented yet".format(module_path, attribute))
    return obj


def require_layer(name: str) -> Any:
    """Build the layer registered under ``name``, skipping if it is missing."""
    module_path, attribute, factory, _shape = registry.LAYERS[name]
    return factory(require(module_path, attribute))


def require_loss(name: str) -> Any:
    """Return the loss callable registered under ``name``."""
    module_path, attribute = registry.LOSSES[name]
    return require(module_path, attribute)


def require_optimizer(name: str) -> Any:
    """Build the optimizer registered under ``name``."""
    module_path, attribute, factory = registry.OPTIMIZERS[name]
    return factory(require(module_path, attribute))
