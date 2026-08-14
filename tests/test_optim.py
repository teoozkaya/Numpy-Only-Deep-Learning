"""Optimizer contracts: descend a quadratic and keep per-parameter state."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pytest

from tests import registry
from tests.conftest import require_optimizer


def _step(optimizer, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
    """Take one step, tolerating both in-place and returning `step` styles."""
    returned = optimizer.step(params, grads)
    return params if returned is None else returned


@pytest.mark.parametrize("name", sorted(registry.OPTIMIZERS))
def test_minimizes_a_quadratic(name: str) -> None:
    """f(w) = 0.5 * ||w||^2 must fall monotonically from a fixed start."""
    optimizer = require_optimizer(name)
    params = {"w": np.array([3.0, -4.0, 0.5])}

    losses = []
    for _ in range(1000):
        losses.append(0.5 * float(np.sum(params["w"] ** 2)))
        params = _step(optimizer, params, {"w": params["w"].copy()})

    final = 0.5 * float(np.sum(params["w"] ** 2))
    assert final < losses[0], "{}: loss did not decrease ({:.4f} -> {:.4f})".format(
        name, losses[0], final
    )
    assert final < 1e-2, "{}: did not converge, final loss {:.4f}".format(name, final)


@pytest.mark.parametrize("name", sorted(registry.OPTIMIZERS))
def test_keeps_state_per_parameter(name: str) -> None:
    """Two parameters with different gradients must not share state.

    A momentum/RMSProp/Adam buffer keyed by anything other than the parameter
    name will leak one parameter's history into the other, and the symptom is
    exactly this: identical updates for different gradients.
    """
    optimizer = require_optimizer(name)
    params = {"a": np.zeros(3), "b": np.zeros(3)}
    grads = {"a": np.full(3, 1.0), "b": np.full(3, -2.0)}

    for _ in range(3):
        params = _step(optimizer, params, {k: v.copy() for k, v in grads.items()})

    assert not np.allclose(params["a"], params["b"]), (
        "{}: parameters with opposite gradients received identical updates".format(name)
    )
    assert np.all(params["a"] < 0.0), "{}: 'a' moved against its gradient".format(name)
    assert np.all(params["b"] > 0.0), "{}: 'b' moved against its gradient".format(name)


@pytest.mark.parametrize("name", sorted(registry.OPTIMIZERS))
def test_zero_gradient_leaves_params_untouched(name: str) -> None:
    """With a zero gradient and no prior history, nothing should move."""
    optimizer = require_optimizer(name)
    params = {"w": np.array([1.0, 2.0, 3.0])}
    before = params["w"].copy()

    params = _step(optimizer, params, {"w": np.zeros(3)})

    assert np.allclose(params["w"], before), (
        "{}: a zero gradient still changed the parameters".format(name)
    )
