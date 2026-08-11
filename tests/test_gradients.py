"""Numerical gradient checks for every layer and loss in `core`.

Run with `pytest tests/test_gradients.py -v` before committing anything in
`core/`. A failure prints the layer and the relative error of the offending
gradient.

One caveat worth knowing before you trust a failure: numerical checks are
unreliable at kinks. ReLU is not differentiable at x = 0, so an input within
one step size h of zero makes the central difference straddle the kink and
report a relative error near 1.0 for perfectly correct code. Inputs here are
standard normal, so this is rare -- but if a ReLU-family layer fails once and
passes with a different seed, that is what happened, not a bug in your
backward pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests import registry
from tests.conftest import require_layer, require_loss
from tests.gradient_check import numerical_gradient, rel_error

TOL = 1e-6


@pytest.mark.parametrize("name", sorted(registry.LAYERS))
def test_layer_gradients(name: str, rng: np.random.Generator) -> None:
    """Analytic backward matches central differences for every layer."""
    from tests.gradient_check import check_layer

    layer = require_layer(name)
    shape = (registry.BATCH,) + registry.LAYERS[name][3]
    x = rng.standard_normal(shape)

    errors = check_layer(layer, x, seed=1234)

    bad = {key: err for key, err in errors.items() if not err < TOL}
    assert not bad, "{}: relative error too large -> {}".format(
        name,
        ", ".join("{}={:.3e}".format(key, err) for key, err in sorted(bad.items())),
    )


def _loss_inputs(name: str, rng: np.random.Generator):
    """Return (prediction, target) suitable for the given loss."""
    n = registry.BATCH
    if name == "mse":
        return rng.standard_normal((n, 3)), rng.standard_normal((n, 3))
    if name == "bce":
        logits = rng.standard_normal((n, 1))
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        targets = rng.integers(0, 2, size=(n, 1)).astype(np.float64)
        return probabilities, targets
    if name == "softmax_cross_entropy":
        return rng.standard_normal((n, 5)), rng.integers(0, 5, size=n)
    raise KeyError("no inputs defined for loss {!r}".format(name))


@pytest.mark.parametrize("name", sorted(registry.LOSSES))
def test_loss_gradients(name: str, rng: np.random.Generator) -> None:
    """dx returned by each loss matches central differences on the loss value."""
    loss_fn = require_loss(name)
    prediction, target = _loss_inputs(name, rng)
    prediction = np.array(prediction, dtype=np.float64)

    loss, dx = loss_fn(prediction, target)

    assert np.isscalar(loss) or np.ndim(loss) == 0, (
        "{}: loss must be a scalar, got shape {}".format(name, np.shape(loss))
    )
    assert np.shape(dx) == prediction.shape, (
        "{}: dx has shape {} but the prediction has shape {}".format(
            name, np.shape(dx), prediction.shape
        )
    )

    numeric = numerical_gradient(lambda p: float(loss_fn(p, target)[0]), prediction)
    error = rel_error(numeric, dx)
    assert error < TOL, "{}: relative error {:.3e}".format(name, error)


@pytest.mark.parametrize("name", sorted(registry.LAYERS))
def test_backward_is_linear_in_dout(name: str, rng: np.random.Generator) -> None:
    """backward(a*dout) == a*backward(dout): the chain rule is linear.

    This catches a backward pass that ignores its upstream gradient entirely,
    which a single gradient check with dout = ones would happily accept.
    """
    layer = require_layer(name)
    shape = (registry.BATCH,) + registry.LAYERS[name][3]
    x = rng.standard_normal(shape)

    out = layer.forward(x)
    dout = rng.standard_normal(np.shape(out))

    layer.forward(x)
    dx_once = np.array(layer.backward(dout), copy=True)
    layer.forward(x)
    dx_scaled = np.array(layer.backward(3.0 * dout), copy=True)

    error = rel_error(3.0 * dx_once, dx_scaled)
    assert error < 1e-10, "{}: backward is not linear in dout ({:.3e})".format(
        name, error
    )
