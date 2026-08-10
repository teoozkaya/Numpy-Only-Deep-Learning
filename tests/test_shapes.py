"""Shape and caching contracts from the project conventions."""

from __future__ import annotations

import numpy as np
import pytest

from tests import registry
from tests.conftest import require_layer, require_loss


@pytest.mark.parametrize("name", sorted(registry.LAYERS))
def test_backward_returns_input_shape(name: str, rng: np.random.Generator) -> None:
    """dx must have exactly the shape of the forward input."""
    layer = require_layer(name)
    shape = (registry.BATCH,) + registry.LAYERS[name][3]
    x = rng.standard_normal(shape)

    out = layer.forward(x)
    dx = layer.backward(np.ones(np.shape(out)))

    assert np.shape(dx) == x.shape, (
        "{}: forward got {} and backward returned {}".format(name, x.shape, np.shape(dx))
    )


@pytest.mark.parametrize("name", sorted(registry.LAYERS))
def test_param_grads_match_param_shapes(name: str, rng: np.random.Generator) -> None:
    """Every entry of grads has the shape of the parameter it belongs to."""
    layer = require_layer(name)
    params = getattr(layer, "params", None)
    if not isinstance(params, dict) or not params:
        pytest.skip("{} has no parameters".format(name))

    shape = (registry.BATCH,) + registry.LAYERS[name][3]
    out = layer.forward(rng.standard_normal(shape))
    layer.backward(np.ones(np.shape(out)))

    grads = getattr(layer, "grads", None)
    assert isinstance(grads, dict), "{}: backward did not fill a grads dict".format(name)
    for key, param in params.items():
        assert key in grads, "{}: no gradient for parameter {!r}".format(name, key)
        assert np.shape(grads[key]) == np.shape(param), (
            "{}: grad[{!r}] has shape {} but the parameter has shape {}".format(
                name, key, np.shape(grads[key]), np.shape(param)
            )
        )


@pytest.mark.parametrize("name", sorted(registry.LAYERS))
def test_batch_dimension_is_first(name: str, rng: np.random.Generator) -> None:
    """Forward preserves N, and a batch of one is not a special case."""
    layer = require_layer(name)
    feature_shape = registry.LAYERS[name][3]

    for n in (1, registry.BATCH):
        out = layer.forward(rng.standard_normal((n,) + feature_shape))
        assert np.shape(out)[0] == n, (
            "{}: input batch {} produced output batch {}".format(
                name, n, np.shape(out)[0]
            )
        )


@pytest.mark.parametrize("name", sorted(registry.LAYERS))
def test_cache_is_cleared_on_each_forward(name: str, rng: np.random.Generator) -> None:
    """Two forwards in a row must not leave the first one's cache behind.

    Convention: layers cache what backward needs in `self.cache`, cleared on
    every forward call. If the second forward appends to a stale cache instead
    of replacing it, backward silently uses the wrong activations.
    """
    layer = require_layer(name)
    shape = (registry.BATCH,) + registry.LAYERS[name][3]
    if not hasattr(layer, "cache"):
        pytest.skip("{} does not expose a cache".format(name))

    layer.forward(rng.standard_normal(shape))
    first = layer.cache
    first_size = len(first) if hasattr(first, "__len__") else None

    layer.forward(rng.standard_normal(shape))
    second = layer.cache
    if first_size is not None and hasattr(second, "__len__"):
        assert len(second) == first_size, (
            "{}: cache grew from {} to {} entries across two forwards".format(
                name, first_size, len(second)
            )
        )


@pytest.mark.parametrize("name", sorted(registry.LOSSES))
def test_loss_returns_scalar_and_dx(name: str, rng: np.random.Generator) -> None:
    """Losses return (loss, dx) with dx shaped like the prediction."""
    from tests.test_gradients import _loss_inputs

    loss_fn = require_loss(name)
    prediction, target = _loss_inputs(name, rng)

    result = loss_fn(prediction, target)
    assert isinstance(result, tuple) and len(result) == 2, (
        "{}: expected a (loss, dx) tuple, got {!r}".format(name, type(result))
    )

    loss, dx = result
    assert np.ndim(loss) == 0, "{}: loss must be a scalar".format(name)
    assert np.shape(dx) == np.shape(prediction), (
        "{}: dx has shape {}, prediction has shape {}".format(
            name, np.shape(dx), np.shape(prediction)
        )
    )
