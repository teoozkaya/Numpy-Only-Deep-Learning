"""Tests for the batching helpers in `data/`."""

from __future__ import annotations

import numpy as np
import pytest

from data.batching import DataLoader, one_hot, train_val_split


def test_loader_covers_every_sample_exactly_once() -> None:
    X = np.arange(10).reshape(10, 1)
    y = np.arange(10)
    loader = DataLoader(X, y, batch_size=3, shuffle=True, seed=0)

    seen = np.concatenate([batch_y for _, batch_y in loader])

    assert sorted(seen.tolist()) == list(range(10))
    assert len(loader) == 4


def test_loader_keeps_x_and_y_aligned() -> None:
    X = np.arange(20).reshape(20, 1)
    y = np.arange(20) * 2
    loader = DataLoader(X, y, batch_size=6, shuffle=True, seed=1)

    for batch_x, batch_y in loader:
        assert np.array_equal(batch_x.ravel() * 2, batch_y)


def test_drop_last_discards_the_short_batch() -> None:
    X = np.zeros((10, 3))
    y = np.zeros(10)
    loader = DataLoader(X, y, batch_size=4, shuffle=False, drop_last=True)

    sizes = [batch_x.shape[0] for batch_x, _ in loader]

    assert sizes == [4, 4]
    assert len(loader) == 2


def test_shuffle_false_preserves_order() -> None:
    X = np.arange(6).reshape(6, 1)
    y = np.arange(6)
    loader = DataLoader(X, y, batch_size=2, shuffle=False)

    assert np.array_equal(np.concatenate([b for _, b in loader]), y)


def test_loader_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        DataLoader(np.zeros((5, 2)), np.zeros(4), batch_size=2)


def test_one_hot_shape_and_content() -> None:
    encoded = one_hot(np.array([0, 2, 1]), num_classes=4)

    assert encoded.shape == (3, 4)
    assert np.array_equal(encoded.sum(axis=1), np.ones(3))
    assert encoded[1, 2] == 1.0


def test_one_hot_infers_class_count() -> None:
    assert one_hot(np.array([0, 3])).shape == (2, 4)


def test_train_val_split_partitions_the_data() -> None:
    X = np.arange(100).reshape(100, 1)
    y = np.arange(100)

    X_train, y_train, X_val, y_val = train_val_split(X, y, val_fraction=0.2, seed=0)

    assert X_train.shape[0] == 80 and X_val.shape[0] == 20
    assert sorted(np.concatenate([y_train, y_val]).tolist()) == list(range(100))
    assert np.array_equal(X_train.ravel(), y_train)
