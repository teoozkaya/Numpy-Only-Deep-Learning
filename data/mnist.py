"""MNIST download, parsing and normalization.

The raw IDX files are cached under ``root/raw`` and the decoded arrays under
``root/mnist.npz`` so that only the first call touches the network.
"""

from __future__ import annotations

import gzip
import os
import struct
import urllib.request
from typing import Dict, Tuple

import numpy as np

MIRRORS = (
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
)

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mnist")


def _download(filename: str, root: str) -> str:
    """Download ``filename`` into ``root/raw`` if it is not cached yet.

    Returns the local path of the file.
    """
    raw_dir = os.path.join(root, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    path = os.path.join(raw_dir, filename)
    if os.path.exists(path):
        return path

    errors = []
    for mirror in MIRRORS:
        url = mirror + filename
        try:
            print("downloading {} ...".format(url))
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = response.read()
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            errors.append("{}: {}".format(url, exc))
            continue
        tmp = path + ".part"
        with open(tmp, "wb") as handle:
            handle.write(payload)
        os.replace(tmp, path)
        return path

    raise RuntimeError(
        "could not download {} from any mirror:\n  {}".format(
            filename, "\n  ".join(errors)
        )
    )


def _read_idx_images(path: str) -> np.ndarray:
    """Decode an IDX image file into an array of shape (N, 28, 28), uint8."""
    with gzip.open(path, "rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError("{}: bad magic number {}".format(path, magic))
        buffer = handle.read(count * rows * cols)
    return np.frombuffer(buffer, dtype=np.uint8).reshape(count, rows, cols)


def _read_idx_labels(path: str) -> np.ndarray:
    """Decode an IDX label file into an array of shape (N,), uint8."""
    with gzip.open(path, "rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError("{}: bad magic number {}".format(path, magic))
        buffer = handle.read(count)
    return np.frombuffer(buffer, dtype=np.uint8)


def _load_raw(root: str) -> Dict[str, np.ndarray]:
    """Return the four raw MNIST arrays, using the .npz cache when present."""
    cache = os.path.join(root, "mnist.npz")
    if os.path.exists(cache):
        with np.load(cache) as archive:
            return {key: archive[key] for key in FILES}

    arrays = {}
    for key, filename in FILES.items():
        path = _download(filename, root)
        if key.endswith("images"):
            arrays[key] = _read_idx_images(path)
        else:
            arrays[key] = _read_idx_labels(path)

    os.makedirs(root, exist_ok=True)
    np.savez_compressed(cache, **arrays)
    return arrays


def load_mnist(
    root: str = DEFAULT_ROOT,
    flatten: bool = True,
    normalize: bool = True,
    standardize: bool = False,
    validation_size: int = 5000,
    dtype: type = np.float32,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Load MNIST as train / validation / test splits.

    Args:
        root: directory used to cache the downloaded files.
        flatten: if True images have shape (N, 784), otherwise (N, 1, 28, 28).
        normalize: scale pixel values from [0, 255] to [0, 1].
        standardize: subtract the training mean and divide by the training std
            (per feature). Applied after ``normalize``.
        validation_size: number of training images held out for validation.
        dtype: floating point type of the returned images.
        seed: seed of the shuffle used to carve out the validation split.

    Returns:
        Dict with keys ``X_train``, ``y_train``, ``X_val``, ``y_val``,
        ``X_test``, ``y_test``. Images have shape (N, 784) or (N, 1, 28, 28)
        and labels shape (N,) with integer values in [0, 9].
    """
    raw = _load_raw(root)

    X_train = raw["train_images"].astype(dtype)
    y_train = raw["train_labels"].astype(np.int64)
    X_test = raw["test_images"].astype(dtype)
    y_test = raw["test_labels"].astype(np.int64)

    if normalize:
        X_train /= 255.0
        X_test /= 255.0

    if flatten:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
    else:
        X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
        X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

    if not 0 <= validation_size < X_train.shape[0]:
        raise ValueError(
            "validation_size must be in [0, {}), got {}".format(
                X_train.shape[0], validation_size
            )
        )

    rng = np.random.default_rng(seed)
    perm = rng.permutation(X_train.shape[0])
    val_idx, train_idx = perm[:validation_size], perm[validation_size:]
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    X_train, y_train = X_train[train_idx], y_train[train_idx]

    if standardize:
        mean = X_train.mean(axis=0, keepdims=True)
        std = X_train.std(axis=0, keepdims=True) + 1e-8
        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std
        X_test = (X_test - mean) / std

    return {
        "X_train": np.ascontiguousarray(X_train, dtype=dtype),
        "y_train": y_train,
        "X_val": np.ascontiguousarray(X_val, dtype=dtype),
        "y_val": y_val,
        "X_test": np.ascontiguousarray(X_test, dtype=dtype),
        "y_test": y_test,
    }


def sample_batch(
    X: np.ndarray, y: np.ndarray, size: int, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw a small random batch, handy for overfitting sanity checks.

    Args:
        X: inputs of shape (N, ...).
        y: labels of shape (N,).
        size: number of samples to draw without replacement.

    Returns:
        (X_batch, y_batch) of shapes (size, ...) and (size,).
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=size, replace=False)
    return X[idx], y[idx]
