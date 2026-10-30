# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Datasets for the class-incremental comparison and their fixed input representation.

Both datasets have ten classes introduced in five two-class experiences (``TASK_ORDER``):

* ``make_synthetic_2d``: ten Gaussian clusters on a ring in the plane.
* ``load_projected_mnist``: MNIST digits, normalised and mapped to ``N_COMPONENTS``
  dimensions by a fixed random orthonormal projection (never fitted to data), so the same
  explicit polynomial machinery applies to both.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .mnist import load_mnist

TASK_ORDER = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))
N_COMPONENTS = 16


def make_synthetic_2d(
    seed, n_train_per_class=160, n_test_per_class=240, separation=2.5, noise=0.45
):
    """Ten ring-arranged Gaussian classes with independent train and test samples.

    Returns (train_x, train_y, test_x, test_y, class_means).
    """
    if n_train_per_class < 1 or n_test_per_class < 1:
        raise ValueError("sample counts per class must be positive")
    if not np.isfinite(separation) or separation <= 0:
        raise ValueError("separation must be finite and positive")
    if not np.isfinite(noise) or noise <= 0:
        raise ValueError("noise must be finite and positive")
    rng = np.random.default_rng(seed)
    angles = 2 * np.pi * np.arange(10) / 10
    means = separation * np.column_stack((np.cos(angles), np.sin(angles)))
    train_x = np.concatenate(
        [rng.normal(means[c], noise, size=(n_train_per_class, 2)) for c in range(10)]
    ).astype(np.float32)
    test_x = np.concatenate(
        [rng.normal(means[c], noise, size=(n_test_per_class, 2)) for c in range(10)]
    ).astype(np.float32)
    return (
        train_x,
        np.repeat(np.arange(10), n_train_per_class),
        test_x,
        np.repeat(np.arange(10), n_test_per_class),
        means.astype(np.float32),
    )


def make_projection(seed: int, input_dim: int = 784, n_components: int = N_COMPONENTS):
    """Deterministic orthonormal random projection matrix (input_dim, n_components)."""
    if not 1 <= n_components <= input_dim:
        raise ValueError("n_components must be between 1 and input_dim")
    matrix = np.random.default_rng(seed).standard_normal((input_dim, n_components))
    projection, _ = np.linalg.qr(matrix, mode="reduced")
    return projection.astype(np.float32)


def project_images(images: np.ndarray, projection: np.ndarray) -> np.ndarray:
    """Apply the fixed MNIST normalisation, then the shared projection."""
    normalized = (np.asarray(images, dtype=np.float32) - 0.1307) / 0.3081
    if normalized.ndim != 2 or normalized.shape[1] != projection.shape[0]:
        raise ValueError(
            "images must be a 2D array matching projection input dimension"
        )
    return normalized @ projection


def load_projected_mnist(root, projection_seed=0, download=True):
    """Load MNIST from ``root`` and project it. Returns (train_x, train_y, test_x, test_y)."""
    train_x, train_y, test_x, test_y = load_mnist(Path(root), download=download)
    projection = make_projection(projection_seed)
    return (
        project_images(train_x, projection),
        train_y,
        project_images(test_x, projection),
        test_y,
    )


def holdout_split(train_x, train_y, fraction, seed=123):
    """Random split of the training data into (train, validation) arrays.

    Use the validation part, never the test set, to choose hyper-parameters.
    """
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between 0 and 1")
    order = np.random.default_rng(seed).permutation(len(train_x))
    cut = len(train_x) - round(fraction * len(train_x))
    fit, held = order[:cut], order[cut:]
    for part, name in ((fit, "training"), (held, "validation")):
        if len(np.unique(train_y[part])) != len(np.unique(train_y)):
            raise ValueError(
                f"the {name} part is missing a class; use more data or another fraction"
            )
    return train_x[fit], train_y[fit], train_x[held], train_y[held]
