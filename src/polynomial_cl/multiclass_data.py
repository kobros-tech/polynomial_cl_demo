# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Reproducible multiclass 2D data for class-incremental learning."""

from __future__ import annotations

import numpy as np

TASK_ORDER = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))


def make_synthetic_2d(
    seed, n_train_per_class=160, n_test_per_class=240, separation=2.5, noise=0.45
):
    """Generate ten ring-arranged Gaussian classes and independent samples."""
    if n_train_per_class < 1 or n_test_per_class < 1:
        raise ValueError("sample counts per class must be positive")
    if not np.isfinite(separation) or separation <= 0:
        raise ValueError("separation must be finite and positive")
    if not np.isfinite(noise) or noise <= 0:
        raise ValueError("noise must be finite and positive")
    rng = np.random.default_rng(seed)
    angles = 2 * np.pi * np.arange(10) / 10
    means = separation * np.column_stack((np.cos(angles), np.sin(angles)))
    train_x = np.concatenate([
        rng.normal(means[c], noise, size=(n_train_per_class, 2)) for c in range(10)
    ]).astype(np.float32)
    test_x = np.concatenate([
        rng.normal(means[c], noise, size=(n_test_per_class, 2)) for c in range(10)
    ]).astype(np.float32)
    return (
        train_x,
        np.repeat(np.arange(10), n_train_per_class),
        test_x,
        np.repeat(np.arange(10), n_test_per_class),
        means.astype(np.float32),
    )
