# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Explicit polynomial feature map for 2-D inputs (total degree <= d, constant included).

Ordering: by total degree k = 0..d, and within a degree by decreasing power of x1:
    degree 0: 1
    degree 1: x1, x2
    degree 2: x1^2, x1*x2, x2^2
    degree 3: x1^3, x1^2*x2, x1*x2^2, x2^3 ...
Consequence: the features of degree d are a *prefix* of those of degree d+1, so a model
can be expanded by appending zero coefficients without touching existing ones.

Note: this is explicit feature expansion (related to polynomial kernels). It is NOT an
implicit-kernel SVM.
"""

from __future__ import annotations

import numpy as np


def _validate_degree(degree: int) -> int:
    """Require a non-negative integer degree; booleans are not valid degrees."""
    if isinstance(degree, (bool, np.bool_)) or not isinstance(degree, (int, np.integer)):
        raise TypeError("degree must be a non-negative integer")
    if degree < 0:
        raise ValueError("degree must be >= 0")
    return int(degree)


def feature_exponents(degree: int):
    """List of (i, j) meaning x1^i * x2^j, in the canonical order."""
    degree = _validate_degree(degree)
    return [(i, k - i) for k in range(degree + 1) for i in range(k, -1, -1)]


def _name(i: int, j: int) -> str:
    parts = []
    for var, p in (("x1", i), ("x2", j)):
        if p == 1:
            parts.append(var)
        elif p > 1:
            parts.append(f"{var}^{p}")
    return "*".join(parts) if parts else "1"


def feature_names(degree: int):
    return [_name(i, j) for i, j in feature_exponents(degree)]


def n_features(degree: int) -> int:
    degree = _validate_degree(degree)
    return (degree + 1) * (degree + 2) // 2


def polynomial_features(x, degree: int) -> np.ndarray:
    """Design matrix Phi_d(x) of shape (n, n_features(degree))."""
    degree = _validate_degree(degree)
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if x.shape[1] != 2:
        raise ValueError("expected inputs of shape (n, 2)")
    p = np.arange(degree + 1)
    p1 = x[:, :1] ** p  # (n, d+1): x1^0..x1^d
    p2 = x[:, 1:] ** p
    return np.column_stack([p1[:, i] * p2[:, j] for i, j in feature_exponents(degree)])


class PolynomialFeatures:
    """Small convenience wrapper exposing degree, names, and feature count."""

    def __init__(self, degree: int):
        self.degree = _validate_degree(degree)
        self.exponents = feature_exponents(self.degree)
        self.names = feature_names(self.degree)
        self.n_features = len(self.names)

    def __call__(self, x):
        return polynomial_features(x, self.degree)
