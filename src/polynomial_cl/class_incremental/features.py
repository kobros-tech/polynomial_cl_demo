# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Explicit polynomial features in any number of variables, and zero-initialised expansion.

Unlike ``polynomial_cl.features`` (2-D, unscaled, used by the binary experiments), this map
works for any input dimension and divides every degree group by the square root of its size,
so that higher-degree groups do not dominate merely because they contain more monomials.
Monomials are ordered by total degree, so the features of degree d are a prefix of those of
degree d + 1; a model can grow by appending zero coefficients without changing its outputs.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

MAX_DEGREE = 3


def monomial_exponents(n_components: int, degree: int) -> np.ndarray:
    """Exponent rows of all monomials of total degree <= degree (prefix-stable order)."""
    if n_components < 1 or degree < 0:
        raise ValueError("n_components must be positive and degree non-negative")
    rows = [np.zeros(n_components, dtype=np.int16)]
    for total_degree in range(1, degree + 1):
        for indices in itertools.combinations_with_replacement(
            range(n_components), total_degree
        ):
            rows.append(np.bincount(indices, minlength=n_components).astype(np.int16))
    return np.stack(rows)


def n_polynomial_features(n_components: int, degree: int) -> int:
    """Number of monomials of total degree at most degree, including the constant."""
    if n_components < 1 or degree < 0:
        raise ValueError("n_components must be positive and degree non-negative")
    return math.comb(n_components + degree, degree)


def scaled_polynomial_features(
    x: np.ndarray, degree: int, chunk_size: int = 1024
) -> np.ndarray:
    """Degree-balanced polynomial design matrix, built in chunks to bound peak memory.

    Each positive-degree group is divided by sqrt(number of monomials in that degree); the
    constant feature stays 1.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] < 1:
        raise ValueError("x must have shape (n_samples, n_components)")
    if degree < 0 or chunk_size < 1:
        raise ValueError("degree must be non-negative and chunk_size positive")

    exponents = monomial_exponents(x.shape[1], degree)
    degree_of_feature = exponents.sum(axis=1)
    scales = np.ones(len(exponents), dtype=np.float32)
    for total_degree in range(1, degree + 1):
        group_size = math.comb(x.shape[1] + total_degree - 1, total_degree)
        scales[degree_of_feature == total_degree] = 1.0 / math.sqrt(group_size)

    out = np.empty((len(x), len(exponents)), dtype=np.float32)
    for start in range(0, len(x), chunk_size):
        block = x[start : start + chunk_size]
        powers = np.power(block[:, None, :], exponents[None, :, :])
        out[start : start + len(block)] = np.prod(powers, axis=2) * scales
    return out


def expand_weights(weights: np.ndarray, new_feature_count: int) -> np.ndarray:
    """Append zero coefficients (rows); every existing logit is preserved exactly."""
    if new_feature_count < weights.shape[0]:
        raise ValueError("feature expansion cannot shrink the weight matrix")
    if new_feature_count == weights.shape[0]:
        return weights
    expanded = np.zeros((new_feature_count, weights.shape[1]), dtype=weights.dtype)
    expanded[: weights.shape[0]] = weights
    return expanded
