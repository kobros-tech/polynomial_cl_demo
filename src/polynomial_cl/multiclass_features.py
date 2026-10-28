# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Polynomial features and fixed projections for multiclass CL experiments."""

from __future__ import annotations

import itertools
import math

import numpy as np

N_COMPONENTS = 16
MAX_DEGREE = 3


def make_projection(seed: int, input_dim: int = 784, n_components: int = N_COMPONENTS):
    """Create a deterministic orthonormal random projection without fitting on data."""
    if not 1 <= n_components <= input_dim:
        raise ValueError("n_components must be between 1 and input_dim")
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((input_dim, n_components))
    projection, _ = np.linalg.qr(matrix, mode="reduced")
    return projection.astype(np.float32)


def monomial_exponents(n_components: int, degree: int) -> np.ndarray:
    """Return prefix-stable exponents for all monomials of total degree <= degree."""
    if n_components < 1 or degree < 0:
        raise ValueError("n_components must be positive and degree non-negative")
    rows = [np.zeros(n_components, dtype=np.int16)]
    for total_degree in range(1, degree + 1):
        for indices in itertools.combinations_with_replacement(range(n_components), total_degree):
            rows.append(np.bincount(indices, minlength=n_components).astype(np.int16))
    return np.stack(rows)


def n_polynomial_features(n_components: int, degree: int) -> int:
    """Number of monomials of total degree at most degree, including the constant."""
    if n_components < 1 or degree < 0:
        raise ValueError("n_components must be positive and degree non-negative")
    return math.comb(n_components + degree, degree)


def polynomial_features_projected(
    projected: np.ndarray, degree: int, chunk_size: int = 1024
) -> np.ndarray:
    """Build a degree-balanced polynomial basis in chunks to limit peak memory.

    Each positive-degree group is divided by sqrt(number of monomials in that
    degree). The constant remains 1. Chunking bounds temporary memory use.
    """
    projected = np.asarray(projected, dtype=np.float32)
    if projected.ndim != 2 or projected.shape[1] < 1:
        raise ValueError("projected must have shape (n_samples, n_components)")
    if degree < 0 or chunk_size < 1:
        raise ValueError("degree must be non-negative and chunk_size positive")

    exponents = monomial_exponents(projected.shape[1], degree)
    degree_per_feature = exponents.sum(axis=1)
    scales = np.ones(len(exponents), dtype=np.float32)
    for total_degree in range(1, degree + 1):
        count = math.comb(projected.shape[1] + total_degree - 1, total_degree)
        scales[degree_per_feature == total_degree] = 1.0 / math.sqrt(count)

    output = np.empty((len(projected), len(exponents)), dtype=np.float32)
    for start in range(0, len(projected), chunk_size):
        stop = min(start + chunk_size, len(projected))
        block = projected[start:stop]
        powers = np.power(block[:, None, :], exponents[None, :, :])
        output[start:stop] = np.prod(powers, axis=2) * scales
    return output


def project_images(images: np.ndarray, projection: np.ndarray) -> np.ndarray:
    """Apply fixed MNIST normalization and the shared projection."""
    normalized = (np.asarray(images, dtype=np.float32) - 0.1307) / 0.3081
    if normalized.ndim != 2 or normalized.shape[1] != projection.shape[0]:
        raise ValueError("images must be a 2D array matching projection input dimension")
    return normalized @ projection


def expand_weights(weights: np.ndarray, new_feature_count: int) -> np.ndarray:
    """Append zero coefficients, preserving every existing logit exactly."""
    if new_feature_count < weights.shape[0]:
        raise ValueError("feature expansion cannot shrink the weight matrix")
    if new_feature_count == weights.shape[0]:
        return weights
    expanded = np.zeros((new_feature_count, weights.shape[1]), dtype=weights.dtype)
    expanded[: weights.shape[0]] = weights
    return expanded
