# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Softmax-head training for class-incremental experiments (NumPy only)."""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """Compute numerically stable row-wise softmax probabilities."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def fit_task(
    weights: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    replay_features: np.ndarray,
    replay_labels: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
    eligible_classes: np.ndarray | None = None,
    update_scale: np.ndarray | None = None,
    null_basis: np.ndarray | None = None,
    feature_scale: np.ndarray | None = None,
) -> np.ndarray:
    """Train a softmax head on the current task's examples plus replay examples.

    If eligible_classes is given, the softmax competition and the gradient are restricted to
    those output columns, so output weights of classes that have not been introduced yet are
    never touched. The weight array is updated in place and returned. Replay examples are
    appended to the current examples for this call only.

    Optional restrictions on the update (used by the strategy comparison):
    ``update_scale`` (n_features, n_classes) multiplies the gradient elementwise (0 freezes a
    coefficient, 0.1 slows it); ``null_basis`` (n_features, k) projects every gradient column
    onto its span; if the basis spans the null space of a design matrix, the logits of those
    design rows are unchanged by every update. ``feature_scale`` (n_features,) gives each
    feature its own step size (diagonal preconditioner D): the step is D g. Combined with
    ``null_basis`` it is D^(1/2) N N^T D^(1/2) g, so ``null_basis`` must then span the null
    space of the design scaled by D^(1/2); this stays a descent direction and keeps the
    constraint exactly.
    """
    if features.ndim != 2 or len(features) == 0:
        raise ValueError("features must be a non-empty 2D array")
    if labels.shape != (len(features),):
        raise ValueError("labels must have one entry per feature row")
    if weights.ndim != 2 or weights.shape[0] != features.shape[1]:
        raise ValueError("weights must have shape (n_features, n_classes)")
    if len(replay_labels):
        if replay_features.ndim != 2 or replay_features.shape[1] != features.shape[1]:
            raise ValueError("replay_features must match the current feature dimension")
        if replay_labels.shape != (len(replay_features),):
            raise ValueError("replay labels must match replay feature rows")
        features = np.concatenate((features, replay_features), axis=0)
        labels = np.concatenate((labels, replay_labels), axis=0)
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not np.issubdtype(labels.dtype, np.integer):
        labels = labels.astype(np.int64)
    if np.any(labels < 0) or np.any(labels >= weights.shape[1]):
        raise ValueError("labels must index an output class")
    if update_scale is not None and update_scale.shape != weights.shape:
        raise ValueError("update_scale must have the same shape as weights")
    if null_basis is not None and null_basis.shape[0] != weights.shape[0]:
        raise ValueError("null_basis must have one row per feature")
    if feature_scale is not None:
        feature_scale = np.asarray(feature_scale, dtype=weights.dtype)
        if feature_scale.shape != (weights.shape[0],) or not np.all(feature_scale > 0):
            raise ValueError(
                "feature_scale must be positive with one entry per feature"
            )
        scale_root = np.sqrt(feature_scale)[:, None]
    if eligible_classes is None:
        eligible_classes = np.arange(weights.shape[1], dtype=np.int64)
    else:
        eligible_classes = np.asarray(eligible_classes, dtype=np.int64)
        if (
            eligible_classes.ndim != 1
            or len(eligible_classes) == 0
            or np.any(eligible_classes < 0)
            or np.any(eligible_classes >= weights.shape[1])
            or np.any(np.diff(eligible_classes) <= 0)
        ):
            raise ValueError("eligible_classes must be sorted, unique, and in range")
        if not np.isin(labels, eligible_classes).all():
            raise ValueError("labels must belong to eligible_classes")

    label_columns = np.searchsorted(eligible_classes, labels)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.permutation(len(labels))
        for start in range(0, len(labels), batch_size):
            batch = order[start : start + batch_size]
            xb, yb = features[batch], label_columns[batch]
            probabilities = softmax(xb @ weights[:, eligible_classes])
            probabilities[np.arange(len(yb)), yb] -= 1.0
            gradient = xb.T @ probabilities / len(yb)
            if update_scale is not None:
                gradient = gradient * update_scale[:, eligible_classes]
            if feature_scale is None:
                if null_basis is not None:
                    gradient = null_basis @ (null_basis.T @ gradient)
            elif null_basis is None:
                gradient = gradient * feature_scale[:, None]
            else:
                gradient = scale_root * (
                    null_basis @ (null_basis.T @ (scale_root * gradient))
                )
            weights[:, eligible_classes] -= learning_rate * gradient
    return weights


def null_space_basis(design: np.ndarray, rtol: float = 1e-10) -> np.ndarray:
    """Orthonormal basis (n_features, k) of {v : design @ v = 0}; k = 0 if nothing is free."""
    design = np.asarray(design, dtype=np.float64)
    if design.shape[0] == 0:
        return np.eye(design.shape[1], dtype=np.float32)
    _, singular, vt = np.linalg.svd(design, full_matrices=True)
    rank = int(np.sum(singular > rtol * max(singular[0], 1.0)))
    return vt[rank:].T.astype(np.float32)


def cross_entropy(weights, features, labels, n_classes):
    """Mean softmax cross-entropy when only the first ``n_classes`` outputs compete."""
    probabilities = softmax(features @ weights[:, :n_classes])
    return float(
        -np.mean(np.log(probabilities[np.arange(len(labels)), labels] + 1e-12))
    )
