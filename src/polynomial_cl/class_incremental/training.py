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
) -> np.ndarray:
    """Train a softmax head on the current task's examples plus replay examples.

    If eligible_classes is given, the softmax competition and the gradient are restricted to
    those output columns, so output weights of classes that have not been introduced yet are
    never touched. The weight array is updated in place and returned. Replay examples are
    appended to the current examples for this call only.
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
            weights[:, eligible_classes] -= learning_rate * (
                xb.T @ probabilities / len(yb)
            )
    return weights
