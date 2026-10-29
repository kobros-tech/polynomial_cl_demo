# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Matched class-incremental comparison of polynomial feature expansion.

Ten classes arrive in five two-class experiences. A single softmax head with ten outputs is
trained on the current experience plus a fixed per-class replay memory; training and
evaluation only let classes introduced so far compete, so the output weights of future
classes are never trained. Three conditions are compared on the *same* task order, replay
indices, seed, optimiser and number of epochs; they differ only in polynomial degree:

* ``fixed_linear``:   degree 1 throughout (no expansion).
* ``expanding``:      degree 1, 2, then 3 from experience 3 on; new coefficients start at zero.
* ``fixed_capacity``: degree 3 from experience 1 (same final capacity, no gradual growth).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ..metrics import mean_std_by_condition
from .data import TASK_ORDER
from .features import (
    MAX_DEGREE,
    expand_weights,
    n_polynomial_features,
    scaled_polynomial_features,
)
from .training import fit_task

CONDITIONS = ("fixed_linear", "expanding", "fixed_capacity")
DEGREES = {
    "fixed_linear": (1, 1, 1, 1, 1),
    "expanding": (1, 2, 3, 3, 3),
    "fixed_capacity": (MAX_DEGREE,) * 5,
}
SUMMARY_METRICS = ("final_mean_accuracy", "final_current_accuracy", "mean_forgetting")

# Defaults shared by the command line tool and the notebook.
EPOCHS = 3
LEARNING_RATE = 0.1
BATCH_SIZE = 128
MEMORY_PER_CLASS = 200

# Largest allowed change of any logit when zero coefficients are appended (float32 rounding).
EXPANSION_TOLERANCE = 1e-5


class FeatureData(NamedTuple):
    """Degree-``MAX_DEGREE`` design matrices; lower degrees are column prefixes of these."""

    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    n_components: int


def make_feature_data(train_x, train_y, test_x, test_y) -> FeatureData:
    """Compute the polynomial features once; every condition and seed reuses them."""
    return FeatureData(
        scaled_polynomial_features(train_x, MAX_DEGREE),
        np.asarray(train_y),
        scaled_polynomial_features(test_x, MAX_DEGREE),
        np.asarray(test_y),
        np.shape(train_x)[1],
    )


def make_task_plan(labels, seed, memory_per_class):
    """Shuffled training indices per experience, and one fixed replay memory per class."""
    if memory_per_class < 1:
        raise ValueError("memory_per_class must be positive")
    rng = np.random.default_rng(seed)
    tasks, memory = [], {}
    for classes in TASK_ORDER:
        indices = np.flatnonzero(np.isin(labels, classes))
        if len(indices) == 0:
            raise ValueError(f"training data is missing task classes {classes}")
        tasks.append(indices[rng.permutation(len(indices))])
        for class_id in classes:
            candidates = np.flatnonzero(labels == class_id)
            if len(candidates) == 0:
                raise ValueError(f"training data is missing class {class_id}")
            memory[class_id] = rng.choice(
                candidates, min(memory_per_class, len(candidates)), replace=False
            )
    return tasks, memory


def accuracy_by_task(weights, test_x, test_y, n_tasks_seen):
    """Accuracy on every experience seen so far (NaN for later ones).

    Predictions are the argmax over the classes introduced so far only.
    """
    accuracy = np.full(len(TASK_ORDER), np.nan)
    for task, classes in enumerate(TASK_ORDER[:n_tasks_seen]):
        rows = np.flatnonzero(np.isin(test_y, classes))
        logits = test_x[rows] @ weights
        predictions = logits[:, : 2 * n_tasks_seen].argmax(axis=1)
        accuracy[task] = np.mean(predictions == test_y[rows])
    return accuracy


def run_condition(
    condition,
    data: FeatureData,
    task_indices,
    memory_indices,
    *,
    seed,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE,
):
    """Run one condition; returns a JSON-serialisable record with the accuracy matrix."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    degrees = DEGREES[condition]
    n_tasks = len(degrees)
    weights = np.zeros(
        (n_polynomial_features(data.n_components, degrees[0]), 10), np.float32
    )
    matrix = np.full((n_tasks, n_tasks), np.nan)
    feature_counts, expansion_changes = [], []

    for task, indices in enumerate(task_indices):
        count = n_polynomial_features(data.n_components, degrees[task])
        if count > weights.shape[0]:
            # Expansion invariant: appending zero coefficients must not change any logit.
            probe = data.test_x[:256, :count]
            before = probe[:, : weights.shape[0]] @ weights
            weights = expand_weights(weights, count)
            change = float(np.max(np.abs(probe @ weights - before)))
            if change > EXPANSION_TOLERANCE:
                raise RuntimeError(
                    f"zero-initialized expansion changed logits by {change}"
                )
            expansion_changes.append(change)
        feature_counts.append(count)

        if task:
            replay = np.concatenate([memory_indices[c] for c in range(2 * task)])
        else:
            replay = np.empty(0, dtype=np.int64)
        weights = fit_task(
            weights,
            data.train_x[indices, :count],
            data.train_y[indices].astype(np.int64, copy=False),
            data.train_x[replay, :count],
            data.train_y[replay].astype(np.int64, copy=False),
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            seed=seed + task,
            eligible_classes=np.arange(2 * (task + 1)),
        )
        matrix[task, : task + 1] = accuracy_by_task(
            weights, data.test_x[:, :count], data.test_y, task + 1
        )[: task + 1]

    # Forgetting of an experience: best accuracy it ever had minus its final accuracy.
    forgetting = [
        max(0.0, float(np.nanmax(matrix[:, task]) - matrix[-1, task]))
        for task in range(n_tasks)
    ]
    return {
        "condition": condition,
        "seed": seed,
        "active_degree_by_task": list(degrees),
        "feature_count_by_task": feature_counts,
        "expansion_max_logit_changes": expansion_changes,
        "accuracy_matrix": [
            [float(v) if np.isfinite(v) else None for v in row] for row in matrix
        ],
        "final_mean_accuracy": float(np.nanmean(matrix[-1])),
        "final_current_accuracy": float(matrix[-1, -1]),
        "forgetting_by_task": forgetting,
        # The last experience has no later evaluation, so it is excluded from the mean.
        "mean_forgetting": float(np.mean(forgetting[:-1])),
    }


def summarize_runs(runs):
    """Mean and sample standard deviation over seeds, per condition."""
    return mean_std_by_condition(runs, SUMMARY_METRICS)


def run_comparison(
    data: FeatureData,
    seeds,
    *,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    memory_per_class=MEMORY_PER_CLASS,
):
    """Run all three conditions for every seed; within a seed they share task order and replay.

    ``data`` comes from ``make_feature_data`` (computed once, reusable across calls).
    """
    if not len(seeds):
        raise ValueError("at least one seed is required")
    runs = []
    for seed in seeds:
        tasks, memory = make_task_plan(data.train_y, seed, memory_per_class)
        for condition in CONDITIONS:
            runs.append(
                run_condition(
                    condition,
                    data,
                    tasks,
                    memory,
                    seed=seed,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                )
            )
    return {"runs": runs, "summary": summarize_runs(runs)}
