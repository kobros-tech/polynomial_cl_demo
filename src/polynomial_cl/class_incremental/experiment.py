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
from .training import cross_entropy, fit_task, null_space_basis

CONDITIONS = ("fixed_linear", "expanding", "fixed_capacity")
DEGREES = {
    "fixed_linear": (1, 1, 1, 1, 1),
    "expanding": (1, 2, 3, 3, 3),
    "fixed_capacity": (MAX_DEGREE,) * 5,
}
SUMMARY_METRICS = ("final_mean_accuracy", "final_current_accuracy", "mean_forgetting")

# Update rules compared on the ``expanding`` schedule (same definitions as the binary
# experiments in polynomial_cl.training). A coefficient is "old" if it existed before the
# current experience: its feature row was already present AND its class column was already
# introduced. New feature rows and the new classes' columns always train freely.
STRATEGIES = (
    "ordinary",
    "ordinary_no_replay",
    "small_old_lr",
    "frozen_old",
    "strategy_b",
)
OLD_LR_FACTOR = 0.1
N_REF_PER_CLASS = (
    3  # strategy_b: reference inputs per old class (first stored replay examples)
)
STRATEGY_METRICS = (
    *SUMMARY_METRICS,
    "mean_old_logit_change",
    "mean_old_flip_fraction",
    "mean_loss_reduction",
)

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


INDICATORS = (
    "train_loss_by_task",  # [loss before, loss after] on this experience's training set (+ replay)
    "l1_update_by_task",  # sum |w_after - w_before| over the coefficients used this experience
    "n_trainable_by_task",  # coefficients that may change (null-space dimension x classes for B)
    "null_dim_by_task",  # strategy_b only: dimension left free by the constraint
    "old_logit_change_by_task",  # max change of old-class logits on old-class test inputs
    "old_flip_fraction_by_task",  # old-class test inputs whose prediction among old classes changed
    "ref_logit_change_by_task",  # strategy_b only: max logit change at the reference inputs
)


def _mean_after_first(values):
    """Mean over experiences 2..5 (experience 1 has no old function to preserve)."""
    return float(np.mean([v for v in values[1:] if v is not None]))


def _record_indicators(
    out, weights, before, data, count, task, old_test, refs, scale, null_basis,
    loss_before, loss_after, train_x, train_y,
):  # fmt: skip
    n_seen, old_classes = 2 * (task + 1), 2 * task
    used = (slice(None, count), slice(None, n_seen))
    out["train_loss_by_task"].append([loss_before, loss_after])
    out["l1_update_by_task"].append(float(np.abs(weights - before)[used].sum()))
    if null_basis is not None:
        trainable = null_basis.shape[1] * n_seen
    elif scale is not None:
        trainable = int((scale[used] > 0).sum())
    else:
        trainable = count * n_seen
    out["n_trainable_by_task"].append(int(trainable))
    out["null_dim_by_task"].append(
        None if null_basis is None else int(null_basis.shape[1])
    )
    if task:
        x = data.test_x[old_test, :count]
        old_before, old_after = (
            x @ before[:, :old_classes],
            x @ weights[:, :old_classes],
        )
        out["old_logit_change_by_task"].append(
            float(np.max(np.abs(old_after - old_before)))
        )
        flips = old_before.argmax(axis=1) != old_after.argmax(axis=1)
        out["old_flip_fraction_by_task"].append(float(flips.mean()))
    else:
        out["old_logit_change_by_task"].append(None)
        out["old_flip_fraction_by_task"].append(None)
    if refs is None:
        out["ref_logit_change_by_task"].append(None)
    else:
        ref_x = data.train_x[refs, :count]
        change = np.abs(ref_x @ weights[:, :n_seen] - ref_x @ before[:, :n_seen])
        out["ref_logit_change_by_task"].append(float(change.max()))


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
    strategy="ordinary",
    n_ref_per_class=N_REF_PER_CLASS,
    feature_power=0.0,
):
    """Run one condition; returns a JSON-serialisable record with the accuracy matrix.

    ``strategy`` selects the update rule (see ``STRATEGIES``); the default is plain training with
    replay. Besides accuracy, every record carries indicators of whether the model is learning
    (loss before/after, size of the update, trainable coefficients) and how much the old
    function moved (change of old-class logits on old-class test inputs).

    ``feature_power`` p > 0 gives every feature its own step size
    ``learning_rate * (max_rms / rms_i) ** p`` (rms of feature i on the experience's training
    set): p = 0 is one shared rate, p = 1 diagonal preconditioning.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy}")
    degrees = DEGREES[condition]
    n_tasks = len(degrees)
    weights = np.zeros(
        (n_polynomial_features(data.n_components, degrees[0]), 10), np.float32
    )
    matrix = np.full((n_tasks, n_tasks), np.nan)
    feature_counts, expansion_changes = [], []
    indicators = {k: [] for k in INDICATORS}

    for task, indices in enumerate(task_indices):
        old_rows = weights.shape[0] if task else 0
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

        old_classes = 2 * task
        n_seen = 2 * (task + 1)
        if task and strategy != "ordinary_no_replay":
            replay = np.concatenate([memory_indices[c] for c in range(old_classes)])
        else:
            replay = np.empty(0, dtype=np.int64)
        train_x = np.concatenate(
            [data.train_x[indices, :count], data.train_x[replay, :count]]
        )
        train_y = data.train_y[np.concatenate([indices, replay])].astype(
            np.int64, copy=False
        )
        feature_scale = None
        if feature_power:
            rms = np.sqrt(np.mean(train_x.astype(np.float64) ** 2, axis=0))
            feature_scale = (
                (rms.max() / np.maximum(rms, 1e-12)) ** feature_power
            ).astype(np.float32)
        scale = null_basis = refs = None
        if strategy in ("frozen_old", "small_old_lr"):
            scale = np.ones(weights.shape, np.float32)
            scale[:old_rows, :old_classes] = (
                0.0 if strategy == "frozen_old" else OLD_LR_FACTOR
            )
        elif strategy == "strategy_b" and task:
            refs = np.concatenate(
                [memory_indices[c][:n_ref_per_class] for c in range(old_classes)]
            )
            ref_design = data.train_x[refs, :count]
            if feature_scale is not None:
                ref_design = ref_design * np.sqrt(feature_scale)
            null_basis = null_space_basis(ref_design)
        old_test = np.flatnonzero(data.test_y < old_classes)
        before = weights.copy()
        loss_before = cross_entropy(weights, train_x, train_y, n_seen)
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
            eligible_classes=np.arange(n_seen),
            update_scale=scale,
            null_basis=null_basis,
            feature_scale=feature_scale,
        )
        _record_indicators(
            indicators, weights, before, data, count, task, old_test, refs, scale, null_basis,
            loss_before, cross_entropy(weights, train_x, train_y, n_seen), train_x, train_y,
        )  # fmt: skip
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
        "strategy": strategy,
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
        **indicators,
        "mean_old_logit_change": _mean_after_first(
            indicators["old_logit_change_by_task"]
        ),
        "mean_old_flip_fraction": _mean_after_first(
            indicators["old_flip_fraction_by_task"]
        ),
        "mean_loss_reduction": float(
            np.mean(
                [before - after for before, after in indicators["train_loss_by_task"]]
            )
        ),
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
    feature_power=0.0,
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
                    feature_power=feature_power,
                )
            )
    return {"runs": runs, "summary": summarize_runs(runs)}


def run_strategy_comparison(
    data: FeatureData,
    seeds,
    *,
    strategies=STRATEGIES,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    memory_per_class=MEMORY_PER_CLASS,
    n_ref_per_class=N_REF_PER_CLASS,
    feature_power=0.0,
):
    """Compare update rules on the ``expanding`` schedule; same task order and replay per seed."""
    if not len(seeds):
        raise ValueError("at least one seed is required")
    runs = []
    for seed in seeds:
        tasks, memory = make_task_plan(data.train_y, seed, memory_per_class)
        for strategy in strategies:
            runs.append(
                run_condition(
                    "expanding", data, tasks, memory, seed=seed, epochs=epochs,
                    learning_rate=learning_rate, batch_size=batch_size,
                    strategy=strategy, n_ref_per_class=n_ref_per_class, feature_power=feature_power,
                )
            )  # fmt: skip
    return {
        "runs": runs,
        "summary": mean_std_by_condition(runs, STRATEGY_METRICS, key="strategy"),
    }
