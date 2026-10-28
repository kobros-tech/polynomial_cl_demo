# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Reusable class-incremental training, replay, evaluation, and aggregation."""

from __future__ import annotations

import numpy as np

from .multiclass import fit_task
from .multiclass_data import TASK_ORDER
from .multiclass_features import (
    MAX_DEGREE,
    expand_weights,
    n_polynomial_features,
    polynomial_features_projected,
)

CONDITIONS = ("fixed_linear", "expanding", "fixed_capacity")


def mean_forgetting(forgetting_by_task):
    """Average forgetting over tasks with a later evaluation, excluding the final task."""
    values = np.asarray(forgetting_by_task, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("forgetting_by_task must contain at least two task values")
    return float(np.mean(values[:-1]))


def make_task_plan(labels, seed, memory_per_class):
    """Create reproducible two-class tasks and one fixed replay memory per class."""
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


def masking_prediction_diagnostics(weights, test_features, labels, task):
    """Measure how masking unseen classes changes predictions on the same examples."""
    eligible_classes = np.arange(2 * (task + 1))
    eligible_examples = np.isin(labels, eligible_classes)
    logits = test_features[eligible_examples] @ weights
    eligible_labels = labels[eligible_examples].astype(np.int64, copy=False)

    unmasked_predictions = logits.argmax(axis=1)
    masked_logits = logits.copy()
    masked_logits[:, len(eligible_classes):] = -np.inf
    masked_predictions = masked_logits.argmax(axis=1)
    changed = unmasked_predictions != masked_predictions
    unmasked_correct = unmasked_predictions == eligible_labels
    masked_correct = masked_predictions == eligible_labels
    n_examples = int(eligible_labels.size)
    return {
        "experience": int(task + 1),
        "n_test_examples": n_examples,
        "n_predictions_changed": int(changed.sum()),
        "prediction_change_rate": float(changed.mean()) if n_examples else 0.0,
        "n_correct_masked": int(masked_correct.sum()),
        "n_correct_unmasked": int(unmasked_correct.sum()),
        "accuracy_masked": float(masked_correct.mean()) if n_examples else 0.0,
        "accuracy_unmasked": float(unmasked_correct.mean()) if n_examples else 0.0,
    }

def evaluate_seen_tasks(weights, test_features, labels, task, mask_unseen=True):
    """Evaluate seen tasks with either current-class or fixed ten-class eligibility."""
    results = np.full(5, np.nan)
    eligible = np.arange(2 * (task + 1))
    for previous_task, classes in enumerate(TASK_ORDER[:task + 1]):
        indices = np.flatnonzero(np.isin(labels, classes))
        logits = test_features[indices] @ weights
        if mask_unseen:
            logits[:, np.setdiff1d(np.arange(10), eligible)] = -np.inf
        results[previous_task] = np.mean(logits.argmax(axis=1) == labels[indices])
    return results


def run_condition(condition, train_features, train_labels, test_features, test_labels,
                  task_indices, memory_indices, *, seed, epochs, learning_rate,
                  batch_size, precomputed_features=None):
    """Run one condition and record the complete task-by-experience accuracy matrix."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    active_degrees = {
        "fixed_linear": [1] * 5,
        "expanding": [1, 2, 3, 3, 3],
        "fixed_capacity": [MAX_DEGREE] * 5,
    }[condition]
    n_components = train_features.shape[1]
    weights = np.zeros((n_polynomial_features(n_components, active_degrees[0]), 10),
                       dtype=np.float32)
    matrix = np.full((5, 5), np.nan)
    all_classes_matrix = np.full((5, 5), np.nan)
    feature_counts, expansion_changes = [], []
    diagnostics_by_experience = []
    masking_diagnostics_by_experience = []
    replay_memory_available_by_task = []
    replay_examples_in_batch_by_task = []
    replay_example_exposures_by_task = []
    replay_examples_by_class_in_batch_by_task = []
    expansion_prediction_flips = []
    if precomputed_features is None:
        train_poly = polynomial_features_projected(train_features, max(active_degrees))
        test_poly = polynomial_features_projected(test_features, max(active_degrees))
    else:
        train_poly, test_poly = precomputed_features

    for task, indices in enumerate(task_indices):
        degree = active_degrees[task]
        count = n_polynomial_features(n_components, degree)
        old_count = weights.shape[0]
        if count > old_count:
            probe = test_poly[:min(256, len(test_poly)), :count]
            before = probe[:, :old_count] @ weights
            weights = expand_weights(weights, count)
            after = probe @ weights
            change = float(np.max(np.abs(after - before)))
            expansion_prediction_flips.append(
                int(np.count_nonzero(after.argmax(axis=1) != before.argmax(axis=1)))
            )
            if change > 1e-5:
                raise RuntimeError(
                    f"zero-initialized expansion changed logits by {change}"
                )
            expansion_changes.append(change)
        feature_counts.append(count)
        current_x = train_poly[indices, :count]
        current_y = train_labels[indices].astype(np.int64, copy=False)
        replay_idx = (np.concatenate([memory_indices[c] for c in range(2 * task)])
                      if task else np.empty(0, dtype=np.int64))
        replay_memory_available_by_task.append(
            int(sum(len(memory_indices[c]) for c in range(2 * (task + 1))))
        )
        replay_examples_in_batch_by_task.append(int(len(replay_idx)))
        replay_example_exposures_by_task.append(int(len(replay_idx) * epochs))
        replay_examples_by_class_in_batch_by_task.append({
            str(class_id): int(len(memory_indices[class_id]))
            for class_id in range(2 * task)
        })
        weights = fit_task(
            weights, current_x, current_y, train_poly[replay_idx, :count],
            train_labels[replay_idx].astype(np.int64, copy=False),
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            seed=seed + task,
            eligible_classes=np.arange(2 * (task + 1)),
        )
        matrix[task, :task + 1] = evaluate_seen_tasks(
            weights, test_poly[:, :count], test_labels, task, mask_unseen=True
        )[:task + 1]
        all_classes_matrix[task, :task + 1] = evaluate_seen_tasks(
            weights, test_poly[:, :count], test_labels, task, mask_unseen=False
        )[:task + 1]
        masking_diagnostics_by_experience.append(
            masking_prediction_diagnostics(
                weights, test_poly[:, :count], test_labels, task
            )
        )

        # Keep class-level evidence alongside aggregate task accuracy. The primary
        # confusion matrix includes only examples whose true classes have arrived.
        eligible_classes = np.arange(2 * (task + 1))
        eligible_test = np.isin(test_labels, eligible_classes)
        eligible_logits = test_poly[eligible_test, :count] @ weights
        eligible_logits[:, 2 * (task + 1):] = -np.inf
        eligible_predictions = eligible_logits.argmax(axis=1)
        eligible_labels = test_labels[eligible_test].astype(np.int64, copy=False)
        task_confusion = np.zeros((10, 10), dtype=np.int64)
        np.add.at(task_confusion, (eligible_labels, eligible_predictions), 1)
        class_accuracy = {}
        for class_id in eligible_classes:
            class_rows = eligible_labels == class_id
            class_accuracy[str(int(class_id))] = float(
                np.mean(eligible_predictions[class_rows] == class_id)
            )
        diagnostics_by_experience.append({
            "experience": task + 1,
            "eligible_classes": eligible_classes.tolist(),
            "class_accuracy": class_accuracy,
            "confusion_matrix": task_confusion.tolist(),
            "evaluated_examples": int(eligible_labels.size),
        })

    forgetting = [
        max(0.0, float(np.nanmax(matrix[:, task]) - matrix[-1, task]))
        for task in range(5)
    ]
    all_classes_forgetting = [
        max(
            0.0,
            float(
                np.nanmax(all_classes_matrix[:, task]) - all_classes_matrix[-1, task]
            ),
        )
        for task in range(5)
    ]
    final_predictions = (test_poly[:, :weights.shape[0]] @ weights).argmax(axis=1)
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (test_labels.astype(np.int64), final_predictions), 1)
    return {
        "condition": condition, "seed": seed, "active_degree_by_task": active_degrees,
        "feature_count_by_task": feature_counts,
        "expansion_max_logit_changes": expansion_changes,
        "expansion_prediction_flips": expansion_prediction_flips,
        "accuracy_matrix": [
            [float(v) if np.isfinite(v) else None for v in row]
            for row in matrix
        ],
        "accuracy_matrix_all_classes_eligible": [
            [float(v) if np.isfinite(v) else None for v in row]
            for row in all_classes_matrix
        ],
        "final_mean_accuracy": float(np.nanmean(matrix[-1])),
        "final_current_accuracy": float(matrix[-1, -1]),
        "mean_forgetting": mean_forgetting(forgetting), "forgetting_by_task": forgetting,
        "mean_forgetting_definition": "mean over tasks with at least one subsequent evaluation; final task excluded",
        "mean_forgetting_all_classes_eligible": mean_forgetting(all_classes_forgetting),
        "forgetting_by_task_all_classes_eligible": all_classes_forgetting,
        "final_weights": weights.tolist(),
        "final_confusion_matrix": confusion.tolist(),
        "diagnostics_by_experience": diagnostics_by_experience,
        "masking_diagnostics_by_experience": masking_diagnostics_by_experience,
        "replay_memory_available_by_task": replay_memory_available_by_task,
        "replay_examples_in_batch_by_task": replay_examples_in_batch_by_task,
        "replay_example_exposures_by_task": replay_example_exposures_by_task,
        "replay_examples_by_class_in_batch_by_task": replay_examples_by_class_in_batch_by_task,
        "replay_stored_examples": int(sum(map(len, memory_indices.values()))),
        "replay_examples_by_class": {
            str(k): int(len(v)) for k, v in memory_indices.items()
        },
    }


def summarize_runs(runs):
    """Aggregate means and sample standard deviations for each condition.

    Forgetting aggregates exclude the final task because it has no later
    experience in which to forget; per-task values remain available in each run.
    """
    summary = {}
    for condition in CONDITIONS:
        subset = [row for row in runs if row["condition"] == condition]
        summary[condition] = {}
        for metric in (
            "final_mean_accuracy", "final_current_accuracy", "mean_forgetting",
            "mean_forgetting_all_classes_eligible",
        ):
            values = np.asarray([row[metric] for row in subset], dtype=np.float64)
            summary[condition][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
    return summary
