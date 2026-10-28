# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Legacy Split MNIST ablation using a global ten-class training/evaluation head.

Prefer experiments.continual_comparison for new class-incremental results. This
entry point remains available to reproduce the historical protocol; it deliberately
keeps all ten output classes eligible and writes to a separate legacy-named file by
default so it cannot overwrite the committed historical result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.polynomial_feature_utils import (
    CONDITIONS,
    MAX_DEGREE,
    N_COMPONENTS,
    expand_weights,
    make_projection,
    monomial_exponents,
    n_polynomial_features,
    polynomial_features_projected,
    project_images,
)
from experiments.split_mnist_utils import load_mnist
from polynomial_cl.multiclass import fit_task

DEFAULT_OUTPUT = Path("results/split_mnist_expansion_legacy.json")


def make_task_plan(
    labels: np.ndarray, seed: int, memory_per_class: int
) -> tuple[list[np.ndarray], dict[int, np.ndarray]]:
    """Create identical task order and per-class replay indices for every condition."""
    rng = np.random.default_rng(seed)
    task_indices = []
    memory_indices = {}
    for task in range(5):
        classes = (2 * task, 2 * task + 1)
        indices = np.flatnonzero(np.isin(labels, classes))
        task_indices.append(indices[rng.permutation(len(indices))])
        for digit in classes:
            candidates = np.flatnonzero(labels == digit)
            chosen = rng.choice(
                candidates, min(memory_per_class, len(candidates)), replace=False
            )
            memory_indices[digit] = chosen
    return task_indices, memory_indices


def evaluate(
    weights: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    task: int,
) -> np.ndarray:
    """Return accuracy on each task seen so far, using the shared global 10-way head."""
    results = np.full(5, np.nan, dtype=np.float64)
    for previous_task in range(task + 1):
        classes = (2 * previous_task, 2 * previous_task + 1)
        indices = np.flatnonzero(np.isin(test_labels, classes))
        predictions = (test_features[indices] @ weights).argmax(axis=1)
        results[previous_task] = np.mean(predictions == test_labels[indices])
    return results


def run_condition(
    condition: str,
    train_projected: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    task_indices: list[np.ndarray],
    memory_indices: dict[int, np.ndarray],
    *,
    seed: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
) -> dict:
    """Run one condition using the same task/replay indices as the other conditions."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    active_degrees = {
        "fixed_linear": [1] * 5,
        "expanding": [1, 2, 3, 3, 3],
        "fixed_capacity": [MAX_DEGREE] * 5,
    }[condition]
    weights = np.zeros((n_polynomial_features(N_COMPONENTS, active_degrees[0]), 10), dtype=np.float32)
    accuracy_matrix = np.full((5, 5), np.nan, dtype=np.float64)
    feature_counts = []
    expansion_max_logit_changes = []

    for task, indices in enumerate(task_indices):
        degree = active_degrees[task]
        active_count = n_polynomial_features(N_COMPONENTS, degree)
        old_count = weights.shape[0]
        if active_count > old_count:
            probe = test_features[: min(256, len(test_features))]
            old_logits = probe[:, :old_count] @ weights
            weights = expand_weights(weights, active_count)
            new_logits = probe[:, :active_count] @ weights
            max_change = float(np.max(np.abs(new_logits - old_logits)))
            if max_change > 1e-5:
                raise RuntimeError(
                    f"zero-initialized expansion changed old logits by {max_change}"
                )
            expansion_max_logit_changes.append(max_change)
        else:
            weights = expand_weights(weights, active_count)
        feature_counts.append(active_count)

        current_x = polynomial_features_projected(
            train_projected[indices], MAX_DEGREE
        )[:, :active_count]
        current_y = train_labels[indices].astype(np.int64, copy=False)
        old_indices = [
            memory_indices[digit]
            for digit in range(2 * task)
        ]
        if old_indices:
            replay_idx = np.concatenate(old_indices)
            replay_x = polynomial_features_projected(
                train_projected[replay_idx], MAX_DEGREE
            )[:, :active_count]
            replay_y = train_labels[replay_idx].astype(np.int64, copy=False)
        else:
            replay_x = np.empty((0, active_count), dtype=np.float32)
            replay_y = np.empty(0, dtype=np.int64)

        weights = fit_task(
            weights,
            current_x,
            current_y,
            replay_x,
            replay_y,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            seed=seed + task,
        )
        task_accuracy = evaluate(
            weights,
            test_features[:, :active_count],
            test_labels,
            task,
        )
        accuracy_matrix[task, : task + 1] = task_accuracy[: task + 1]
        print(
            f"seed={seed} condition={condition} task={task + 1}/5 "
            f"degree={degree} features={active_count} "
            f"current_acc={accuracy_matrix[task, task]:.4f} "
            f"mean_seen_acc={np.nanmean(accuracy_matrix[task, : task + 1]):.4f}"
        )

    forgetting = [
        max(0.0, float(np.nanmax(accuracy_matrix[:, task]) - accuracy_matrix[-1, task]))
        for task in range(5)
    ]
    return {
        "condition": condition,
        "seed": seed,
        "active_degree_by_task": active_degrees,
        "feature_count_by_task": feature_counts,
        "expansion_max_logit_changes": expansion_max_logit_changes,
        "accuracy_matrix": [
            [float(value) if np.isfinite(value) else None for value in row]
            for row in accuracy_matrix
        ],
        "final_mean_accuracy": float(np.nanmean(accuracy_matrix[-1])),
        "final_current_accuracy": float(accuracy_matrix[-1, -1]),
        "mean_forgetting": float(np.mean(forgetting)),
        "forgetting_by_task": forgetting,
    }


def summarize_runs(runs: list[dict]) -> dict:
    """Summarize each condition over seeds with mean and sample standard deviation."""
    summary = {}
    metrics = (
        "final_mean_accuracy",
        "final_current_accuracy",
        "mean_forgetting",
    )
    for condition in CONDITIONS:
        subset = [row for row in runs if row["condition"] == condition]
        summary[condition] = {}
        for metric in metrics:
            values = np.asarray([row[metric] for row in subset], dtype=np.float64)
            summary[condition][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
    return summary


def run_experiment(args) -> dict:
    x_train, y_train, x_test, y_test = load_mnist(Path(args.data_dir))
    projection = make_projection(args.projection_seed)
    train_projected = project_images(x_train, projection)
    test_projected = project_images(x_test, projection)
    test_features = polynomial_features_projected(test_projected, MAX_DEGREE)
    runs = []
    for seed in args.seeds:
        task_indices, memory_indices = make_task_plan(
            y_train, seed, args.memory_per_class
        )
        for condition in CONDITIONS:
            runs.append(
                run_condition(
                    condition,
                    train_projected,
                    y_train,
                    test_features,
                    y_test,
                    task_indices,
                    memory_indices,
                    seed=seed,
                    epochs=args.epochs,
                    learning_rate=args.lr,
                    batch_size=args.batch_size,
                )
            )
    return {
        "benchmark": "Legacy Split MNIST projected polynomial-feature expansion ablation",
        "protocol_name": "legacy-global-ten-class-evaluation",
        "protocol": {
            "seeds": args.seeds,
            "epochs_per_task": args.epochs,
            "memory_per_class": args.memory_per_class,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "projection_seed": args.projection_seed,
            "projection_dimensions": N_COMPONENTS,
            "maximum_degree": MAX_DEGREE,
            "feature_counts_by_degree": {
                str(degree): n_polynomial_features(N_COMPONENTS, degree)
                for degree in range(1, MAX_DEGREE + 1)
            },
            "task_order": [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
            "replay": "same seeded per-class training indices for all conditions",
            "conditions": {
                "fixed_linear": "degree 1 for all five tasks",
                "expanding": "degree 1, then 2, then 3 for tasks 3-5; new weights zero-initialized",
                "fixed_capacity": "degree 3 available from the first task",
            },
        },
        "runs": runs,
        "summary": summarize_runs(runs),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/mnist")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--memory-per-class", type=int, default=200)
    parser.add_argument("--projection-seed", type=int, default=1729)
    args = parser.parse_args()
    if (
        not args.seeds
        or args.epochs < 1
        or args.batch_size < 1
        or args.memory_per_class < 1
    ):
        parser.error("seeds must be non-empty; epochs, batch-size, and memory-per-class must be positive")

    payload = run_experiment(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
