# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Independently audit historical and corrected Split MNIST result files.

This diagnostic deliberately separates what a saved accuracy matrix proves from
what requires a rerun: legacy summaries cannot recover class eligibility,
per-class confusions, replay contents, or expansion-time predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _matrix(value: object, name: str) -> np.ndarray:
    matrix = np.asarray(
        [[np.nan if cell is None else float(cell) for cell in row] for row in value],
        dtype=np.float64,
    )
    if matrix.shape != (5, 5):
        raise ValueError(f"{name} must be a 5x5 accuracy matrix")
    if np.isinf(matrix).any() or np.any((matrix < 0) | (matrix > 1)):
        raise ValueError(f"{name} must contain accuracies in [0, 1] or null")
    for row in range(5):
        if not np.isfinite(matrix[row, row]):
            raise ValueError(f"{name} is missing diagonal accuracy at experience {row + 1}")
        if np.isfinite(matrix[row, row + 1 :]).any():
            raise ValueError(f"{name} contains values for tasks not yet introduced")
        if not np.isfinite(matrix[row, : row + 1]).all():
            raise ValueError(f"{name} is missing an introduced-task accuracy")
    return matrix


def _metrics(matrix: np.ndarray) -> dict:
    """Recompute final mean accuracy and max-to-final forgetting from a matrix."""
    final = matrix[-1]
    forgetting_by_task = []
    best_accuracy_by_task = []
    best_experience_by_task = []
    for task in range(5):
        history = matrix[task:, task]
        best_offset = int(np.argmax(history))
        best = float(history[best_offset])
        best_accuracy_by_task.append(best)
        best_experience_by_task.append(task + best_offset + 1)
        forgetting_by_task.append(max(0.0, best - float(final[task])))
    return {
        "final_mean_accuracy": float(np.mean(final)),
        "final_current_accuracy": float(final[-1]),
        "mean_forgetting": float(np.mean(forgetting_by_task)),
        "forgetting_by_task": forgetting_by_task,
        "best_accuracy_by_task": best_accuracy_by_task,
        "best_experience_by_task": best_experience_by_task,
    }


def _matches(recorded: object, recomputed: float, atol: float = 1e-8) -> bool:
    return recorded is not None and bool(
        np.isclose(float(recorded), recomputed, rtol=0.0, atol=atol)
    )


def audit_historical_result(payload: dict, source: str = "<memory>") -> dict:
    """Audit a legacy one-seed result whose matrix contains all task accuracies."""
    matrix = _matrix(payload.get("accuracy_matrix"), "accuracy_matrix")
    metrics = _metrics(matrix)
    checks = {
        key: _matches(payload.get(key), metrics[key])
        for key in ("final_mean_accuracy", "mean_forgetting")
    }
    return {
        "source": source,
        "schema": "legacy_split_mnist_seed",
        "seed": payload.get("seed"),
        "reported_metrics": {
            key: payload.get(key) for key in checks
        },
        "recomputed_metrics": metrics,
        "reported_metrics_match_matrix": checks,
        "matrix_consistent": all(checks.values()),
        "historical_protocol_limitations": [
            "No explicit class-eligibility rule is stored with this result.",
            "No per-class predictions or confusion matrices are stored.",
            "No replay samples/counts or expansion-time logits are stored.",
            "The matrix alone cannot identify the causal source of forgetting.",
        ],
    }


def audit_comparison_result(payload: dict, source: str = "<memory>") -> dict:
    """Audit corrected shared-runner output, including diagnostic matrices."""
    if payload.get("protocol_version") != "class-incremental-v1":
        raise ValueError("unsupported or missing comparison protocol_version")
    audits = []
    for run in payload.get("runs", []):
        primary = _matrix(run.get("accuracy_matrix"), "accuracy_matrix")
        all_eligible = _matrix(
            run.get("accuracy_matrix_all_classes_eligible"),
            "accuracy_matrix_all_classes_eligible",
        )
        primary_metrics = _metrics(primary)
        all_metrics = _metrics(all_eligible)
        checks = {
            "final_mean_accuracy": _matches(
                run.get("final_mean_accuracy"), primary_metrics["final_mean_accuracy"]
            ),
            "final_current_accuracy": _matches(
                run.get("final_current_accuracy"), primary_metrics["final_current_accuracy"]
            ),
            "mean_forgetting": _matches(
                run.get("mean_forgetting"), primary_metrics["mean_forgetting"]
            ),
            "mean_forgetting_all_classes_eligible": _matches(
                run.get("mean_forgetting_all_classes_eligible"),
                all_metrics["mean_forgetting"],
            ),
        }
        confusion = np.asarray(run.get("final_confusion_matrix"), dtype=np.int64)
        if confusion.shape != (10, 10) or np.any(confusion < 0):
            raise ValueError("final_confusion_matrix must be a non-negative 10x10 matrix")
        experience_diagnostics = run.get("diagnostics_by_experience", [])
        replay_counts = run.get("replay_examples_in_batch_by_task", [])
        replay_exposures = run.get("replay_example_exposures_by_task", [])
        if experience_diagnostics:
            if len(experience_diagnostics) != 5:
                raise ValueError("diagnostics_by_experience must contain five entries")
            for index, diagnostic in enumerate(experience_diagnostics):
                expected_classes = list(range(2 * (index + 1)))
                if diagnostic.get("eligible_classes") != expected_classes:
                    raise ValueError("diagnostic eligible classes do not match protocol")
                class_accuracy = diagnostic.get("class_accuracy", {})
                if sorted(map(int, class_accuracy)) != expected_classes:
                    raise ValueError("per-class accuracy is missing an eligible class")
                task_confusion = np.asarray(diagnostic.get("confusion_matrix"))
                if task_confusion.shape != (10, 10):
                    raise ValueError("per-experience confusion matrix must be 10x10")
                if int(task_confusion.sum()) != int(diagnostic.get("evaluated_examples", -1)):
                    raise ValueError("per-experience confusion count disagrees with evaluated examples")
        if replay_counts and (len(replay_counts) != 5 or replay_counts[0] != 0):
            raise ValueError("replay batch counts must have five entries and start at zero")
        if replay_exposures and (
            len(replay_exposures) != 5 or replay_exposures[0] != 0
        ):
            raise ValueError("replay exposure counts must have five entries and start at zero")
        audits.append({
            "condition": run.get("condition"),
            "seed": run.get("seed"),
            "reported_metrics_match_matrices": checks,
            "matrix_consistent": all(checks.values()),
            "recomputed_primary_metrics": primary_metrics,
            "recomputed_all_classes_eligible_metrics": all_metrics,
            "final_confusion_examples": int(confusion.sum()),
            "per_experience_diagnostics_present": len(experience_diagnostics) == 5,
            "replay_examples_in_batch_by_task": replay_counts,
            "replay_example_exposures_by_task": replay_exposures,
            "expansion_max_logit_changes": run.get("expansion_max_logit_changes", []),
            "expansion_prediction_flips": run.get("expansion_prediction_flips", []),
            "expansion_predictions_unchanged": all(
                flips == 0 for flips in run.get("expansion_prediction_flips", [])
            ),
        })
    return {
        "source": source,
        "schema": "class-incremental-v1",
        "dataset": payload.get("dataset"),
        "run_count": len(audits),
        "all_runs_matrix_consistent": all(item["matrix_consistent"] for item in audits),
        "runs": audits,
    }


def audit_file(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_version") == "class-incremental-v1":
        return audit_comparison_result(payload, str(path))
    if "accuracy_matrix" in payload:
        return audit_historical_result(payload, str(path))
    if isinstance(payload.get("runs"), list):
        # The historical expansion ablation stores multiple condition/seed runs.
        audits = [
            audit_historical_result(run, f"{path}#runs[{index}]")
            for index, run in enumerate(payload["runs"])
        ]
        return {
            "source": str(path),
            "schema": "legacy_expansion_ablation",
            "run_count": len(audits),
            "all_runs_matrix_consistent": all(item["matrix_consistent"] for item in audits),
            "runs": audits,
        }
    raise ValueError(f"unrecognized result schema: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results",
        nargs="+",
        type=Path,
        help="Historical or corrected JSON result files to audit.",
    )
    parser.add_argument("--output", type=Path, help="Optional path for the audit JSON.")
    args = parser.parse_args()
    report = {"audits": [audit_file(path) for path in args.results]}
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
