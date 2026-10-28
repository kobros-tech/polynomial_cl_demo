# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

from experiments.diagnose_split_mnist import (
    _metrics,
    audit_comparison_result,
    audit_historical_result,
)


def accuracy_matrix():
    return [
        [0.90, None, None, None, None],
        [0.80, 0.85, None, None, None],
        [0.70, 0.80, 0.88, None, None],
        [0.65, 0.75, 0.80, 0.90, None],
        [0.60, 0.70, 0.75, 0.80, 0.92],
    ]


def test_historical_metrics_are_recomputed_from_matrix():
    matrix = np.asarray([
        [np.nan if value is None else value for value in row]
        for row in accuracy_matrix()
    ])
    metrics = _metrics(matrix)
    payload = {
        "seed": 2,
        "accuracy_matrix": accuracy_matrix(),
        "final_mean_accuracy": metrics["final_mean_accuracy"],
        "mean_forgetting": metrics["mean_forgetting"],
    }

    audit = audit_historical_result(payload, "seed2.json")

    assert audit["matrix_consistent"] is True
    assert audit["recomputed_metrics"]["final_current_accuracy"] == pytest.approx(0.92)
    assert audit["recomputed_metrics"]["best_experience_by_task"] == [1, 2, 3, 4, 5]
    assert len(audit["historical_protocol_limitations"]) >= 3


def test_historical_audit_detects_stale_summary_metric():
    payload = {
        "accuracy_matrix": accuracy_matrix(),
        "final_mean_accuracy": 0.1,
        "mean_forgetting": 0.2,
    }

    audit = audit_historical_result(payload)

    assert audit["matrix_consistent"] is False
    assert audit["reported_metrics_match_matrix"]["final_mean_accuracy"] is False
    assert audit["reported_metrics_match_matrix"]["mean_forgetting"] is False


def test_corrected_audit_checks_both_matrices_and_per_experience_diagnostics():
    matrix = accuracy_matrix()
    numeric = np.asarray([
        [np.nan if value is None else value for value in row]
        for row in matrix
    ])
    metrics = _metrics(numeric)
    run = {
        "condition": "expanding",
        "seed": 0,
        "accuracy_matrix": matrix,
        "accuracy_matrix_all_classes_eligible": matrix,
        "final_mean_accuracy": metrics["final_mean_accuracy"],
        "final_current_accuracy": metrics["final_current_accuracy"],
        "mean_forgetting": metrics["mean_forgetting"],
        "mean_forgetting_all_classes_eligible": metrics["mean_forgetting"],
        "final_confusion_matrix": np.eye(10, dtype=int).tolist(),
        "diagnostics_by_experience": [
            {
                "eligible_classes": list(range(2 * (experience + 1))),
                "class_accuracy": {
                    str(class_id): 0.5
                    for class_id in range(2 * (experience + 1))
                },
                "confusion_matrix": np.zeros((10, 10), dtype=int).tolist(),
                "evaluated_examples": 0,
            }
            for experience in range(5)
        ],
        "replay_examples_in_batch_by_task": [0, 4, 8, 12, 16],
        "replay_example_exposures_by_task": [0, 4, 8, 12, 16],
        "expansion_max_logit_changes": [0.0, 1e-6],
    }

    audit = audit_comparison_result(
        {"protocol_version": "class-incremental-v1", "dataset": "split_mnist", "runs": [run]}
    )

    assert audit["all_runs_matrix_consistent"] is True
    assert audit["runs"][0]["per_experience_diagnostics_present"] is True
    assert audit["runs"][0]["final_confusion_examples"] == 10
    assert audit["runs"][0]["replay_examples_in_batch_by_task"] == [0, 4, 8, 12, 16]


def test_corrected_audit_rejects_wrong_class_eligibility():
    run = {
        "condition": "fixed_linear",
        "seed": 0,
        "accuracy_matrix": accuracy_matrix(),
        "accuracy_matrix_all_classes_eligible": accuracy_matrix(),
        "final_mean_accuracy": 0.754,
        "final_current_accuracy": 0.92,
        "mean_forgetting": 0.136,
        "mean_forgetting_all_classes_eligible": 0.136,
        "final_confusion_matrix": np.zeros((10, 10), dtype=int).tolist(),
        "diagnostics_by_experience": [
            {
                "eligible_classes": [0] if experience == 0 else list(range(2 * (experience + 1))),
                "class_accuracy": {
                    str(class_id): 1.0
                    for class_id in ([0] if experience == 0 else range(2 * (experience + 1)))
                },
                "confusion_matrix": np.zeros((10, 10), dtype=int).tolist(),
            }
            for experience in range(5)
        ],
        "replay_examples_in_batch_by_task": [0, 4, 8, 12, 16],
    }
    with pytest.raises(ValueError, match="eligible classes"):
        audit_comparison_result(
            {"protocol_version": "class-incremental-v1", "runs": [run]}
        )
