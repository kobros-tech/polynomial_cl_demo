# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

import gzip
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from polynomial_cl.multiclass_data import make_synthetic_2d
from polynomial_cl.multiclass_experiment import (
    CONDITIONS,
    evaluate_seen_tasks,
    make_task_plan,
    mean_forgetting,
    run_condition,
)


def test_synthetic_dataset_is_reproducible_and_has_independent_splits():
    first = make_synthetic_2d(4, n_train_per_class=5, n_test_per_class=7)
    second = make_synthetic_2d(4, n_train_per_class=5, n_test_per_class=7)
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
    x_train, y_train, x_test, y_test, means = first
    assert x_train.shape == (50, 2)
    assert x_test.shape == (70, 2)
    assert set(y_train) == set(range(10))
    assert set(y_test) == set(range(10))
    assert means.shape == (10, 2)
    assert not np.array_equal(x_train[:5], x_test[:5])


def test_task_plan_introduces_two_classes_and_replay_is_deterministic():
    labels = np.repeat(np.arange(10), 6)
    tasks_a, memory_a = make_task_plan(labels, seed=12, memory_per_class=2)
    tasks_b, memory_b = make_task_plan(labels, seed=12, memory_per_class=2)
    assert len(tasks_a) == 5
    for task, classes in zip(tasks_a, ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))):
        assert set(labels[task]) == set(classes)
    for class_id in range(10):
        np.testing.assert_array_equal(memory_a[class_id], memory_b[class_id])
        assert len(memory_a[class_id]) == 2
        assert np.all(labels[memory_a[class_id]] == class_id)


def test_evaluation_masks_unseen_classes():
    weights = np.zeros((1, 10), dtype=np.float32)
    weights[0, 9] = 100.0
    features = np.ones((4, 1), dtype=np.float32)
    labels = np.array([0, 1, 0, 1])
    masked = evaluate_seen_tasks(weights, features, labels, task=0, mask_unseen=True)
    unmasked = evaluate_seen_tasks(weights, features, labels, task=0, mask_unseen=False)
    # Class 9 wins every unmasked prediction, but is ineligible in experience 1.
    assert masked[0] == pytest.approx(0.5)
    assert unmasked[0] == pytest.approx(0.0)
    assert np.isnan(masked[1:]).all()
    assert np.isnan(unmasked[1:]).all()


def test_all_conditions_run_with_common_matrix_and_expansion_invariant():
    x_train, y_train, x_test, y_test, _ = make_synthetic_2d(
        9, n_train_per_class=4, n_test_per_class=5, separation=3.0, noise=0.2
    )
    tasks, memory = make_task_plan(y_train, seed=2, memory_per_class=1)
    results = {
        condition: run_condition(
            condition, x_train, y_train, x_test, y_test, tasks, memory,
            seed=2, epochs=1, learning_rate=0.01, batch_size=8,
        )
        for condition in CONDITIONS
    }
    assert results["fixed_linear"]["feature_count_by_task"] == [3] * 5
    assert results["expanding"]["feature_count_by_task"] == [3, 6, 10, 10, 10]
    assert results["fixed_capacity"]["feature_count_by_task"] == [10] * 5
    for result in results.values():
        matrix = result["accuracy_matrix"]
        assert len(matrix) == 5
        assert all(len(row) == 5 for row in matrix)
        all_classes = result["accuracy_matrix_all_classes_eligible"]
        assert len(all_classes) == 5
        assert all(len(row) == 5 for row in all_classes)
        assert all(
            matrix[row][col] is None
            for row in range(5)
            for col in range(row + 1, 5)
        )
        assert np.isfinite(result["final_mean_accuracy"])
        assert np.isfinite(result["mean_forgetting"])
        assert np.isfinite(result["mean_forgetting_all_classes_eligible"])
        confusion = np.asarray(result["final_confusion_matrix"])
        assert confusion.shape == (10, 10)
        assert confusion.sum() == len(y_test)
    assert len(results["expanding"]["expansion_max_logit_changes"]) == 2
    assert max(results["expanding"]["expansion_max_logit_changes"]) <= 1e-5
    assert results["expanding"]["expansion_prediction_flips"] == [0, 0]



def test_shared_runner_returns_versioned_protocol_and_seed_summary():
    from experiments.continual_comparison import run_experiment

    args = SimpleNamespace(
        dataset="synthetic_2d",
        data_seed=3,
        synthetic_train_per_class=4,
        synthetic_test_per_class=5,
        class_separation=3.0,
        class_noise=0.2,
        seeds=[0],
        memory_per_class=1,
        epochs=1,
        lr=0.01,
        batch_size=8,
    )
    result = run_experiment(args)
    assert result["protocol_version"] == "class-incremental-v1"
    assert result["dataset"] == "synthetic_2d"
    assert len(result["runs"]) == len(CONDITIONS)
    assert set(result["summary"]) == set(CONDITIONS)
    for run in result["runs"]:
        assert run["final_confusion_matrix"]
        assert run["accuracy_matrix_all_classes_eligible"]
        assert len(run["diagnostics_by_experience"]) == 5
        assert len(run["masking_diagnostics_by_experience"]) == 5
        assert run["replay_examples_in_batch_by_task"] == [0, 2, 4, 6, 8]
        assert run["replay_example_exposures_by_task"] == [0, 2, 4, 6, 8]
        assert len(run["replay_examples_by_class_in_batch_by_task"]) == 5
        if run["condition"] == "expanding":
            assert len(run["expansion_prediction_flips"]) == 2
            assert all(flips == 0 for flips in run["expansion_prediction_flips"])
        else:
            assert run["expansion_prediction_flips"] == []



def _write_tiny_mnist(root):
    rng = np.random.default_rng(71)
    train_labels = np.repeat(np.arange(10, dtype=np.uint8), 2)
    test_labels = np.arange(10, dtype=np.uint8)
    train_images = rng.integers(0, 256, size=(20, 28, 28), dtype=np.uint8)
    test_images = rng.integers(0, 256, size=(10, 28, 28), dtype=np.uint8)

    def write_images(name, images):
        payload = struct.pack(">IIII", 2051, len(images), 28, 28) + images.tobytes()
        with gzip.open(root / name, "wb") as stream:
            stream.write(payload)

    def write_labels(name, labels):
        payload = struct.pack(">II", 2049, len(labels)) + labels.tobytes()
        with gzip.open(root / name, "wb") as stream:
            stream.write(payload)

    root.mkdir(parents=True, exist_ok=True)
    write_images("train-images-idx3-ubyte.gz", train_images)
    write_labels("train-labels-idx1-ubyte.gz", train_labels)
    write_images("t10k-images-idx3-ubyte.gz", test_images)
    write_labels("t10k-labels-idx1-ubyte.gz", test_labels)


def test_shared_runner_runs_split_mnist_end_to_end_without_network(tmp_path):
    from experiments.continual_comparison import run_experiment

    _write_tiny_mnist(tmp_path)
    args = SimpleNamespace(
        dataset="split_mnist",
        data_dir=str(tmp_path),
        projection_seed=4,
        data_seed=0,
        synthetic_train_per_class=4,
        synthetic_test_per_class=5,
        class_separation=3.0,
        class_noise=0.2,
        seeds=[0],
        memory_per_class=1,
        epochs=1,
        lr=0.01,
        batch_size=8,
    )

    result = run_experiment(args)

    assert result["dataset"] == "split_mnist"
    assert result["protocol_version"] == "class-incremental-v1"
    assert len(result["runs"]) == len(CONDITIONS)
    for run in result["runs"]:
        assert len(run["accuracy_matrix"]) == 5
        assert len(run["diagnostics_by_experience"]) == 5
        assert len(run["masking_diagnostics_by_experience"]) == 5
        assert run["final_confusion_matrix"]
        assert run["replay_examples_in_batch_by_task"] == [0, 2, 4, 6, 8]
        assert run["expansion_prediction_flips"] == (
            [0, 0] if run["condition"] == "expanding" else []
        )


def test_forgetting_aggregate_excludes_final_task_but_keeps_per_task_values():
    from polynomial_cl.multiclass_experiment import summarize_runs

    # The last task has no opportunity to forget, so its zero must not dilute
    # the aggregate. All five task-level values remain available.
    forgetting = [0.8, 0.6, 0.4, 0.2, 0.0]
    assert mean_forgetting(forgetting) == pytest.approx(0.5)
    assert mean_forgetting([0.4, 0.2, 0.0]) == pytest.approx(0.3)
    runs = []
    for condition in CONDITIONS:
        runs.append({
            "condition": condition,
            "final_mean_accuracy": 0.5,
            "final_current_accuracy": 0.5,
            "mean_forgetting": mean_forgetting(forgetting),
            "mean_forgetting_all_classes_eligible": mean_forgetting(forgetting),
            "forgetting_by_task": forgetting,
            "forgetting_by_task_all_classes_eligible": forgetting,
        })
    summary = summarize_runs(runs)
    assert summary["expanding"]["mean_forgetting"]["mean"] == pytest.approx(0.5)


def test_masking_prediction_diagnostics_reports_changed_predictions():
    from polynomial_cl.multiclass_experiment import masking_prediction_diagnostics

    weights = np.zeros((1, 10), dtype=np.float32)
    weights[0, 9] = 100.0
    features = np.ones((4, 1), dtype=np.float32)
    labels = np.array([0, 1, 0, 1])

    result = masking_prediction_diagnostics(weights, features, labels, task=0)

    assert result["n_test_examples"] == 4
    assert result["n_predictions_changed"] == 4
    assert result["prediction_change_rate"] == pytest.approx(1.0)
    assert result["accuracy_masked"] == pytest.approx(0.5)
    assert result["accuracy_unmasked"] == pytest.approx(0.0)


def test_masking_prediction_diagnostics_reports_no_changes_when_all_classes_eligible():
    from polynomial_cl.multiclass_experiment import masking_prediction_diagnostics

    weights = np.zeros((1, 10), dtype=np.float32)
    weights[0, 9] = 100.0
    features = np.ones((4, 1), dtype=np.float32)
    labels = np.array([0, 1, 8, 9])

    result = masking_prediction_diagnostics(weights, features, labels, task=4)

    assert result["n_predictions_changed"] == 0
    assert result["prediction_change_rate"] == pytest.approx(0.0)
    assert result["accuracy_masked"] == pytest.approx(result["accuracy_unmasked"])



def test_load_mnist_no_download_reports_missing_files_without_fetch(tmp_path, monkeypatch):
    from polynomial_cl import mnist

    def unexpected_fetch(*args, **kwargs):
        raise AssertionError("fetch must not be called when downloads are disabled")

    monkeypatch.setattr(mnist, "fetch", unexpected_fetch)
    with pytest.raises(FileNotFoundError, match="download is disabled") as error:
        mnist.load_mnist(tmp_path, download=False)

    for filename in mnist.FILES.values():
        assert filename in str(error.value)


def test_load_mnist_no_download_loads_existing_archives(tmp_path):
    from polynomial_cl.mnist import load_mnist

    _write_tiny_mnist(tmp_path)
    x_train, y_train, x_test, y_test = load_mnist(tmp_path, download=False)

    assert x_train.shape == (20, 784)
    assert y_train.shape == (20,)
    assert x_test.shape == (10, 784)
    assert y_test.shape == (10,)
