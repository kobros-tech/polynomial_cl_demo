# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np
import pytest

from experiments.split_mnist_expansion import (
    CONDITIONS,
    DEFAULT_OUTPUT,
    expand_weights,
    make_projection,
    make_task_plan,
    monomial_exponents,
    n_polynomial_features,
    polynomial_features_projected,
    project_images,
    summarize_runs,
)


def test_polynomial_feature_counts_and_prefix_order():
    assert n_polynomial_features(16, 1) == 17
    assert n_polynomial_features(16, 2) == 153
    assert n_polynomial_features(16, 3) == 969
    degree_one = monomial_exponents(4, 1)
    degree_two = monomial_exponents(4, 2)
    np.testing.assert_array_equal(degree_two[: len(degree_one)], degree_one)


def test_polynomial_features_are_finite_and_prefix_stable():
    projected = np.random.default_rng(4).normal(size=(7, 4)).astype(np.float32)
    linear = polynomial_features_projected(projected, 1)
    cubic = polynomial_features_projected(projected, 3)
    assert linear.shape == (7, n_polynomial_features(4, 1))
    assert cubic.shape == (7, n_polynomial_features(4, 3))
    assert np.isfinite(cubic).all()
    np.testing.assert_array_equal(cubic[:, : linear.shape[1]], linear)


def test_projection_is_deterministic_and_orthonormal():
    first = make_projection(seed=12, input_dim=20, n_components=5)
    second = make_projection(seed=12, input_dim=20, n_components=5)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first.T @ first, np.eye(5), atol=1e-6)
    images = np.zeros((3, 20), dtype=np.float32)
    assert project_images(images, first).shape == (3, 5)


def test_zero_initialised_expansion_preserves_existing_logits():
    rng = np.random.default_rng(5)
    old_features = rng.normal(size=(11, 17)).astype(np.float32)
    old_weights = rng.normal(size=(17, 10)).astype(np.float32)
    before = old_features @ old_weights
    expanded = expand_weights(old_weights, 153)
    new_features = rng.normal(size=(11, 153)).astype(np.float32)
    new_features[:, :17] = old_features
    np.testing.assert_array_equal(expanded[:17], old_weights)
    np.testing.assert_array_equal(expanded[17:], 0.0)
    np.testing.assert_allclose(new_features @ expanded, before, atol=1e-6)


def test_expansion_cannot_shrink_weights():
    with pytest.raises(ValueError, match="cannot shrink"):
        expand_weights(np.zeros((10, 2)), 9)


def test_task_and_memory_plan_is_reproducible_and_complete():
    labels = np.repeat(np.arange(10), 8)
    first_tasks, first_memory = make_task_plan(labels, seed=3, memory_per_class=2)
    second_tasks, second_memory = make_task_plan(labels, seed=3, memory_per_class=2)
    assert len(first_tasks) == 5
    assert set(first_memory) == set(range(10))
    for first, second in zip(first_tasks, second_tasks):
        np.testing.assert_array_equal(first, second)
    for digit in range(10):
        np.testing.assert_array_equal(first_memory[digit], second_memory[digit])
        assert len(first_memory[digit]) == 2
        assert np.all(labels[first_memory[digit]] == digit)


def test_summary_reports_mean_and_sample_std():
    runs = []
    for condition, accuracy, forgetting in (
        ("fixed_linear", 0.6, 0.2),
        ("fixed_linear", 0.8, 0.1),
        ("expanding", 0.7, 0.15),
        ("expanding", 0.9, 0.05),
        ("fixed_capacity", 0.75, 0.12),
        ("fixed_capacity", 0.85, 0.08),
    ):
        runs.append(
            {
                "condition": condition,
                "final_mean_accuracy": accuracy,
                "final_current_accuracy": accuracy,
                "mean_forgetting": forgetting,
            }
        )
    summary = summarize_runs(runs)
    assert set(summary) == set(CONDITIONS)
    assert summary["fixed_linear"]["final_mean_accuracy"]["mean"] == pytest.approx(0.7)
    assert summary["fixed_linear"]["final_mean_accuracy"]["std"] == pytest.approx(
        np.std([0.6, 0.8], ddof=1)
    )



def test_all_split_mnist_conditions_run_on_tiny_paired_data():
    from experiments.split_mnist_expansion import run_condition

    rng = np.random.default_rng(22)
    train_projected = rng.normal(size=(20, 16)).astype(np.float32)
    test_projected = rng.normal(size=(20, 16)).astype(np.float32)
    labels = np.repeat(np.arange(10), 2)
    task_indices, memory_indices = make_task_plan(
        labels, seed=7, memory_per_class=1
    )
    test_features = polynomial_features_projected(test_projected, 3)

    results = {}
    for condition in CONDITIONS:
        results[condition] = run_condition(
            condition,
            train_projected,
            labels,
            test_features,
            labels,
            task_indices,
            memory_indices,
            seed=7,
            epochs=1,
            learning_rate=0.01,
            batch_size=4,
        )

    assert results["fixed_linear"]["feature_count_by_task"] == [17] * 5
    assert results["expanding"]["feature_count_by_task"] == [17, 153, 969, 969, 969]
    assert results["fixed_capacity"]["feature_count_by_task"] == [969] * 5
    for result in results.values():
        assert len(result["accuracy_matrix"]) == 5
        assert np.isfinite(result["final_mean_accuracy"])
        assert np.isfinite(result["mean_forgetting"])
    assert len(results["expanding"]["expansion_max_logit_changes"]) == 2
    assert max(results["expanding"]["expansion_max_logit_changes"]) <= 1e-5



def test_legacy_runner_default_cannot_overwrite_historical_results():
    assert DEFAULT_OUTPUT == Path("results/split_mnist_expansion_legacy.json")
    assert DEFAULT_OUTPUT != Path("results/split_mnist_expansion.json")
