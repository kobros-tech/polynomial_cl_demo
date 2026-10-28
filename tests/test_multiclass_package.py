# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

from polynomial_cl.multiclass_data import TASK_ORDER, make_synthetic_2d
from polynomial_cl.multiclass_features import (
    MAX_DEGREE,
    expand_weights,
    make_projection,
    n_polynomial_features,
    polynomial_features_projected,
)


def test_synthetic_multiclass_data_is_reproducible_and_has_ten_classes():
    first = make_synthetic_2d(seed=7, n_train_per_class=3, n_test_per_class=4)
    second = make_synthetic_2d(seed=7, n_train_per_class=3, n_test_per_class=4)

    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
    x_train, y_train, x_test, y_test, means = first
    assert x_train.shape == (30, 2)
    assert x_test.shape == (40, 2)
    assert means.shape == (10, 2)
    np.testing.assert_array_equal(np.unique(y_train), np.arange(10))
    np.testing.assert_array_equal(np.unique(y_test), np.arange(10))
    assert tuple(class_id for pair in TASK_ORDER for class_id in pair) == tuple(range(10))


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_projected_polynomial_feature_count_and_finiteness(degree):
    x = np.array([[0.0, 1.0], [0.5, -0.25]], dtype=np.float32)
    features = polynomial_features_projected(x, degree, chunk_size=1)
    assert features.shape == (2, n_polynomial_features(2, degree))
    assert np.isfinite(features).all()


def test_zero_initialized_weight_expansion_preserves_logits():
    rng = np.random.default_rng(3)
    old_features = rng.normal(size=(12, 6)).astype(np.float32)
    old_weights = rng.normal(size=(6, 10)).astype(np.float32)
    new_features = np.concatenate(
        [old_features, np.zeros((12, 4), dtype=np.float32)], axis=1
    )
    expanded = expand_weights(old_weights, 10)
    np.testing.assert_allclose(old_features @ old_weights, new_features @ expanded)


def test_projection_is_reproducible_and_orthonormal():
    first = make_projection(seed=9, input_dim=12, n_components=4)
    second = make_projection(seed=9, input_dim=12, n_components=4)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first.T @ first, np.eye(4), atol=1e-6)


def test_package_runner_executes_a_small_class_incremental_run():
    from polynomial_cl.multiclass_experiment import make_task_plan, run_condition

    x_train, y_train, x_test, y_test, _ = make_synthetic_2d(
        seed=2, n_train_per_class=8, n_test_per_class=5
    )
    task_indices, memory_indices = make_task_plan(y_train, seed=4, memory_per_class=2)
    train_features = polynomial_features_projected(x_train, degree=MAX_DEGREE)
    test_features = polynomial_features_projected(x_test, degree=MAX_DEGREE)

    result = run_condition(
        "expanding",
        x_train,
        y_train,
        x_test,
        y_test,
        task_indices,
        memory_indices,
        seed=4,
        epochs=1,
        learning_rate=0.05,
        batch_size=16,
        precomputed_features=(train_features, test_features),
    )

    assert result["accuracy_matrix"][-1][-1] is not None
    assert len(result["accuracy_matrix"]) == 5
    assert result["feature_count_by_task"][-1] == n_polynomial_features(2, 3)
    assert result["replay_stored_examples"] == 20
    assert len(result["diagnostics_by_experience"]) == 5
