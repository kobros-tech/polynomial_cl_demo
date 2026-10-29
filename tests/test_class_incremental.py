# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

from polynomial_cl.class_incremental import (
    CONDITIONS,
    TASK_ORDER,
    make_feature_data,
    make_synthetic_2d,
    make_task_plan,
    run_comparison,
    run_condition,
    summarize_runs,
)
from polynomial_cl.class_incremental import experiment as experiment_module
from polynomial_cl.class_incremental.data import make_projection, project_images
from polynomial_cl.class_incremental.experiment import accuracy_by_task
from polynomial_cl.class_incremental.features import (
    expand_weights,
    monomial_exponents,
    n_polynomial_features,
    scaled_polynomial_features,
)
from polynomial_cl.class_incremental.training import fit_task


# ---------------------------------------------------------------- data
def test_synthetic_data_is_reproducible_with_independent_splits():
    first = make_synthetic_2d(4, n_train_per_class=5, n_test_per_class=7)
    second = make_synthetic_2d(4, n_train_per_class=5, n_test_per_class=7)
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
    x_train, y_train, x_test, y_test, means = first
    assert (
        x_train.shape == (50, 2) and x_test.shape == (70, 2) and means.shape == (10, 2)
    )
    assert set(y_train) == set(y_test) == set(range(10))
    assert not np.array_equal(x_train[:5], x_test[:5])


@pytest.mark.parametrize(
    "kwargs", [{"n_train_per_class": 0}, {"separation": 0.0}, {"noise": float("nan")}]
)
def test_synthetic_data_rejects_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        make_synthetic_2d(0, **kwargs)


def test_tasks_introduce_every_class_exactly_once():
    assert [c for pair in TASK_ORDER for c in pair] == list(range(10))


def test_projection_is_deterministic_and_orthonormal():
    first = make_projection(seed=9, input_dim=12, n_components=4)
    np.testing.assert_array_equal(
        first, make_projection(seed=9, input_dim=12, n_components=4)
    )
    np.testing.assert_allclose(first.T @ first, np.eye(4), atol=1e-6)
    assert project_images(np.zeros((3, 12)), first).shape == (3, 4)
    with pytest.raises(ValueError):
        project_images(np.zeros((3, 11)), first)


# ------------------------------------------------------------ features
def test_feature_counts_match_the_documented_sizes():
    assert [n_polynomial_features(16, d) for d in (1, 2, 3)] == [17, 153, 969]
    for degree in range(4):
        assert len(monomial_exponents(4, degree)) == n_polynomial_features(4, degree)


def test_features_are_prefix_stable_and_finite():
    x = np.random.default_rng(4).normal(size=(7, 4)).astype(np.float32)
    linear, cubic = scaled_polynomial_features(x, 1), scaled_polynomial_features(x, 3)
    assert np.isfinite(cubic).all()
    np.testing.assert_array_equal(cubic[:, : linear.shape[1]], linear)


def test_features_known_values_and_degree_group_scaling():
    x = np.array([[2.0, 3.0]], dtype=np.float32)
    got = scaled_polynomial_features(x, 2)[0]
    # constant stays 1; degree-1 group has 2 monomials, degree-2 group has 3.
    expected = [1, 2 / 2**0.5, 3 / 2**0.5, 4 / 3**0.5, 6 / 3**0.5, 9 / 3**0.5]
    np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_feature_chunking_does_not_change_the_result():
    x = np.random.default_rng(0).normal(size=(10, 3)).astype(np.float32)
    np.testing.assert_array_equal(
        scaled_polynomial_features(x, 3, chunk_size=3), scaled_polynomial_features(x, 3)
    )


def test_zero_initialised_expansion_preserves_logits_exactly():
    rng = np.random.default_rng(5)
    old_features = rng.normal(size=(11, 17)).astype(np.float32)
    old_weights = rng.normal(size=(17, 10)).astype(np.float32)
    expanded = expand_weights(old_weights, 153)
    new_features = rng.normal(size=(11, 153)).astype(np.float32)
    new_features[:, :17] = old_features
    np.testing.assert_array_equal(expanded[:17], old_weights)
    np.testing.assert_array_equal(expanded[17:], 0.0)
    np.testing.assert_allclose(
        new_features @ expanded, old_features @ old_weights, atol=1e-5
    )
    assert expand_weights(old_weights, 17) is old_weights
    with pytest.raises(ValueError, match="cannot shrink"):
        expand_weights(old_weights, 16)


# ------------------------------------------------------------ training
def _fit(weights, x, y, eligible, replay=None, epochs=2):
    rx, ry = (
        replay
        if replay
        else (np.empty((0, x.shape[1]), np.float32), np.empty(0, np.int64))
    )
    return fit_task(
        weights, x, y, rx, ry, epochs=epochs, learning_rate=0.1, batch_size=3, seed=5,
        eligible_classes=np.array(eligible),
    )  # fmt: skip


def test_training_never_updates_future_class_weights():
    x = np.array([[1.0, 0.5], [-0.5, 1.0], [0.2, -1.0]], dtype=np.float32)
    trained = _fit(np.zeros((2, 10), np.float32), x, np.array([0, 1, 0]), [0, 1])
    assert np.any(trained[:, :2] != 0)
    np.testing.assert_array_equal(trained[:, 2:], 0.0)


def test_training_rejects_labels_outside_eligible_classes():
    with pytest.raises(ValueError, match="belong to eligible_classes"):
        _fit(
            np.zeros((2, 4), np.float32),
            np.ones((2, 2), np.float32),
            np.array([0, 2]),
            [0, 1],
        )


def test_training_separates_two_easy_classes_and_uses_replay_rows():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(-2, 0.3, (20, 1)), rng.normal(2, 0.3, (20, 1))])
    x = np.column_stack([np.ones(40), x]).astype(np.float32)
    y = np.repeat([0, 1], 20)
    w = _fit(np.zeros((2, 10), np.float32), x, y, [0, 1], epochs=30)
    assert np.mean((x @ w[:, :2]).argmax(axis=1) == y) == 1.0
    # Replay rows of an earlier class enter the update: the class-0 column differs.
    with_replay = _fit(
        np.zeros((2, 10), np.float32), x[20:], y[20:], [0, 1], replay=(x[:5], y[:5])
    )
    without = _fit(np.zeros((2, 10), np.float32), x[20:], y[20:], [0, 1])
    assert not np.array_equal(with_replay, without)


# ---------------------------------------------------------- task plan
def test_task_plan_is_deterministic_and_respects_classes_and_memory():
    labels = np.repeat(np.arange(10), 6)
    tasks, memory = make_task_plan(labels, seed=12, memory_per_class=2)
    again_tasks, again_memory = make_task_plan(labels, seed=12, memory_per_class=2)
    for task, classes, repeat in zip(tasks, TASK_ORDER, again_tasks):
        assert set(labels[task]) == set(classes)
        np.testing.assert_array_equal(task, repeat)
    for class_id in range(10):
        np.testing.assert_array_equal(memory[class_id], again_memory[class_id])
        assert len(memory[class_id]) == 2 and np.all(
            labels[memory[class_id]] == class_id
        )
    _, capped = make_task_plan(labels, seed=0, memory_per_class=100)
    assert all(len(m) == 6 for m in capped.values())  # never more than are available
    with pytest.raises(ValueError):
        make_task_plan(labels, seed=0, memory_per_class=0)
    with pytest.raises(ValueError, match="missing"):
        make_task_plan(labels[labels < 8], seed=0, memory_per_class=1)


def test_evaluation_only_lets_introduced_classes_compete():
    weights = np.zeros((1, 10), dtype=np.float32)
    weights[0, 9] = 100.0  # class 9 would win everything if it were allowed to compete
    features = np.ones((4, 1), dtype=np.float32)
    labels = np.array([0, 1, 0, 1])
    accuracy = accuracy_by_task(weights, features, labels, n_tasks_seen=1)
    assert accuracy[0] == pytest.approx(0.5)  # ties between 0 and 1 resolve to class 0
    assert np.isnan(accuracy[1:]).all()


# --------------------------------------------------------- experiment
@pytest.fixture(scope="module")
def tiny_data():
    x_train, y_train, x_test, y_test, _ = make_synthetic_2d(
        9, n_train_per_class=6, n_test_per_class=5, separation=3.0, noise=0.2
    )
    return make_feature_data(x_train, y_train, x_test, y_test)


def _run_all(data, seed=2):
    tasks, memory = make_task_plan(data.train_y, seed, memory_per_class=2)
    return {
        c: run_condition(c, data, tasks, memory, seed=seed, epochs=1, learning_rate=0.05,
                         batch_size=8)
        for c in CONDITIONS
    }  # fmt: skip


def test_conditions_differ_only_in_degree_schedule(tiny_data):
    results = _run_all(tiny_data)
    assert results["fixed_linear"]["feature_count_by_task"] == [3] * 5
    assert results["expanding"]["feature_count_by_task"] == [3, 6, 10, 10, 10]
    assert results["fixed_capacity"]["feature_count_by_task"] == [10] * 5


def test_expansion_invariant_is_checked_and_recorded(tiny_data):
    results = _run_all(tiny_data)
    assert len(results["expanding"]["expansion_max_logit_changes"]) == 2
    assert max(results["expanding"]["expansion_max_logit_changes"]) <= 1e-5
    assert results["fixed_linear"]["expansion_max_logit_changes"] == []
    assert results["fixed_capacity"]["expansion_max_logit_changes"] == []


def test_a_broken_expansion_is_rejected(tiny_data, monkeypatch):
    def corrupt(weights, new_count):
        out = np.ones((new_count, weights.shape[1]), dtype=weights.dtype)
        out[: weights.shape[0]] = weights
        return out

    monkeypatch.setattr(experiment_module, "expand_weights", corrupt)
    tasks, memory = make_task_plan(tiny_data.train_y, 0, 2)
    with pytest.raises(RuntimeError, match="changed logits"):
        run_condition("expanding", tiny_data, tasks, memory, seed=0, epochs=1)


def test_accuracy_matrix_is_lower_triangular_and_metrics_follow_from_it(tiny_data):
    for result in _run_all(tiny_data).values():
        matrix = np.array(
            [[np.nan if v is None else v for v in r] for r in result["accuracy_matrix"]]
        )
        assert matrix.shape == (5, 5)
        assert np.isnan(matrix[np.triu_indices(5, k=1)]).all()
        assert np.isfinite(matrix[np.tril_indices(5)]).all()
        # Metrics recomputed independently from the matrix.
        forgetting = [max(0.0, np.max(matrix[t:, t]) - matrix[-1, t]) for t in range(5)]
        np.testing.assert_allclose(result["forgetting_by_task"], forgetting)
        assert result["mean_forgetting"] == pytest.approx(
            np.mean(forgetting[:-1])
        )  # last excluded
        assert result["final_mean_accuracy"] == pytest.approx(matrix[-1].mean())
        assert result["final_current_accuracy"] == pytest.approx(matrix[-1, -1])


def test_unknown_condition_is_rejected(tiny_data):
    tasks, memory = make_task_plan(tiny_data.train_y, 0, 2)
    with pytest.raises(ValueError, match="unknown condition"):
        run_condition("nope", tiny_data, tasks, memory, seed=0)


def test_comparison_is_deterministic_and_summarises_over_seeds(tiny_data):
    kwargs = {
        "epochs": 1,
        "learning_rate": 0.05,
        "batch_size": 8,
        "memory_per_class": 2,
    }
    first = run_comparison(tiny_data, [0, 1], **kwargs)
    second = run_comparison(tiny_data, [0, 1], **kwargs)
    assert first == second
    assert len(first["runs"]) == 2 * len(CONDITIONS)
    assert set(first["summary"]) == set(CONDITIONS)
    values = [
        r["final_mean_accuracy"] for r in first["runs"] if r["condition"] == "expanding"
    ]
    stats = first["summary"]["expanding"]["final_mean_accuracy"]
    assert stats["mean"] == pytest.approx(np.mean(values))
    assert stats["std"] == pytest.approx(np.std(values, ddof=1))
    with pytest.raises(ValueError, match="seed"):
        run_comparison(tiny_data, [], **kwargs)


def test_summary_uses_sample_std_and_zero_for_a_single_seed():
    rows = [
        {"condition": c, "final_mean_accuracy": a, "final_current_accuracy": a, "mean_forgetting": f}
        for c, a, f in (("expanding", 0.7, 0.1), ("expanding", 0.9, 0.3), ("fixed_linear", 0.5, 0.2))
    ]  # fmt: skip
    summary = summarize_runs(rows)
    assert summary["expanding"]["final_mean_accuracy"]["mean"] == pytest.approx(0.8)
    assert summary["expanding"]["final_mean_accuracy"]["std"] == pytest.approx(
        np.std([0.7, 0.9], ddof=1)
    )
    assert summary["fixed_linear"]["mean_forgetting"] == {"mean": 0.2, "std": 0.0}
