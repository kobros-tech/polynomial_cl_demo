# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Degree-control matrix M: effective step ``H M`` for every strategy."""

import numpy as np
import pytest

from polynomial_cl.continual import make_feature_data, make_task_plan, run_condition
from polynomial_cl.continual.degree_rates import (
    apply_degree_control,
    as_degree_control,
    degree_of_feature,
    step_scale_for,
)
from polynomial_cl.continual.experiment import MAX_DEGREE
from polynomial_cl.continual.strategies import STRATEGIES
from polynomial_cl.training.multiclass import fit_task, null_space_basis

RATES = [3.0, 3.0, 0.75, 3.0]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(10, 16)) * 1.5
    y = np.repeat(np.arange(10), 50)
    x = (centres[y] + rng.normal(size=(500, 16))).astype(np.float32)
    ty = np.repeat(np.arange(10), 20)
    tx = (centres[ty] + rng.normal(size=(200, 16))).astype(np.float32)
    return make_feature_data(x, y, tx, ty)


def run(data, strategy="ordinary", factors=None, rates=RATES, seed=3, **kw):
    tasks, memory = make_task_plan(data.train_y, seed, 20)
    return run_condition(
        "expanding",
        data,
        tasks,
        memory,
        seed=seed,
        epochs=2,
        batch_size=32,
        strategy=strategy,
        learning_rate=rates,
        sampling="class_balanced",
        degree_control_factors=factors,
        **kw,
    )


def same(a, b):
    keys = (
        "accuracy_matrix",
        "forgetting_by_task",
        "train_loss_by_task",
        "l1_update_by_task",
    )
    return all(a[k] == b[k] for k in keys)


# ------------------------------------------------------------------ validation
@pytest.mark.parametrize(
    "bad",
    [
        [1, 1, 1],
        [1] * 5,
        1.0,
        [[1, 1, 1, 1]],
        [1, 1, 1, -0.1],
        [1, 1, 1, 1.01],
        [1, 1, 1, np.nan],
        [1, 1, 1, np.inf],
        [True, 1, 1, 1],
        ["a", "b", "c", "d"],
    ],
)
def test_invalid_factors_are_rejected(bad):
    with pytest.raises((ValueError, TypeError)):
        as_degree_control(bad)


def test_run_condition_rejects_invalid_factors(data):
    with pytest.raises(ValueError):
        run(data, factors=[1, 1, 1])
    with pytest.raises(ValueError):
        run(data, factors=[1, 1, 2, 1])


def test_boundary_values_are_valid():
    assert as_degree_control([0, 1, 0.05, 0.5]).tolist() == [0, 1, 0.05, 0.5]
    assert as_degree_control([0, 0, 0, 0]).tolist() == [0, 0, 0, 0]


def test_effective_rate_is_the_product():
    out = apply_degree_control(np.array(RATES), [1, 0, 0.05, 0.5])
    assert out.tolist() == [3.0, 0.0, 0.75 * 0.05, 1.5]


# ------------------------------------------------------------------ equivalence
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_all_ones_reproduces_the_original_run_exactly(data, strategy):
    assert same(run(data, strategy), run(data, strategy, [1.0] * 4))
    assert same(run(data, strategy), run(data, strategy, [1, 1, 1, 1]))


@pytest.mark.parametrize("strategy", ["ordinary", "frozen_old", "strategy_b"])
@pytest.mark.parametrize("factors", [[1, 0, 1, 1], [1, 1, 0.05, 1], [0.25, 1, 1, 0]])
def test_control_equals_the_product_as_learning_rate(data, strategy, factors):
    """Zero (or 0.05, ...) control == the same effective learning rate, bit for bit."""
    effective = list(np.array(RATES) * np.array(factors))
    assert same(
        run(data, strategy, factors), run(data, strategy, None, rates=effective)
    )


def test_zero_factor_equals_zero_learning_rate_for_ordinary_sgd(data):
    a = run(data, "ordinary", [1, 1, 0, 1])
    b = run(data, "ordinary", None, rates=[3.0, 3.0, 0.0, 3.0])
    assert same(a, b)


# ------------------------------------------------------------------ freezing / scaling
@pytest.mark.parametrize("strategy", ["ordinary", "frozen_old", "strategy_b"])
def test_zero_factor_freezes_that_group_and_others_still_move(data, strategy):
    r = run(data, strategy, [1, 0, 1, 1])
    change = r["degree_diagnostics"]["coef_change_l1"]
    for task_row in change:
        assert task_row[1] == 0.0
    assert any(row[0] > 0 for row in change)
    assert any(row[2] is not None and row[2] > 0 for row in change)


def test_all_zero_factors_train_nothing(data):
    r = run(data, "ordinary", [0, 0, 0, 0])
    assert all(v == 0.0 for v in r["l1_update_by_task"])
    assert all(v == 0 for v in r["n_trainable_by_task"])
    assert r["train_loss_by_task"][0][0] == r["train_loss_by_task"][0][1]
    assert r["effective_degree_learning_rates"] == [0.0] * 4


def test_all_zero_factors_also_work_for_strategy_b_and_frozen(data):
    for strategy in ("strategy_b", "frozen_old"):
        r = run(data, strategy, [0, 0, 0, 0])
        assert all(v == 0.0 for v in r["l1_update_by_task"])


def test_factor_005_scales_one_update_by_5_percent_and_leaves_others_alone():
    rng = np.random.default_rng(1)
    n_features = 969  # degree 3 on 16 components
    group = degree_of_feature(16, MAX_DEGREE)
    x = rng.normal(size=(64, n_features)).astype(np.float32) * 0.1
    y = rng.integers(0, 4, 64)
    empty = (np.empty((0, n_features), np.float32), np.empty(0, np.int64))
    base = np.array(RATES)

    def step(factors):
        w = np.zeros((n_features, 10), np.float32)
        lr, scale = step_scale_for(apply_degree_control(base, factors), group)
        fit_task(
            w,
            x,
            y,
            *empty,
            epochs=1,
            learning_rate=lr,
            batch_size=64,
            seed=0,
            eligible_classes=np.arange(4),
            group_scale=scale,
            group_index=group,
        )
        return w

    ref, ctrl = step([1, 1, 1, 1]), step([1, 1, 0.05, 1])
    g2 = group == 2
    assert np.allclose(ctrl[g2], 0.05 * ref[g2], rtol=1e-5, atol=1e-9)
    assert np.abs(ref[g2]).max() > 0
    assert np.array_equal(ctrl[~g2], ref[~g2])  # uncontrolled groups bit-identical


def test_frozen_old_keeps_its_freezing_and_adds_the_control(data):
    a = run(data, "frozen_old", [1, 1, 1, 1])
    b = run(data, "frozen_old", [1, 0.25, 1, 1])
    assert a["n_trainable_by_task"] == b["n_trainable_by_task"]
    assert a["accuracy_matrix"] != b["accuracy_matrix"]
    da, db = a["degree_diagnostics"], b["degree_diagnostics"]
    assert (
        da["n_trainable"] == db["n_trainable"]
    )  # the control never changes what is frozen
    for task in range(1, 5):  # old rows stay frozen: fewer trainable than coefficients
        assert da["n_trainable"][task][0] < da["n_coefficients"][task][0]
    assert all(row[1] == 3.0 * 0.25 for row in db["lr"])  # effective rate of degree 1
    assert all(row[1] == 3.0 for row in da["lr"])


# ------------------------------------------------------------------ Strategy B
@pytest.mark.parametrize(
    "factors", [[1, 1, 1, 1], [1, 0.05, 1, 1], [0, 1, 0.25, 1], [1, 0, 0.05, 1]]
)
def test_strategy_b_preserves_reference_logits_under_control(data, factors):
    r = run(data, "strategy_b", factors)
    changes = [v for v in r["ref_logit_change_by_task"] if v is not None]
    assert changes and max(changes) < 1e-3


def test_strategy_b_projection_uses_the_controlled_step_scale():
    """sqrt(D) N N^T sqrt(D) g leaves the reference design unchanged for D = H M."""
    rng = np.random.default_rng(2)
    group = degree_of_feature(16, MAX_DEGREE)[:153]  # degree 2
    design = rng.normal(size=(6, 153))
    d = apply_degree_control(np.array(RATES), [1, 0.05, 0.25, 1])[group]
    root = np.sqrt(d)
    basis = null_space_basis(design * root).astype(np.float64)
    g = rng.normal(size=(153, 4))
    step = root[:, None] * (basis @ (basis.T @ (root[:, None] * g)))
    assert np.abs(design @ step).max() < 1e-4
    # a mask applied AFTER the projection of the uncontrolled step breaks preservation
    plain = null_space_basis(design).astype(np.float64)
    masked = np.sqrt(np.array(RATES)[group] * 0 + 1)[:, None] * (plain @ (plain.T @ g))
    masked[group == 1] *= 0.0
    assert np.abs(design @ masked).max() > 1e-3


# ------------------------------------------------------------------ study module
def test_individual_candidates_change_one_degree_at_a_time():
    from polynomial_cl.continual.degree_control import FACTORS, individual_candidates

    cands = individual_candidates()
    assert len(cands) == 1 + 4 * (len(FACTORS) - 1)
    for vec in cands.values():
        assert sum(v != 1.0 for v in vec) <= 1


def test_combinations_are_limited_and_need_two_changed_degrees():
    from polynomial_cl.continual.degree_control import (
        combination_candidates,
        individual_candidates,
        name_of,
    )

    scores = dict.fromkeys(individual_candidates(), 0.0)
    scores["m[0,1,1,1]"] = 0.1
    scores["m[1,1,0.25,1]"] = 0.2
    multi = combination_candidates(scores)
    assert "m[0,1,0.25,1]" in multi
    assert all(sum(v != 1.0 for v in vec) >= 2 for vec in multi.values())
    assert name_of((1, 1, 1, 1)) not in multi
    assert len(multi) < 12


def test_study_requires_disjoint_seeds(data):
    from polynomial_cl.continual.degree_control import run_degree_control_study

    with pytest.raises(ValueError):
        run_degree_control_study(data, data, [0, 1], [1, 2])
