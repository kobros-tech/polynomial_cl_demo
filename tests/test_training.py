# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

from polynomial_cl.data import make_experiences
from polynomial_cl.model import PolynomialLogisticModel, bce_with_logits, sigmoid
from polynomial_cl.training import (
    make_config,
    preservation_null_basis,
    run_continual,
    train,
)


def _linear_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (n, 2))
    return x, (x[:, 0] + 0.5 * x[:, 1] > 0).astype(float)


def _circle_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (n, 2))
    return x, ((x**2).sum(1) < 0.5).astype(float)


# ---------------------------------------------------------------- numerics
def test_stable_bce_and_sigmoid_extremes():
    z = np.array([-800.0, 800.0, 0.0])
    assert np.isfinite(bce_with_logits(z, np.array([1.0, 0.0, 1.0])))
    assert np.all(np.isfinite(sigmoid(z))) and sigmoid(np.array([0.0]))[0] == 0.5


def test_gradient_matches_finite_differences():
    x, y = _circle_data(50)
    m = PolynomialLogisticModel(2, np.random.default_rng(0).normal(size=6) * 0.3)
    phi = m.design(x)
    g = phi.T @ (sigmoid(phi @ m.w) - y) / len(y)
    for k in range(6):
        e = np.zeros(6)
        e[k] = 1e-6
        num = (
            bce_with_logits(phi @ (m.w + e), y) - bce_with_logits(phi @ (m.w - e), y)
        ) / 2e-6
        assert abs(num - g[k]) < 1e-6


# ---------------------------------------------------------------- training
def test_loss_decreases_on_learnable_data():
    x, y = _linear_data()
    r = train(PolynomialLogisticModel(1), x, y, lr=0.5, epochs=200)
    assert r.loss_history[-1] < 0.5 * r.loss_history[0]
    assert np.all(
        np.diff(r.loss_history) <= 1e-12
    )  # full-batch GD, small step: monotone


def test_training_is_reproducible_and_seed_matters():
    x, y = _circle_data()
    kw = {"lr": 0.3, "epochs": 15, "batch_size": 32}
    a = train(PolynomialLogisticModel(2), x, y, seed=5, **kw).w_after
    b = train(PolynomialLogisticModel(2), x, y, seed=5, **kw).w_after
    c = train(PolynomialLogisticModel(2), x, y, seed=6, **kw).w_after
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_frozen_coefficients_do_not_change():
    x, y = _circle_data()
    m = PolynomialLogisticModel(2, np.array([0.1, 0.2, -0.3, 0.4, 0.5, -0.6]))
    mask = np.array([False, True, False, True, True, False])
    r = train(m, x, y, lr=0.5, epochs=50, trainable=mask)
    assert np.array_equal(r.w_after[~mask], m.w[~mask])
    assert np.any(r.w_after[mask] != m.w[mask])
    assert np.array_equal(
        m.w, [0.1, 0.2, -0.3, 0.4, 0.5, -0.6]
    )  # input model untouched


def test_zero_initialised_coefficient_can_learn():
    x, y = _circle_data()
    r = train(
        PolynomialLogisticModel(2), x, y, lr=0.5, epochs=100
    )  # starts at all zeros
    names = r.model.feature_names
    assert (
        r.w_after[names.index("x1^2")] != 0.0 and r.w_after[names.index("x2^2")] != 0.0
    )


def test_per_coefficient_learning_rates():
    x, y = _circle_data()
    lr = np.array([0.5, 0.0, 0.5, 0.05, 0.5, 0.5])
    r = train(PolynomialLogisticModel(2), x, y, lr=lr, epochs=50)
    assert r.w_after[1] == 0.0  # lr = 0 behaves as frozen
    assert np.array_equal(r.lr_vector, lr)


# ---------------------------------------------------------------- preservation
@pytest.fixture(scope="module")
def setup():
    exps = make_experiences(n_train=100, n_test=200, seed=2)
    x1, y1 = exps[0].train.x, exps[0].train.y
    old = train(
        PolynomialLogisticModel(2), x1, y1, lr=0.5, epochs=300
    ).model  # f_old, learned on exp 1
    exp5 = exps[4].train
    return exps, old, exp5


def test_null_basis_properties():
    ref = np.random.default_rng(0).uniform(-1, 1, (4, 2))
    from polynomial_cl.features import polynomial_features

    N = preservation_null_basis(3, ref)
    assert N.shape == (10, 6)
    np.testing.assert_allclose(polynomial_features(ref, 3) @ N, 0, atol=1e-12)
    np.testing.assert_allclose(N.T @ N, np.eye(6), atol=1e-12)


def test_frozen_old_component_unchanged_but_combined_predictions_change(setup):
    """A: f_old is untouched, yet unconstrained g changes the combined classifier."""
    exps, old, exp5 = setup
    f_old = old.expanded(3)
    w_snapshot = f_old.w.copy()
    res = train(
        PolynomialLogisticModel(3),
        exp5.x,
        exp5.y,
        lr=0.5,
        epochs=200,
        offset=f_old.score(exp5.x),
    )
    combined = f_old.with_added(res.model)
    assert np.array_equal(f_old.w, w_snapshot)  # component preserved
    xo = exps[0].test.x
    assert (
        np.max(np.abs(combined.score(xo) - f_old.score(xo))) > 1e-3
    )  # combined function moved
    assert (
        np.mean(combined.predict(xo) != f_old.predict(xo)) > 0
    )  # ... and some predictions flipped


def test_preservation_constraint_holds_on_reference_inputs_only(setup):
    """B: g(x_ref)=0 to rounding, but nothing is guaranteed away from x_ref."""
    exps, old, exp5 = setup
    f_old = old.expanded(3)
    ref = exps[0].train.x[:3]
    N = preservation_null_basis(3, ref)
    assert N.shape[1] == 7
    res = train(
        PolynomialLogisticModel(3),
        exp5.x,
        exp5.y,
        lr=0.5,
        epochs=200,
        offset=f_old.score(exp5.x),
        null_basis=N,
    )
    combined = f_old.with_added(res.model)
    assert np.max(np.abs(combined.score(ref) - f_old.score(ref))) < 1e-9
    xo = exps[0].test.x
    assert (
        np.max(np.abs(combined.score(xo) - f_old.score(xo))) > 1e-3
    )  # still moves elsewhere
    assert np.max(np.abs(res.model.score(ref))) < 1e-9


def test_overconstrained_component_cannot_learn(setup):
    _, _, exp5 = setup
    ref = np.random.default_rng(0).uniform(
        -1, 1, (20, 2)
    )  # 20 generic points > 10 features
    N = preservation_null_basis(3, ref)
    assert N.shape[1] == 0
    res = train(
        PolynomialLogisticModel(3), exp5.x, exp5.y, lr=0.5, epochs=20, null_basis=N
    )
    assert np.all(res.w_after == 0.0)  # reported as a failure to learn, not "fixed"


def test_strategy_b_in_the_loop_respects_constraint_and_frozen_mask():
    exps = make_experiences(n_train=60, n_test=80, seed=1)
    for name in ("strategy_b_all", "strategy_b_new"):
        res = run_continual(
            exps,
            make_config(
                name, epochs=30, m_current=40, m_replay=40, capacity=40, stop_after=6
            ),
        )
        for r in res.records[1:]:
            assert r.ref_score_change < 1e-9
            assert r.frozen_max_change == 0.0
            assert r.null_dim is not None
    res_new = run_continual(
        exps,
        make_config(
            "frozen_old",
            epochs=30,
            m_current=40,
            m_replay=40,
            capacity=40,
            stop_after=4,
        ),
    )
    assert all(r.frozen_max_change == 0.0 for r in res_new.records)


def test_strategy_a_all_equals_ordinary_training():
    """Documented fact: an unconstrained g over all coefficients is ordinary training."""
    exps = make_experiences(n_train=60, n_test=80, seed=1)
    kw = {
        "epochs": 40,
        "m_current": 40,
        "m_replay": 40,
        "capacity": 40,
        "stop_after": 4,
    }
    a = run_continual(exps, make_config("ordinary", **kw)).records[-1].w_after
    b = run_continual(exps, make_config("strategy_a_all", **kw)).records[-1].w_after
    np.testing.assert_allclose(a, b, atol=1e-10)


def test_preset_cannot_be_silently_overridden():
    with pytest.raises(ValueError):
        make_config("ordinary_no_replay", m_replay=100)
    assert make_config("ordinary_no_replay").m_replay == 0


def test_no_replay_strategy_really_has_no_replay():
    exps = make_experiences(n_train=60, n_test=80, seed=1)
    res = run_continual(
        exps,
        make_config(
            "ordinary_no_replay", epochs=10, m_current=40, capacity=40, stop_after=4
        ),
    )
    assert all(list(r.batch_counts) == [r.t + 1] for r in res.records) or all(
        sum(v for k, v in r.batch_counts.items() if k != r.t + 1) == 0
        for r in res.records
    )
