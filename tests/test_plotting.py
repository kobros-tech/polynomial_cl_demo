# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from polynomial_cl import plotting as P
from polynomial_cl.data import make_experiences
from polynomial_cl.model import PolynomialLogisticModel
from polynomial_cl.training import make_config, run_continual


def test_plot_functions_run_and_do_not_change_model():
    exps = make_experiences(n_train=60, n_test=80, seed=0)
    res = run_continual(
        exps,
        make_config(
            "ordinary", epochs=20, m_current=40, m_replay=40, capacity=40, stop_after=3
        ),
    )
    rec = res.records[-1]
    model = rec.model_after
    before = model.w.copy()
    samples = (exps[0].test.x, exps[0].test.y)

    P.plot_ground_truth(samples=samples, res=40)
    P.plot_predictions(model, samples=samples, res=40)
    P.plot_correctness(model, samples=samples, res=40)
    P.plot_prediction_changes(rec.model_before, rec.model_after, res=40)
    P.plot_step(rec, exps, res=40)
    P.plot_coefficients(rec)
    P.plot_coefficients(model=model)
    P.plot_loss_curve(rec.loss_history)
    P.plot_learning_history(res)
    P.plot_accuracy_matrix(res)
    assert "FROZEN" in P.format_coefficient_table(
        rec
    ) or "trainable" in P.format_coefficient_table(rec)
    plt.close("all")
    assert np.array_equal(model.w, before)


def test_correctness_grid_categories_with_known_models():
    zero = PolynomialLogisticModel(1)  # score 0 everywhere -> predicts class 0
    _, _, cat = P.correctness_grid(zero, 50)
    assert set(np.unique(cat)) <= {0, 3}  # only TN or FN possible
    all_one = PolynomialLogisticModel(1, [1.0, 0.0, 0.0])
    _, _, cat = P.correctness_grid(all_one, 50)
    assert set(np.unique(cat)) <= {1, 2}  # only TP or FP possible
