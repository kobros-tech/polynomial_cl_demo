# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Metrics for measuring learning and preservation (all computed identically for every method)."""

from __future__ import annotations

import numpy as np


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def mean_observed_accuracy(acc_matrix, t) -> float:
    """A_bar_t = mean_i A[t, i] over experiences 0..t (0-based t)."""
    return float(np.mean(acc_matrix[t, : t + 1]))


def old_score_change(score_before, score_after) -> float:
    """max_x |f_after(x) - f_before(x)| over a fixed set of old inputs."""
    return float(np.max(np.abs(np.asarray(score_after) - np.asarray(score_before))))


def prediction_flip_fraction(score_before, score_after) -> float:
    """Fraction of inputs whose predicted label changed."""
    return float(
        np.mean((np.asarray(score_before) > 0) != (np.asarray(score_after) > 0))
    )


def forgetting(acc_matrix, t):
    """For experience i <= t: best accuracy achieved so far (rows i..t) minus current accuracy."""
    return np.array(
        [np.max(acc_matrix[i : t + 1, i]) - acc_matrix[t, i] for i in range(t + 1)]
    )


def parameter_changes(w_before, w_after, trainable_mask):
    """(max |change| over frozen coefficients, sum |change| over trainable coefficients)."""
    d = np.abs(np.asarray(w_after) - np.asarray(w_before))
    m = np.asarray(trainable_mask, dtype=bool)
    return (float(d[~m].max()) if (~m).any() else 0.0), float(d[m].sum())


def summary_table(result):
    """One row (dict) per processed experience."""
    rows = []
    for r in result.records:
        rows.append(
            {
                "exp": r.t + 1,
                "degree": r.degree_after,
                "trainable": r.n_trainable,
                "final_loss": r.loss_history[-1],
                "acc_current": r.acc_current,
                "mean_acc": r.mean_acc,
                "mean_forgetting": r.mean_forgetting,
                "max_forgetting": r.max_forgetting,
                "max_old_score_change": r.max_old_score_change,
                "old_flip_frac": r.old_flip_fraction,
                "frozen_max_change": r.frozen_max_change,
                "trainable_L1_update": r.trainable_l1_update,
            }
        )
    return rows


def format_table(rows, floatfmt="{:.3f}"):
    if not rows:
        return ""
    cols = list(rows[0])

    def cell(v):
        return floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)

    widths = [max(len(c), *(len(cell(r[c])) for r in rows)) for c in cols]

    def line(vals):
        return "  ".join(v.rjust(w) for v, w in zip(vals, widths))

    return "\n".join([line(cols)] + [line([cell(r[c]) for c in cols]) for r in rows])
