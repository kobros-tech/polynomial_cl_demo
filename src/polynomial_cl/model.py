# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Binary logistic regression on explicit polynomial features (NumPy only).

f_d(x) = w . phi_d(x).  The bias is the coefficient of the constant feature (w[0]); it is
not stored separately, so it can never be counted twice.
"""

from __future__ import annotations

import numpy as np

from .features import PolynomialFeatures


def sigmoid(z):
    """Numerically stable logistic function."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def bce_with_logits(z, y):
    """Mean binary cross-entropy computed from logits without log(sigmoid(.)) underflow:
    max(z,0) - z*y + log(1 + exp(-|z|))."""
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.mean(np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z)))))


class PolynomialLogisticModel:
    def __init__(self, degree: int, w=None):
        self.features = PolynomialFeatures(degree)
        n = self.features.n_features
        if w is None:
            self.w = np.zeros(n)
        else:
            w = np.array(w, dtype=float)
            if w.shape != (n,):
                raise ValueError(f"expected {n} coefficients for degree {degree}")
            self.w = w

    # introspection -------------------------------------------------------
    @property
    def degree(self):
        return self.features.degree

    @property
    def feature_names(self):
        return self.features.names

    @property
    def n_features(self):
        return self.features.n_features

    @property
    def bias(self):
        return self.w[0]

    def coefficients(self):
        return dict(zip(self.feature_names, self.w))

    # inference -----------------------------------------------------------
    def design(self, x):
        return self.features(x)

    def score(self, x):
        return self.design(x) @ self.w

    def proba(self, x):
        return sigmoid(self.score(x))

    def predict(self, x):
        return (self.score(x) > 0).astype(int)

    # structure -----------------------------------------------------------
    def copy(self):
        return PolynomialLogisticModel(self.degree, self.w.copy())

    def expanded(self, new_degree: int):
        """Expansion invariant: keep every existing coefficient exactly, append zeros.
        No training happens here; f_{d+1}(x) == f_d(x) for all x."""
        if new_degree < self.degree:
            raise ValueError("can only expand to an equal or larger degree")
        new = PolynomialLogisticModel(new_degree)
        assert (
            new.feature_names[: self.n_features] == self.feature_names
        )  # prefix property
        new.w[: self.n_features] = self.w
        return new

    def with_added(self, g: PolynomialLogisticModel):
        """Combined model f_old + g (g is padded with zeros if it has lower degree)."""
        d = max(self.degree, g.degree)
        out = self.expanded(d)
        out.w = out.w + g.expanded(d).w
        return out


def check_expansion_invariant(old, new, point_sets):
    """Max |score diff| and max |proba diff| between two models over the given inputs."""
    x = np.concatenate([np.atleast_2d(p) for p in point_sets])
    ds = float(np.max(np.abs(new.score(x) - old.score(x))))
    dp = float(np.max(np.abs(new.proba(x) - old.proba(x))))
    return ds, dp
