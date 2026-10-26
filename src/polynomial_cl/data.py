# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Synthetic 2-D binary data with a known, nonlinear ground-truth rule.

Ground truth:  y = 1  iff  x2 > AMPLITUDE * sin(FREQUENCY * x1),   on [-1, 1]^2.

Experiences: experience k (1-based) of K draws x1 only from the k-th vertical strip
of [-1, 1] (x2 is uniform on [-1, 1]).  Every experience therefore sees a different
part of the boundary, so remembering old strips and fitting a new one are different
things.  Train and test samples of an experience come from independent random streams;
the test split is fixed by (seed, experience) and never used for training or replay.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DOMAIN = (-1.0, 1.0)
AMPLITUDE = 0.5
FREQUENCY = 3.0
_SPLIT_CODE = {"train": 0, "test": 1}


# ---------------------------------------------------------------- ground truth
def boundary_curve(x1):
    """x2-coordinate of the true decision boundary at x1."""
    return AMPLITUDE * np.sin(FREQUENCY * np.asarray(x1, dtype=float))


def true_score(x) -> np.ndarray:
    """Signed margin x2 - boundary(x1); the true class is its sign."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    return x[:, 1] - boundary_curve(x[:, 0])


def true_label(x) -> np.ndarray:
    """Ground-truth prediction function (used for plotting true regions)."""
    return (true_score(x) > 0).astype(int)


# ---------------------------------------------------------------------- data
@dataclass(frozen=True)
class Split:
    """Samples of one experience. `kind` is 'train' or 'test' and is checked by the replay buffer."""

    exp_id: int
    kind: str
    x: np.ndarray
    y: np.ndarray

    def __len__(self):
        return len(self.y)


@dataclass(frozen=True)
class Experience:
    exp_id: int  # 1-based
    window: tuple  # x1 range of this experience
    train: Split
    test: Split


def experience_window(exp_id: int, n_experiences: int = 10):
    lo, hi = DOMAIN
    width = (hi - lo) / n_experiences
    return (lo + (exp_id - 1) * width, lo + exp_id * width)


def generate_split(exp_id, n, seed, kind="train", noise=0.0, n_experiences=10) -> Split:
    """`noise` = probability of flipping each label (labels only; inputs are exact)."""
    rng = np.random.default_rng([seed, exp_id, _SPLIT_CODE[kind]])
    a, b = experience_window(exp_id, n_experiences)
    x = np.column_stack([rng.uniform(a, b, n), rng.uniform(*DOMAIN, n)])
    y = true_label(x)
    if noise > 0:
        y = np.where(rng.random(n) < noise, 1 - y, y)
    x.setflags(write=False)
    y.setflags(write=False)
    return Split(exp_id, kind, x, y)


def make_experiences(
    n_experiences=10, n_train=200, n_test=500, seed=0, noise=0.0, test_noise=0.0
):
    """Build all experiences. The test split depends only on (seed, exp, n_test, test_noise)."""
    return [
        Experience(
            k,
            experience_window(k, n_experiences),
            generate_split(k, n_train, seed, "train", noise, n_experiences),
            generate_split(k, n_test, seed, "test", test_noise, n_experiences),
        )
        for k in range(1, n_experiences + 1)
    ]


def combine_splits(splits):
    """Concatenate splits -> (x, y, source_experience_id)."""
    x = np.concatenate([s.x for s in splits])
    y = np.concatenate([s.y for s in splits])
    src = np.concatenate([np.full(len(s), s.exp_id) for s in splits])
    return x, y, src
