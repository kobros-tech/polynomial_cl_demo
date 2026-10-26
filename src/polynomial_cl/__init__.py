# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""polynomial_cl: a small NumPy research instrument for polynomial expansion in continual learning."""

from . import metrics, plotting
from .data import (
    Experience,
    Split,
    boundary_curve,
    combine_splits,
    generate_split,
    make_experiences,
    true_label,
    true_score,
)
from .features import (
    PolynomialFeatures,
    feature_exponents,
    feature_names,
    n_features,
    polynomial_features,
)
from .model import (
    PolynomialLogisticModel,
    bce_with_logits,
    check_expansion_invariant,
    sigmoid,
)
from .replay import ReplayBuffer, build_batch
from .training import (
    STRATEGIES,
    ContinualConfig,
    degree_schedule,
    make_config,
    preservation_null_basis,
    run_continual,
    run_seeds,
    train,
)

__version__ = "0.1.0"

__all__ = [
    "STRATEGIES",
    "ContinualConfig",
    "Experience",
    "PolynomialFeatures",
    "PolynomialLogisticModel",
    "ReplayBuffer",
    "Split",
    "bce_with_logits",
    "boundary_curve",
    "build_batch",
    "check_expansion_invariant",
    "combine_splits",
    "degree_schedule",
    "feature_exponents",
    "feature_names",
    "generate_split",
    "make_config",
    "make_experiences",
    "metrics",
    "n_features",
    "plotting",
    "polynomial_features",
    "preservation_null_basis",
    "run_continual",
    "run_seeds",
    "sigmoid",
    "train",
    "true_label",
    "true_score",
]
