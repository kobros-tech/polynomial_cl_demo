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
from .multiclass_data import TASK_ORDER, make_synthetic_2d
from .multiclass_experiment import (
    CONDITIONS,
    evaluate_seen_tasks,
    make_task_plan,
    run_condition,
    summarize_runs,
)
from .multiclass_features import (
    MAX_DEGREE,
    N_COMPONENTS,
    expand_weights,
    make_projection,
    n_polynomial_features,
    polynomial_features_projected,
    project_images,
)
from .mnist import fetch as fetch_mnist, load_mnist, read_idx
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
    "TASK_ORDER",
    "ContinualConfig",
    "CONDITIONS",
    "Experience",
    "MAX_DEGREE",
    "N_COMPONENTS",
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
    "evaluate_seen_tasks",
    "expand_weights",
    "feature_exponents",
    "feature_names",
    "generate_split",
    "make_config",
    "fetch_mnist",
    "load_mnist",
    "make_projection",
    "make_task_plan",
    "make_synthetic_2d",
    "make_experiences",
    "metrics",
    "n_features",
    "n_polynomial_features",
    "plotting",
    "polynomial_features",
    "polynomial_features_projected",
    "project_images",
    "preservation_null_basis",
    "run_condition",
    "run_continual",
    "run_seeds",
    "read_idx",
    "sigmoid",
    "train",
    "true_label",
    "true_score",
    "summarize_runs",
]
