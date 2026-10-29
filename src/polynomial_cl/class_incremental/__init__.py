# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Class-incremental comparison: fixed linear vs expanding vs fixed-capacity polynomial heads.

Entry points: ``make_synthetic_2d`` / ``load_projected_mnist`` for data and ``run_comparison``
for the experiment (see ``experiment.py`` for the protocol).
"""

from .data import TASK_ORDER, load_projected_mnist, make_synthetic_2d
from .experiment import (
    CONDITIONS,
    STRATEGIES,
    make_feature_data,
    make_task_plan,
    run_comparison,
    run_condition,
    run_strategy_comparison,
    summarize_runs,
)

__all__ = [
    "CONDITIONS",
    "STRATEGIES",
    "TASK_ORDER",
    "load_projected_mnist",
    "make_feature_data",
    "make_synthetic_2d",
    "make_task_plan",
    "run_comparison",
    "run_condition",
    "run_strategy_comparison",
    "summarize_runs",
]
