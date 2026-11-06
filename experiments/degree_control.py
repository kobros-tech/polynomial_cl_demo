# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT
"""Degree-control experiment (Experiments A, B, C) on Split MNIST -> JSON.

Candidates are selected on a validation split with ``--select-seeds``; only the final
configurations are run on the test set with ``--test-seeds`` (disjoint). See
polynomial_cl/continual/degree_control.py for the protocol.
"""

import argparse
import json
from pathlib import Path

from common import add_dataset_arguments, load_dataset

from polynomial_cl.continual import holdout_split, make_feature_data
from polynomial_cl.continual.degree_control import BASELINES, run_degree_control_study
from polynomial_cl.continual.experiment import (
    BATCH_SIZE,
    EPOCHS,
    MEMORY_PER_CLASS,
    N_REF_PER_CLASS,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_dataset_arguments(p)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--strategies", nargs="+", default=list(BASELINES), choices=list(BASELINES)
    )
    p.add_argument("--select-seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--test-seeds", nargs="+", type=int, default=[3, 4, 5, 6, 7, 8, 9])
    p.add_argument("--validation-fraction", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--memory-per-class", type=int, default=MEMORY_PER_CLASS)
    p.add_argument("--n-ref-per-class", type=int, default=N_REF_PER_CLASS)
    args = p.parse_args()
    shared = sorted(set(args.select_seeds) & set(args.test_seeds))
    if shared:
        p.error(f"--select-seeds and --test-seeds overlap on {shared}")
    train_x, train_y, test_x, test_y, data_settings = load_dataset(args)
    fit = holdout_split(train_x, train_y, args.validation_fraction)
    result = run_degree_control_study(
        make_feature_data(*fit),
        make_feature_data(train_x, train_y, test_x, test_y),
        args.select_seeds,
        args.test_seeds,
        strategies=args.strategies,
        epochs=args.epochs,
        batch_size=args.batch_size,
        memory_per_class=args.memory_per_class,
        n_ref_per_class=args.n_ref_per_class,
        log=lambda m: print(m, flush=True),
    )
    payload = {
        "dataset": args.dataset,
        "protocol": {**vars(args), "output": str(args.output), **data_settings},
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=1, allow_nan=False, default=str) + "\n"
    )


if __name__ == "__main__":
    main()
