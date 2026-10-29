# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT
"""Class-incremental comparison (fixed_linear / expanding / fixed_capacity) -> JSON.

Datasets: ``synthetic_2d`` (ten Gaussian classes on a ring) or ``split_mnist`` (MNIST digits
under a fixed 16-dimensional random projection). See polynomial_cl/class_incremental/experiment.py
for the protocol; all logic lives in the package, this script only parses arguments and
writes the result.
"""

import argparse
import json
from pathlib import Path

from polynomial_cl.class_incremental import (
    load_projected_mnist,
    make_feature_data,
    make_synthetic_2d,
    run_comparison,
    run_strategy_comparison,
)
from polynomial_cl.class_incremental.data import N_COMPONENTS, TASK_ORDER
from polynomial_cl.class_incremental.experiment import (
    BATCH_SIZE,
    DEGREES,
    EPOCHS,
    LEARNING_RATE,
    MEMORY_PER_CLASS,
    N_REF_PER_CLASS,
    STRATEGIES,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("synthetic_2d", "split_mnist"), required=True)
    p.add_argument(
        "--experiment",
        choices=("conditions", "strategies"),
        default="conditions",
        help="conditions: fixed_linear/expanding/fixed_capacity; strategies: update rules on 'expanding'",
    )
    p.add_argument(
        "--n-ref-per-class", type=int, default=N_REF_PER_CLASS, help="strategy_b only"
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--memory-per-class", type=int, default=MEMORY_PER_CLASS)
    mnist = p.add_argument_group("split_mnist")
    mnist.add_argument("--data-dir", default="data/mnist")
    mnist.add_argument("--projection-seed", type=int, default=0)
    mnist.add_argument(
        "--no-download",
        dest="download",
        action="store_false",
        help="require the MNIST archives to exist locally; never touch the network",
    )
    synthetic = p.add_argument_group("synthetic_2d")
    synthetic.add_argument("--data-seed", type=int, default=0)
    synthetic.add_argument("--train-per-class", type=int, default=160)
    synthetic.add_argument("--test-per-class", type=int, default=240)
    synthetic.add_argument("--separation", type=float, default=2.5)
    synthetic.add_argument("--noise", type=float, default=0.45)
    args = p.parse_args()
    if not args.seeds:
        p.error("at least one seed is required")
    if min(args.epochs, args.batch_size, args.memory_per_class) < 1:
        p.error("epochs, batch-size and memory-per-class must be positive")

    if args.dataset == "synthetic_2d":
        train_x, train_y, test_x, test_y, means = make_synthetic_2d(
            args.data_seed,
            args.train_per_class,
            args.test_per_class,
            args.separation,
            args.noise,
        )
        data_settings = {
            "data_seed": args.data_seed,
            "train_per_class": args.train_per_class,
            "test_per_class": args.test_per_class,
            "class_separation": args.separation,
            "class_noise_std": args.noise,
            "class_means": means.tolist(),
        }
    else:
        train_x, train_y, test_x, test_y = load_projected_mnist(
            args.data_dir, args.projection_seed, download=args.download
        )
        data_settings = {
            "projection_seed": args.projection_seed,
            "projection_dimensions": N_COMPONENTS,
            "projection": "fixed orthonormal random projection; no data fitting",
        }

    settings = {
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "memory_per_class": args.memory_per_class,
    }
    data = make_feature_data(train_x, train_y, test_x, test_y)
    if args.experiment == "strategies":
        result = run_strategy_comparison(
            data, args.seeds, n_ref_per_class=args.n_ref_per_class, **settings
        )
    else:
        result = run_comparison(data, args.seeds, **settings)
    payload = {
        "dataset": args.dataset,
        "protocol": {
            "seeds": args.seeds,
            "task_order": [list(pair) for pair in TASK_ORDER],
            "epochs_per_experience": args.epochs,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "memory_per_class": args.memory_per_class,
            "experiment": args.experiment,
            "degrees_by_condition": {name: list(d) for name, d in DEGREES.items()},
            "strategies": list(STRATEGIES) if args.experiment == "strategies" else None,
            "n_ref_per_class": args.n_ref_per_class,
            **data_settings,
        },
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
