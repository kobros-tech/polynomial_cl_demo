# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT
"""Matched class-incremental comparison for synthetic 2D data and Split MNIST."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from polynomial_cl.mnist import load_mnist
from polynomial_cl.multiclass_data import TASK_ORDER, make_synthetic_2d
from polynomial_cl.multiclass_features import (
    MAX_DEGREE,
    N_COMPONENTS,
    make_projection,
    polynomial_features_projected,
    project_images,
)
from polynomial_cl.multiclass_experiment import (
    CONDITIONS,
    evaluate_seen_tasks,
    make_task_plan,
    run_condition,
    summarize_runs,
)

PROTOCOL_VERSION = "class-incremental-v1"


def load_dataset(args):
    """Load a dataset and apply its fixed input representation."""
    if args.dataset == "synthetic_2d":
        xtr, ytr, xte, yte, means = make_synthetic_2d(
            args.data_seed,
            args.synthetic_train_per_class,
            args.synthetic_test_per_class,
            args.class_separation, args.class_noise,
        )
        return xtr, ytr, xte, yte, {
            "data_seed": args.data_seed,
            "synthetic_train_per_class": args.synthetic_train_per_class,
            "synthetic_test_per_class": args.synthetic_test_per_class,
            "class_means": means.tolist(),
            "class_separation": args.class_separation,
            "class_noise_std": args.class_noise,
        }
    xtr, ytr, xte, yte = load_mnist(
        Path(args.data_dir), download=getattr(args, "download_mnist", True)
    )
    projection = make_projection(args.projection_seed)
    return project_images(xtr, projection), ytr, project_images(xte, projection), yte, {
        "projection_seed": args.projection_seed, "projection_dimensions": N_COMPONENTS,
        "projection": "fixed orthonormal random projection; no data fitting",
    }


def run_experiment(args):
    xtr, ytr, xte, yte, adapter_metadata = load_dataset(args)
    # Feature maps are condition/seed invariant; compute once to avoid repeated
    # high-memory polynomial expansion for every run on the image dataset.
    feature_cache = (
        polynomial_features_projected(xtr, MAX_DEGREE),
        polynomial_features_projected(xte, MAX_DEGREE),
    )
    runs = []
    for seed in args.seeds:
        tasks, memory = make_task_plan(ytr, seed, args.memory_per_class)
        for condition in CONDITIONS:
            runs.append(run_condition(
                condition, xtr, ytr, xte, yte, tasks, memory,
                seed=seed, epochs=args.epochs,
                learning_rate=args.lr, batch_size=args.batch_size,
                precomputed_features=feature_cache,
            ))
    return {
        "benchmark": (
            "matched class-incremental polynomial-feature expansion comparison"
        ),
        "protocol_version": PROTOCOL_VERSION, "dataset": args.dataset,
        "protocol": {
            "seeds": args.seeds, "experiences": 5, "classes_per_experience": 2,
            "task_order": [list(pair) for pair in TASK_ORDER],
            "epochs_per_experience": args.epochs, "learning_rate": args.lr,
            "batch_size": args.batch_size, "memory_per_class": args.memory_per_class,
            "replay_policy": (
                "fixed seeded per-class training indices shared across conditions"
            ),
            "training_class_eligibility": (
                "softmax loss and gradients include only classes introduced by the current "
                "experience; future output weights receive no training gradients"
            ),
            "evaluation": (
                "primary matrix masks not-yet-introduced classes; a diagnostic matrix "
                "keeps all ten classes eligible at every experience"
            ),
            "conditions": {
                "fixed_linear": "degree 1 throughout",
                "expanding": (
                    "degree 1, 2, then 3 for experiences 3-5; "
                    "new coefficients zero-initialized"
                ),
                "fixed_capacity": "maximum degree available from experience 1",
            }, **adapter_metadata,
        },
        "runs": runs, "summary": summarize_runs(runs),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("synthetic_2d", "split_mnist"), required=True
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--memory-per-class", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", default="data/mnist")
    parser.add_argument(
        "--no-download", dest="download_mnist", action="store_false",
        help="require all MNIST archives to exist locally; never download them",
    )
    parser.set_defaults(download_mnist=True)
    parser.add_argument("--projection-seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--synthetic-train-per-class", type=int, default=160)
    parser.add_argument("--synthetic-test-per-class", type=int, default=240)
    parser.add_argument("--class-separation", type=float, default=2.5)
    parser.add_argument("--class-noise", type=float, default=0.45)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.memory_per_class < 1:
        parser.error("epochs, batch-size, and memory-per-class must be positive")
    if not args.seeds:
        parser.error("at least one seed is required")
    result = run_experiment(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
