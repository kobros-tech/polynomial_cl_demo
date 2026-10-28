# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT
"""Paired-seed ablation: growing polynomial degree versus fixed degree."""

import argparse
import json
from pathlib import Path

import numpy as np

from polynomial_cl.data import make_experiences
from polynomial_cl.training import make_config, run_continual


def summarize(result):
    last = result.records[-1]
    return {
        "final_mean_accuracy": float(last.mean_acc),
        "final_current_accuracy": float(last.acc_current),
        "final_mean_forgetting": float(last.mean_forgetting),
        "final_max_forgetting": float(last.max_forgetting),
        "final_degree": int(last.degree_after),
        "final_trainable_dimensions": int(last.n_trainable),
        "mean_old_prediction_flip_fraction": float(
            np.mean([r.old_flip_fraction for r in result.records[1:]])
        ),
        "mean_max_old_score_change": float(
            np.mean([r.max_old_score_change for r in result.records[1:]])
        ),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--n-experiences", type=int, default=10)
    p.add_argument(
        "--fixed-degree",
        type=int,
        default=None,
        help="fixed-capacity baseline degree (default: final expansion degree, for a matched final capacity)",
    )
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--m-current", type=int, default=100)
    p.add_argument("--m-replay", type=int, default=100)
    p.add_argument("--capacity", type=int, default=100)
    p.add_argument("--output", default="results/synthetic_degree_ablation.json")
    a = p.parse_args()
    if a.n_experiences < 1 or a.epochs < 0:
        p.error("n-experiences must be positive and epochs must be non-negative")
    fixed_degree = a.n_experiences if a.fixed_degree is None else a.fixed_degree
    if fixed_degree < 1:
        p.error("fixed-degree must be positive")
    rows = []
    for seed in a.seeds:
        exps = make_experiences(n_experiences=a.n_experiences, seed=seed)
        shared = {
            "seed": seed,
            "epochs": a.epochs,
            "m_current": a.m_current,
            "m_replay": a.m_replay,
            "capacity": a.capacity,
        }
        conditions = (
            ("expanding", None),
            # Fixed initial degree is the literal no-expansion baseline.
            ("fixed_initial_degree", [1] * a.n_experiences),
            # This control starts with the final capacity, isolating the effect of
            # when features become available rather than final model size.
            ("fixed_capacity", [fixed_degree] * a.n_experiences),
        )
        for name, degrees in conditions:
            cfg = make_config(
                "ordinary", **shared, degrees=degrees, stop_after=a.n_experiences
            )
            result = run_continual(exps, cfg)
            rows.append({"seed": seed, "condition": name, **summarize(result)})
            print(
                f"seed={seed} condition={name} final_mean_accuracy={rows[-1]['final_mean_accuracy']:.4f} final_mean_forgetting={rows[-1]['final_mean_forgetting']:.4f}"
            )
    summary = {}
    for condition in ("expanding", "fixed_initial_degree", "fixed_capacity"):
        subset = [r for r in rows if r["condition"] == condition]
        summary[condition] = {
            k: {
                "mean": float(np.mean([r[k] for r in subset])),
                "std": float(np.std([r[k] for r in subset], ddof=1))
                if len(subset) > 1
                else 0.0,
            }
            for k in (
                "final_mean_accuracy",
                "final_current_accuracy",
                "final_mean_forgetting",
                "final_max_forgetting",
                "final_trainable_dimensions",
                "mean_old_prediction_flip_fraction",
                "mean_max_old_score_change",
            )
        }
    payload = {
        "protocol": {
            "seeds": a.seeds,
            "n_experiences": a.n_experiences,
            "fixed_degree": fixed_degree,
            "comparison": (
                "incremental degree growth vs fixed initial degree (no expansion) "
                "and fixed final-capacity degree (capacity control)"
            ),
            "epochs": a.epochs,
            "m_current": a.m_current,
            "m_replay_per_old_experience": a.m_replay,
            "capacity_per_experience": a.capacity,
        },
        "runs": rows,
        "summary": summary,
    }
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
