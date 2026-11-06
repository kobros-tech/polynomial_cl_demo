# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Degree-control experiment: does reducing single degrees' plasticity help?

Protocol (candidate selection on validation data only; the test set is touched once, for the
final configurations):

* ``BASELINES``: the three fixed configurations (rates, strategy, sampling) -- never changed.
* Experiment A (``individual_candidates``): per strategy, the all-one baseline, then one
  degree at a time with every factor of ``FACTORS`` below 1, the others at 1.
* Experiment B (``combination_candidates``): a *limited* set built from A -- the best factor of
  each degree that beat the baseline on validation, all their pairs, all of them together, and the
  plan's example vectors. No grid search.
* Experiment C (``final_comparison``): per strategy, baseline / best individual / best multiple /
  best overall (all chosen on validation by ``choose_candidate``: Pareto front of accuracy and
  forgetting, then max accuracy - forgetting), run on the test data with the test seeds, paired by
  seed. The Pareto front of the validation candidates is reported as well.
"""

from __future__ import annotations

import itertools
import time

import numpy as np

from .degree_rates import N_DEGREE_GROUPS
from .degree_study import choose_candidate, pareto_front, tradeoff_score
from .experiment import (
    BATCH_SIZE,
    EPOCHS,
    MEMORY_PER_CLASS,
    N_REF_PER_CLASS,
    make_task_plan,
    run_condition,
)

FACTORS = (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0)
BASELINES = {
    "ordinary": {"strategy": "ordinary", "rates": [3.0, 3.0, 3.0, 3.0]},
    "frozen_old": {"strategy": "frozen_old", "rates": [2.5, 2.5, 10.0, 10.0]},
    "strategy_b": {"strategy": "strategy_b", "rates": [3.0, 3.0, 0.75, 3.0]},
}
SAMPLING = "class_balanced"
EXAMPLE_VECTORS = (
    (1, 0, 0.05, 1),
    (0, 1, 0.25, 1),
    (1, 0.05, 0.1, 1),
    (0.05, 0, 0.25, 1),
)
METRICS = ("final_mean_accuracy", "mean_forgetting", "final_current_accuracy")
ONES = (1.0,) * N_DEGREE_GROUPS


def name_of(factors):
    return "m[" + ",".join(f"{v:g}" for v in factors) + "]"


def individual_candidates(factors=FACTORS, n_groups=N_DEGREE_GROUPS):
    """``{name: factor vector}``: the all-one baseline and one degree changed at a time."""
    out = {name_of(ONES): ONES}
    for degree in range(n_groups):
        for f in factors:
            if f == 1.0:
                continue
            vec = [1.0] * n_groups
            vec[degree] = float(f)
            out[name_of(vec)] = tuple(vec)
    return out


def combination_candidates(individual_scores, examples=EXAMPLE_VECTORS):
    """Limited multi-degree vectors from Experiment A validation scores.

    ``individual_scores``: name -> tradeoff score for every vector of ``individual_candidates``.
    A degree contributes only if its best factor beats the baseline score.
    """
    base = individual_scores[name_of(ONES)]
    best = {}
    for name, vec in individual_candidates().items():
        changed = [d for d, v in enumerate(vec) if v != 1.0]
        if len(changed) == 1 and individual_scores[name] > base:
            d = changed[0]
            if d not in best or individual_scores[name] > best[d][0]:
                best[d] = (individual_scores[name], vec[d])
    out = {}

    def add(vec):
        vec = tuple(float(v) for v in vec)
        if vec != ONES and sum(v != 1.0 for v in vec) >= 2:
            out[name_of(vec)] = vec

    degrees = sorted(best)
    for size in range(2, len(degrees) + 1):
        for subset in itertools.combinations(degrees, size):
            vec = [1.0] * N_DEGREE_GROUPS
            for d in subset:
                vec[d] = best[d][1]
            add(vec)
    for vec in examples:
        add(vec)
    return out


def run_seeds(data, seeds, strategy, rates, factors, *, settings):
    """One strategy / rate vector / control vector for every seed; records get wall time."""
    runs = []
    for seed in seeds:
        tasks, memory = make_task_plan(data.train_y, seed, settings["memory_per_class"])
        started = time.perf_counter()
        run = run_condition(
            "expanding",
            data,
            tasks,
            memory,
            seed=seed,
            epochs=settings["epochs"],
            learning_rate=rates,
            batch_size=settings["batch_size"],
            strategy=strategy,
            n_ref_per_class=settings["n_ref_per_class"],
            sampling=SAMPLING,
            degree_control_factors=factors,
        )
        run["runtime_seconds"] = time.perf_counter() - started
        runs.append(run)
    return runs


def mean_metrics(runs):
    return {m: float(np.mean([r[m] for r in runs])) for m in METRICS}


def summarize(runs, metrics=(*METRICS, "runtime_seconds")):
    out = {}
    for m in metrics:
        v = np.array([r[m] for r in runs], dtype=float)
        out[m] = {
            "mean": float(v.mean()),
            "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
        }
    return out


def paired_difference(runs, baseline_runs, metric):
    """Per-seed difference ``runs - baseline`` (same seeds): mean, std, wins and n."""
    if [r["seed"] for r in runs] != [r["seed"] for r in baseline_runs]:
        raise ValueError("paired comparison needs identical seeds in identical order")
    d = np.array([a[metric] - b[metric] for a, b in zip(runs, baseline_runs)])
    n = len(d)
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    return {
        "mean_diff": float(d.mean()),
        "std_diff": sd,
        "t": float(d.mean() / (sd / np.sqrt(n))) if sd > 0 else None,
        "n": n,
        "by_seed": d.tolist(),
    }


def select_for_strategy(validation_data, select_seeds, key, *, settings, log=None):
    """Experiments A and B for one strategy on validation data (no test data involved)."""
    base = BASELINES[key]
    strategy, rates = base["strategy"], base["rates"]
    metrics, vectors = {}, {}

    def measure(group, candidates):
        for name, factors in candidates.items():
            if name in metrics:
                continue
            runs = run_seeds(
                validation_data,
                select_seeds,
                strategy,
                rates,
                factors,
                settings=settings,
            )
            metrics[name] = mean_metrics(runs)
            vectors[name] = list(factors)
            if log:
                m = metrics[name]
                log(
                    f"  [{key}/{group}] {name}: acc {m['final_mean_accuracy']:.3f} "
                    f"forg {m['mean_forgetting']:.3f} newest {m['final_current_accuracy']:.3f}"
                )

    individual = individual_candidates()
    measure("A", individual)
    scores = {n: tradeoff_score(metrics[n]) for n in individual}
    multiple = combination_candidates(scores)
    measure("B", multiple)
    return {
        "strategy": strategy,
        "rates": rates,
        "individual": list(individual),
        "multiple": list(multiple),
        "vectors": vectors,
        "validation_metrics": metrics,
        "pareto_all": pareto_front(metrics),
        "pareto_individual": pareto_front({n: metrics[n] for n in individual}),
        "best_individual": choose_candidate({n: metrics[n] for n in individual}),
        "best_multiple": choose_candidate({n: metrics[n] for n in multiple})
        if multiple
        else None,
        "best_overall": choose_candidate(metrics),
    }


def final_comparison(test_data, test_seeds, key, selection, *, settings, log=None):
    """Experiment C for one strategy: the (up to) four named configurations on the test set."""
    base = BASELINES[key]
    names = {
        "baseline": name_of(ONES),
        "best_individual": selection["best_individual"],
        "best_multiple": selection["best_multiple"],
        "best_overall": selection["best_overall"],
    }
    runs_by_vector = {}
    for label, name in names.items():
        if name is None or name in runs_by_vector:
            continue
        runs_by_vector[name] = run_seeds(
            test_data,
            test_seeds,
            base["strategy"],
            base["rates"],
            selection["vectors"][name],
            settings=settings,
        )
        if log:
            s = summarize(runs_by_vector[name])
            log(
                f"  [{key}/test] {label} {name}: acc {s['final_mean_accuracy']['mean']:.4f} "
                f"forg {s['mean_forgetting']['mean']:.4f}"
            )
    baseline_runs = runs_by_vector[names["baseline"]]
    rows = {}
    for label, name in names.items():
        if name is None:
            rows[label] = None
            continue
        runs = runs_by_vector[name]
        rows[label] = {
            "vector_name": name,
            "degree_control_factors": selection["vectors"][name],
            "degree_learning_rates": base["rates"],
            "effective_degree_learning_rates": runs[0][
                "effective_degree_learning_rates"
            ],
            "strategy": base["strategy"],
            "sampling": SAMPLING,
            "summary": summarize(runs),
            "paired_vs_baseline": {
                m: paired_difference(runs, baseline_runs, m)
                for m in (*METRICS, "runtime_seconds")
            },
            "seeds": [r["seed"] for r in runs],
        }
    return rows


def run_degree_control_study(
    validation_data,
    test_data,
    select_seeds,
    test_seeds,
    *,
    strategies=tuple(BASELINES),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    memory_per_class=MEMORY_PER_CLASS,
    n_ref_per_class=N_REF_PER_CLASS,
    log=None,
):
    """Experiments A, B (validation) and C (test); selection and test seeds must be disjoint."""
    if set(select_seeds) & set(test_seeds):
        raise ValueError("selection and test seeds must be disjoint")
    settings = {
        "epochs": epochs,
        "batch_size": batch_size,
        "memory_per_class": memory_per_class,
        "n_ref_per_class": n_ref_per_class,
    }
    cells = []
    for key in strategies:
        if log:
            log(f"=== {key}: selection on validation data, seeds {list(select_seeds)}")
        selection = select_for_strategy(
            validation_data, select_seeds, key, settings=settings, log=log
        )
        if log:
            log(f"=== {key}: final comparison on test data, seeds {list(test_seeds)}")
        final = final_comparison(
            test_data, test_seeds, key, selection, settings=settings, log=log
        )
        cells.append({"key": key, "selection": selection, "final": final})
    return {
        "select_seeds": list(select_seeds),
        "test_seeds": list(test_seeds),
        "settings": settings,
        "sampling": SAMPLING,
        "factors": list(FACTORS),
        "selection_rule": "pareto(accuracy, forgetting), max accuracy - forgetting (validation)",
        "cells": cells,
    }
