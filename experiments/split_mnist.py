# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT
"""Reproducible Split MNIST class-incremental baseline using only NumPy.

Downloads the canonical MNIST IDX files, groups digits into five two-class
experiences (0/1, 2/3, ...), and evaluates a replay-trained linear softmax model.
This is a benchmark baseline, not a claim that 2-D polynomial degree expansion
transfers directly to image inputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.split_mnist_utils import load_mnist


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_task(w, x, y, replay_x, replay_y, epochs, lr, batch_size, seed):
    if len(replay_y):
        x = np.concatenate([x, replay_x])
        y = np.concatenate([y, replay_y])
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for start in range(0, len(y), batch_size):
            ix = order[start : start + batch_size]
            xb, yb = x[ix], y[ix]
            probs = softmax(xb @ w)
            probs[np.arange(len(yb)), yb] -= 1
            w -= lr * (xb.T @ probs / len(yb))
    return w


def run(args):
    xtr, ytr, xte, yte = load_mnist(Path(args.data_dir))
    # Fixed, seeded per-class memory; only training examples are stored.
    rng = np.random.default_rng(args.seed)
    memory_x, memory_y = [], []
    w = np.zeros((784, 10), dtype=np.float32)
    acc = np.full((5, 5), np.nan)
    for task in range(5):
        classes = np.array([2 * task, 2 * task + 1])
        idx = np.flatnonzero(np.isin(ytr, classes))
        idx = idx[rng.permutation(len(idx))]
        # Map original digit IDs to fixed global output IDs; unseen outputs remain available,
        # but are not evaluated as learned until their task arrives.
        x, y = xtr[idx], ytr[idx]
        rx = np.concatenate(memory_x) if memory_x else np.empty((0, 784), np.float32)
        ry = np.concatenate(memory_y) if memory_y else np.empty((0,), np.int64)
        w = fit_task(
            w, x, y, rx, ry, args.epochs, args.lr, args.batch_size, args.seed + task
        )
        for j in range(task + 1):
            test_idx = np.flatnonzero(np.isin(yte, [2 * j, 2 * j + 1]))
            pred = (xte[test_idx] @ w).argmax(axis=1)
            acc[task, j] = np.mean(pred == yte[test_idx])
        for digit in classes:
            candidates = idx[ytr[idx] == digit]
            chosen = rng.choice(
                candidates, min(args.memory_per_class, len(candidates)), replace=False
            )
            memory_x.append(xtr[chosen].copy())
            memory_y.append(ytr[chosen].copy())
        print(
            f"task={task + 1}/5 digits={classes.tolist()} current_acc={acc[task, task]:.4f} mean_seen_acc={np.nanmean(acc[task, : task + 1]):.4f}"
        )
    forgetting = [max(0.0, np.nanmax(acc[:, j]) - acc[-1, j]) for j in range(5)]
    result = {
        "benchmark": "Split MNIST class-incremental linear softmax + replay",
        "seed": args.seed,
        "epochs_per_task": args.epochs,
        "memory_per_class": args.memory_per_class,
        "accuracy_matrix": [
            [float(value) if np.isfinite(value) else None for value in row]
            for row in acc
        ],
        "final_mean_accuracy": float(np.nanmean(acc[-1])),
        "mean_forgetting": float(np.mean(forgetting)),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    import json

    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data/mnist")
    p.add_argument("--output", default="results/split_mnist_seed0.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--memory-per-class", type=int, default=200)
    a = p.parse_args()
    if a.epochs < 1 or a.batch_size < 1 or a.memory_per_class < 1:
        p.error("epochs, batch-size, and memory-per-class must be positive")
    run(a)


if __name__ == "__main__":
    main()
