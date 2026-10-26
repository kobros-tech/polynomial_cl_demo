# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Gradient-descent training, expansion-aware continual learning loop, and Strategy B.

Everything here is plain NumPy. Key ideas
-----------------------------------------
* `train` supports a global or per-coefficient learning rate, a boolean trainable mask
  (frozen coefficients are never touched; this is asserted), shuffled mini-batches with an
  explicit seed, an optional fixed `offset` score (used to represent f = f_old + g), and an
  optional `null_basis` that restricts every update to a subspace.
* Strategy B represents f_t = f_old + g_t.  f_old is the (expanded) previous model and is
  never modified.  g_t starts at zero.  Two variants are distinguished explicitly:
    A. frozen old component  : g is unconstrained (the old *function* is untouched but the
                               combined predictions may change anywhere).
    B. preserved predictions : g(x_i) = 0 is enforced exactly on reference inputs x_i by
                               restricting g's coefficients to the null space of the
                               reference design matrix.  This preserves f_old at those
                               finitely many inputs only; nothing is guaranteed elsewhere.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from .data import DOMAIN, Experience
from .features import n_features, polynomial_features
from .metrics import (
    accuracy,
    forgetting,
    old_score_change,
    parameter_changes,
    prediction_flip_fraction,
)
from .model import (
    PolynomialLogisticModel,
    bce_with_logits,
    check_expansion_invariant,
    sigmoid,
)
from .replay import ReplayBuffer, build_batch


# ============================================================ basic training
@dataclass
class TrainResult:
    model: PolynomialLogisticModel
    loss_history: np.ndarray  # loss before training, then after every epoch
    w_before: np.ndarray
    w_after: np.ndarray
    lr_vector: np.ndarray  # effective per-coefficient learning rate (0 if frozen)
    trainable: np.ndarray  # boolean mask


def _as_vector(lr, n):
    values = np.asarray(lr, dtype=float)
    if values.ndim == 0:
        values = np.full(n, float(values))
    elif values.shape != (n,):
        raise ValueError(f"lr must be a scalar or have shape ({n},), got {values.shape}")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("learning rates must be finite and non-negative")
    return values.copy()


def _positive_int(value, name, *, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")


def train(
    model,
    x,
    y,
    lr=0.5,
    epochs=100,
    batch_size=None,
    seed=0,
    trainable=None,
    offset=None,
    null_basis=None,
) -> TrainResult:
    """Gradient descent on mean BCE. The input model is not modified; a trained copy is returned.

    lr         scalar or per-coefficient vector (eta_i). Frozen coefficients get eta_i = 0.
    batch_size None = full batch; otherwise shuffled mini-batches (reshuffled every epoch,
               RNG = default_rng(seed)).
    offset     fixed per-sample score added to the model score (f = offset + w.phi(x)).
    null_basis (n_coef, k) matrix; each update step is projected onto its column space.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or len(x) == 0:
        raise ValueError("x must be a non-empty 2D array")
    if y.shape != (len(x),) or not np.all(np.isfinite(y)):
        raise ValueError("y must be a finite 1D array with one label per row of x")
    if not np.all(np.isin(y, [0.0, 1.0])):
        raise ValueError("y must contain only binary labels 0 and 1")
    epochs = _positive_int(epochs, "epochs", allow_zero=True)
    n_coef = model.n_features
    mask = (
        np.ones(n_coef, bool)
        if trainable is None
        else np.asarray(trainable, bool).copy()
    )
    if mask.shape != (n_coef,):
        raise ValueError(f"trainable must have shape ({n_coef},)")
    raw_lr = _as_vector(lr, n_coef)
    if null_basis is not None:
        null_basis = np.asarray(null_basis, dtype=float)
        if null_basis.ndim != 2 or null_basis.shape[0] != n_coef:
            raise ValueError(f"null_basis must have shape ({n_coef}, k)")
        if not np.all(np.isfinite(null_basis)):
            raise ValueError("null_basis must contain only finite values")
        if not np.allclose(null_basis.T @ null_basis, np.eye(null_basis.shape[1]), atol=1e-8):
            raise ValueError("columns of null_basis must be orthonormal")
        active = mask & np.any(np.abs(null_basis) > 1e-12, axis=1)
        active_rates = raw_lr[active]
        if active_rates.size and not np.allclose(active_rates, active_rates[0]):
            raise ValueError(
                "null_basis constraints require one shared learning rate across "
                "all trainable coefficients represented in the null space"
            )
    lr_vec = raw_lr * mask
    off = np.zeros(len(y)) if offset is None else np.asarray(offset, dtype=float)
    if off.shape != (len(y),) or not np.all(np.isfinite(off)):
        raise ValueError("offset must be a finite 1D array with one value per sample")
    phi = model.design(x)
    w0 = model.w.copy()
    w = w0.copy()
    rng = np.random.default_rng(seed)
    n = len(y)
    bs = n if batch_size is None else _positive_int(batch_size, "batch_size")

    def full_loss():
        return bce_with_logits(off + phi @ w, y)

    hist = [full_loss()]
    for _ in range(int(epochs)):
        order = rng.permutation(n) if bs < n else np.arange(n)
        for s in range(0, n, bs):
            idx = order[s : s + bs]
            err = sigmoid(off[idx] + phi[idx] @ w) - y[idx]
            grad = phi[idx].T @ err / len(idx)
            step = lr_vec * grad
            if null_basis is not None:
                step = null_basis @ (null_basis.T @ step)
            w -= step
        hist.append(full_loss())
    assert np.array_equal(w[~mask], w0[~mask]), "a frozen coefficient changed"
    return TrainResult(
        PolynomialLogisticModel(model.degree, w), np.array(hist), w0, w, lr_vec, mask
    )


# ================================================== preservation constraint (B)
def preservation_null_basis(degree, x_ref, trainable=None, rtol=1e-10):
    """Orthonormal basis N (n_coef, k) of {v : Phi(x_ref) v = 0, v_i = 0 for frozen i}.

    Any g with coefficients v = N z satisfies g(x_ref) = 0 exactly (to rounding). If the
    constraints use up all degrees of freedom, k = 0 and g cannot learn anything - that is
    a result to report, not something to repair.
    """
    n = n_features(degree)
    mask = np.ones(n, bool) if trainable is None else np.asarray(trainable, bool)
    cols = np.flatnonzero(mask)
    x_ref = np.atleast_2d(x_ref)
    if len(x_ref) == 0 or len(cols) == 0:
        basis_t = np.eye(len(cols))
    else:
        a = polynomial_features(x_ref, degree)[:, cols]
        _, s, vt = np.linalg.svd(a, full_matrices=True)
        rank = int(np.sum(s > rtol * max(s[0], 1.0))) if len(s) else 0
        basis_t = vt[rank:].T
    basis = np.zeros((n, basis_t.shape[1]))
    basis[cols, :] = basis_t
    return basis


# ============================================================ continual loop
@dataclass
class ContinualConfig:
    degrees: Sequence[int] | None = None  # per experience; default 1,2,...,K
    lr: float = 0.5
    lr_old_factor: float = 1.0  # old coefficients use lr * factor
    freeze_old: bool = False  # old coefficients get mask False
    epochs: int = 300
    batch_size: int | None = None
    capacity: int = 100  # replay buffer capacity per experience
    m_current: int = 100  # samples from the current experience per update
    m_replay: int = 100  # samples from EACH old experience (0 = no replay)
    replace: bool = False  # sampling with replacement?
    composed: bool = False  # Strategy B representation f = f_old + g
    preserve: bool = False  # enforce g(x_ref) = 0 (needs composed=True)
    scope: str = "all"  # composed: g over "all" coefficients or only "new"
    n_ref_per_exp: int = 3  # reference inputs per old experience
    seed: int = 0
    stop_after: int | None = None  # process only the first `stop_after` experiences
    name: str = "ordinary"


def degree_schedule(n, start=1, step=1, max_degree=None):
    d = [start + step * i for i in range(n)]
    return [min(v, max_degree) for v in d] if max_degree else d


STRATEGIES = {
    "ordinary_no_replay": {"m_replay": 0},
    "ordinary": {},
    "frozen_old": {"freeze_old": True},
    "small_old_lr": {"lr_old_factor": 0.1},
    "strategy_a_all": {"composed": True, "scope": "all"},
    "strategy_a_new": {"composed": True, "scope": "new"},
    "strategy_b_all": {"composed": True, "preserve": True, "scope": "all"},
    "strategy_b_new": {"composed": True, "preserve": True, "scope": "new"},
}


def make_config(strategy="ordinary", **overrides) -> ContinualConfig:
    """Preset + shared hyper-parameters. A preset's defining fields (e.g. m_replay=0 for
    'ordinary_no_replay') cannot be overridden, because doing so would silently turn one
    strategy into another."""
    clash = set(STRATEGIES[strategy]) & set(overrides)
    if clash:
        raise ValueError(
            f"strategy '{strategy}' defines {sorted(clash)}; do not override them"
        )
    return ContinualConfig(name=strategy, **{**STRATEGIES[strategy], **overrides})


@dataclass
class StepRecord:
    t: int  # 0-based index of the experience
    degree_before: int
    degree_after: int
    feature_names: list
    model_before: PolynomialLogisticModel  # after previous experience (pre-expansion)
    model_expanded: (
        PolynomialLogisticModel  # immediately after expansion, before training
    )
    model_after: PolynomialLogisticModel
    g_w: np.ndarray | None  # trainable component g (composed mode)
    w_expanded: np.ndarray
    w_after: np.ndarray
    lr_vector: np.ndarray
    trainable: np.ndarray
    n_trainable: int
    null_dim: int | None  # degrees of freedom left by the constraint
    loss_history: np.ndarray
    batch_counts: dict
    invariant_score_diff: float
    invariant_proba_diff: float
    acc_row: np.ndarray  # accuracy on experiences 0..t
    acc_current: float
    mean_acc: float
    forgetting: np.ndarray
    mean_forgetting: float
    max_forgetting: float
    max_old_score_change: float
    old_flip_fraction: float
    ref_score_change: float | None  # max |f_after - f_before| on reference inputs
    ref_flip_fraction: float | None
    frozen_max_change: float | None  # None when the trainable mask has no frozen coefficients
    preserved_component_max_change: float | None  # 0 by construction for composed strategies
    combined_max_coefficient_change: float
    trainable_l1_update: float


@dataclass
class ContinualResult:
    config: ContinualConfig
    experiences: list
    records: list
    acc_matrix: np.ndarray  # [t, i], NaN where i > t


def _probes(seen, seed):
    rng = np.random.default_rng([seed, 12345])
    inside = rng.uniform(*DOMAIN, (500, 2))
    outside = rng.uniform(-3, 3, (500, 2))
    return [e.train.x for e in seen] + [e.test.x for e in seen] + [inside, outside]


def run_continual(
    experiences: Sequence[Experience], cfg: ContinualConfig
) -> ContinualResult:
    """Process experiences one at a time. Only the current train split, the replay buffer
    and the current parameters are used for learning; test splits are used for evaluation
    only, and future experiences are never touched."""
    if not experiences:
        raise ValueError("experiences must not be empty")
    if cfg.scope not in {"all", "new"}:
        raise ValueError("scope must be 'all' or 'new'")
    if not np.isfinite(cfg.lr) or cfg.lr < 0:
        raise ValueError("lr must be finite and non-negative")
    if not np.isfinite(cfg.lr_old_factor) or cfg.lr_old_factor < 0:
        raise ValueError("lr_old_factor must be finite and non-negative")
    _positive_int(cfg.epochs, "epochs", allow_zero=True)
    _positive_int(cfg.capacity, "capacity")
    _positive_int(cfg.m_current, "m_current")
    _positive_int(cfg.m_replay, "m_replay", allow_zero=True)
    _positive_int(cfg.n_ref_per_exp, "n_ref_per_exp", allow_zero=True)
    if cfg.batch_size is not None:
        _positive_int(cfg.batch_size, "batch_size")
    if cfg.stop_after is not None:
        _positive_int(cfg.stop_after, "stop_after")
    K = len(experiences) if cfg.stop_after is None else min(cfg.stop_after, len(experiences))
    degrees = list(cfg.degrees) if cfg.degrees is not None else degree_schedule(len(experiences))
    if len(degrees) < K:
        raise ValueError(f"degrees must contain at least {K} entries for the processed experiences")
    if any(isinstance(d, bool) or not isinstance(d, (int, np.integer)) or d < 1 for d in degrees[:K]):
        raise ValueError("each processed degree must be a positive integer")
    if cfg.preserve and not cfg.composed:
        raise ValueError("preserve=True requires composed=True")
    buffer = ReplayBuffer(cfg.capacity, cfg.seed)
    acc = np.full((len(experiences), len(experiences)), np.nan)
    records, model = [], None

    for t in range(K):
        exp = experiences[t]
        d = degrees[t]
        seen = experiences[: t + 1]
        if model is None:
            model_before = PolynomialLogisticModel(d)
            n_old = 0
        else:
            model_before = model
            n_old = model.n_features
        expanded = model_before.expanded(d)
        ds, dp = check_expansion_invariant(
            model_before, expanded, _probes(seen, cfg.seed)
        )
        assert ds < 1e-9 and dp < 1e-9, f"expansion invariant violated: {ds}, {dp}"

        batch = build_batch(
            exp.train, buffer, cfg.m_current, cfg.m_replay, cfg.replace, cfg.seed, t
        )
        n_new_total = expanded.n_features
        mask = np.ones(n_new_total, bool)
        lr_vec = np.full(n_new_total, cfg.lr)
        if cfg.freeze_old:
            mask[:n_old] = False
        if cfg.composed:
            if cfg.scope == "new":
                mask[:n_old] = False
        else:
            lr_vec[:n_old] *= cfg.lr_old_factor
        seed_t = [cfg.seed, t]

        ref_x, null_basis, null_dim, g_w = None, None, None, None
        if cfg.composed:
            if cfg.preserve and t > 0:
                ref_x = buffer.reference_inputs(cfg.n_ref_per_exp)
                null_basis = preservation_null_basis(d, ref_x, mask)
                null_dim = null_basis.shape[1]
            res = train(
                PolynomialLogisticModel(d),
                batch.x,
                batch.y,
                lr_vec,
                cfg.epochs,
                cfg.batch_size,
                seed_t,
                mask,
                expanded.score(batch.x),
                null_basis,
            )
            after = expanded.with_added(res.model)  # f_old + g
            g_w = res.model.w.copy()
        else:
            res = train(
                expanded,
                batch.x,
                batch.y,
                lr_vec,
                cfg.epochs,
                cfg.batch_size,
                seed_t,
                mask,
            )
            after = res.model

        # ---- evaluation on every seen experience's fixed test split
        for i, e in enumerate(seen):
            acc[t, i] = accuracy(e.test.y, after.predict(e.test.x))
        fg = forgetting(acc, t)
        if t > 0:
            xo = np.concatenate([e.test.x for e in seen[:t]])
            sb, sa = model_before.score(xo), after.score(xo)
            mos, flip = old_score_change(sb, sa), prediction_flip_fraction(sb, sa)
            mean_fg, max_fg = float(fg[:t].mean()), float(fg[:t].max())
        else:
            mos = flip = mean_fg = max_fg = 0.0
        ref_chg = ref_flip = None
        if ref_x is not None and len(ref_x):
            sb, sa = expanded.score(ref_x), after.score(ref_x)
            ref_chg, ref_flip = (
                old_score_change(sb, sa),
                prediction_flip_fraction(sb, sa),
            )
        fmax, l1 = parameter_changes(expanded.w, after.w, mask)
        preserved_change = 0.0 if cfg.composed else None
        combined_change = float(np.max(np.abs(after.w - expanded.w)))

        records.append(
            StepRecord(
                t,
                model_before.degree,
                d,
                expanded.feature_names,
                model_before,
                expanded,
                after,
                g_w,
                expanded.w.copy(),
                after.w.copy(),
                res.lr_vector,
                mask,
                int(mask.sum()),
                null_dim,
                res.loss_history,
                batch.counts,
                ds,
                dp,
                acc[t, : t + 1].copy(),
                float(acc[t, t]),
                float(np.mean(acc[t, : t + 1])),
                fg,
                mean_fg,
                max_fg,
                mos,
                flip,
                ref_chg,
                ref_flip,
                fmax,
                preserved_change,
                combined_change,
                l1,
            )
        )

        buffer.add_experience(
            exp.train
        )  # only permitted (training) samples enter replay
        model = after
    return ContinualResult(cfg, list(experiences), records, acc)


def run_seeds(seeds, cfg: ContinualConfig, data_kwargs=None):
    """Repeat one configuration over several seeds (data + replay + shuffling all depend on the seed)."""
    from .data import make_experiences

    out = []
    for s in seeds:
        exps = make_experiences(seed=s, **(data_kwargs or {}))
        out.append(run_continual(exps, replace(cfg, seed=s)))
    return out
