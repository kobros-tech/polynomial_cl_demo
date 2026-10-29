# polynomial-cl-demo

A small NumPy-only research instrument for asking one question:

> Can incremental polynomial feature expansion (zero-initialised new coefficients, optionally with a preserved old function and replay) help a model learn sequential experiences without forgetting earlier ones?

It does **not** assume the answer is yes. Every mechanism (zero-init expansion, freezing, per-coefficient learning rates, replay, preserved old function) can be switched on and off independently, and every coefficient, score and prediction change is recorded and plottable.

> This project studies *explicit* polynomial feature expansion, which is related to polynomial kernels. It does not implement an implicit-kernel SVM.

## Setup

Only Python, NumPy, Matplotlib and Jupyter are used (no PyTorch / TensorFlow / scikit-learn / pretrained models). CPU only; the whole project runs in about a minute.

```bash
cd polynomial-cl-demo
python -m venv .venv && source .venv/bin/activate     # optional
pip install -e ".[dev]"                                # numpy, matplotlib + pytest, jupyter, nbformat, nbconvert
python -m pytest                                       # test suite
jupyter lab notebooks/                                 # or: jupyter notebook
```

To re-execute both notebooks headlessly:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb
```

Notebook 03 skips its Split MNIST cells when `data/mnist` is absent. The notebooks fall back to `../src` if the package is not installed. They are committed without outputs (keeps the repository small and diffs clean); run them to regenerate every figure and table.
Numbers from a reference run (seed 0, plus five seeds) are summarised in [`REPORT.md`](REPORT.md).

## Layout

```text
src/polynomial_cl/
  data.py       ground-truth rule, per-experience train/test generation (explicit seeds)
  features.py   explicit polynomial map, deterministic ordering, names, counts
  model.py      logistic regression, stable sigmoid/BCE, zero-init expansion, invariant check
  training.py   gradient descent (per-coefficient lr, masks, offsets, null-space projection),
                continual loop run_continual, strategy presets, Strategy B constraint
  replay.py     train-only replay buffer and balanced batch construction
  metrics.py    accuracy, forgetting, old-score change, prediction flips, parameter changes
  plotting.py   regions, predictions, correctness maps, before/after, coefficients, history
  class_incremental/   ten-class comparison (see "Follow-up experiments"): data.py, mnist.py,
                features.py (n-variable scaled polynomial map, zero-init expansion), training.py
                (softmax head), experiment.py (protocol, conditions, summaries)
experiments/    command-line entry points that write JSON: synthetic_degree_ablation.py, class_incremental.py
tests/          52 tests (see below)
notebooks/      01 joint training reference, 02 continual learning + strategy comparison + degree ablation,
                03 class-incremental comparison
REPORT.md       observed behaviour across the ten experiences
```

## The experiment in one paragraph

Inputs live in `[-1, 1]^2`. The true rule is `y = 1 iff x2 > 0.5 sin(3 x1)` (a nonlinear boundary). Experience *k* of 10 samples `x1`
from the *k*-th vertical strip and `x2` uniformly, so each experience reveals a different part of the boundary. This is sequential binary learning under a spatially shifting input distribution, **not class-incremental learning** with new classes arriving over time. Each experience has
its own **training** split and its own fixed **test** split (independent random streams; the test split does not depend on the number
of training samples). The model is `f_d(x) = w . phi_d(x)` with the bias as the coefficient of the constant feature. Starting at degree 1,
the degree grows by one per experience (configurable via `degrees=`). Expansion appends zero coefficients and the loop **asserts**
`f_{d+1}(x) = f_d(x)` for scores and probabilities on training, test, random and out-of-region inputs before any training.

## Strategies (`make_config(name)`)

| name | meaning |
| --- | --- |
| `ordinary_no_replay` | one global learning rate, no replay |
| `ordinary` | one global learning rate, balanced replay |
| `frozen_old` | existing coefficients have mask False (never updated), new ones train |
| `small_old_lr` | existing coefficients use `lr_old_factor = 0.1` |
| `strategy_a_all` / `strategy_a_new` | `f = f_old + g`, **frozen old component**, g unconstrained (over all / only new coefficients) |
| `strategy_b_all` / `strategy_b_new` | as A, plus **preserved old predictions**: `g(x_ref) = 0` enforced exactly on reference inputs |

Two facts worth knowing before reading results (both are tested):

* `strategy_a_all` is numerically identical to `ordinary` (same function class, same starting point), and `strategy_a_new` is identical to `frozen_old`. A and "ordinary" differ in representation, not behaviour.
* Strategy B constrains `g` only at finitely many reference inputs (the first 3 buffered inputs of every old experience, configurable). Implementation: `g`'s coefficients are restricted to the null space of the reference design matrix (SVD), so `g(x_ref) = 0` holds to rounding. This says **nothing** about inputs elsewhere; the metrics measure how much old scores actually move on the full old test sets. If the constraint uses up every degree of freedom, `g` cannot learn and the run reports that (`null_dim = 0`); constraints are never relaxed automatically.
* Presets cannot be silently overridden: `make_config("ordinary_no_replay", m_replay=100)` raises.

## Replay policy

Each experience stores up to `capacity` training samples (seeded choice without replacement). A batch for experience *t* contains
`m_current` current samples plus `m_replay` samples from **each** earlier experience (int or `{exp_id: m}`), with or without replacement.
If fewer than `m` rows are stored and sampling is without replacement, all stored rows are used (never padded, never taken from elsewhere).
Actual per-experience counts are stored in `record.batch_counts`. `ReplayBuffer.add_experience` only accepts `Split(kind="train")`.

## Metrics (per experience, identical for every strategy)

current-experience accuracy; mean accuracy over observed experiences; `max |f_after - f_before|` over old test inputs; fraction of old
test predictions that flipped; forgetting = best earlier accuracy minus current accuracy (per experience); max change of frozen coefficients (reported as unavailable when no coefficients are frozen), change in the preserved old component, maximum combined-coefficient change, and L1 update of trainable coefficients; for Strategy B also the change at reference inputs and the free dimensions left by the constraint.

## Tests

`test_features` (ordering, counts, known values), `test_expansion` (exact copy, zeros, identical scores/probabilities on train/eval/random/outside inputs, invariant checked every step of the loop), `test_training` (finite-difference gradient, loss decrease, reproducibility, frozen coefficients, zero-init learning, per-coefficient lr, **preservation tests**: a frozen old component vs. a preserved combined classifier, constraint satisfied at reference points but not elsewhere, over-constrained g cannot learn, preset-override guard), `test_replay` (no evaluation data, valid labels, reproducibility, composition, insufficient-sample policy), plus two small additions to the suggested layout: `test_data.py` (fixed evaluation sets, noise, distributions) and `test_plotting.py` (plot functions run and never modify the model), `test_class_incremental.py` / `test_mnist.py` (features, expansion invariant and its guard, masked evaluation, forgetting definition, determinism, offline Split MNIST), `test_experiment_scripts.py` (both command-line tools) and `test_notebooks.py` (every notebook compiles and is committed without outputs).

## Reproducibility

Everything is driven by explicit seeds (`np.random.default_rng([seed, experience, stream])`). Re-running a notebook gives the same numbers.
Hyper-parameters (`lr=0.5`, 300 epochs per experience, replay `m=100`, capacity 100) were fixed before the continual experiments were run and were not tuned afterwards; all strategies share them.

## Follow-up experiments: does expanding the polynomial degree help?

Generated files go to `results/` (git-ignored). Both tools need the package installed (`pip install -e .`).

**1. Binary sinusoid task, paired degree ablation** (the setting above; 10 experiences, 300 epochs):

```bash
python experiments/synthetic_degree_ablation.py --seeds 0 1 2 3 4 --output results/synthetic_degree_ablation.json
```

Within each seed the same generated experiences, replay budget, epochs and optimiser are used for `expanding` (degree 1..10), `fixed_initial_degree` (degree 1, no expansion) and `fixed_capacity` (degree 10 from the start). Expanding vs degree 1 measures the value of adding capacity at all; expanding vs `fixed_capacity` measures the value of adding it gradually.

**2. Class-incremental comparison on a ten-class ring and on Split MNIST** (five experiences of two new classes; replay of a fixed number of examples per class; one ten-output softmax head):

```bash
python experiments/class_incremental.py --dataset synthetic_2d --output results/synthetic_2d.json
python experiments/class_incremental.py --dataset split_mnist  --output results/split_mnist.json   # downloads MNIST to data/mnist
```

Conditions `fixed_linear` (degree 1), `expanding` (degrees 1, 2, 3, 3, 3; new coefficients zero-initialised, logit change asserted below 1e-5) and `fixed_capacity` (degree 3 throughout) share task order, replay indices, seed, optimiser and epochs. During training and evaluation only classes introduced so far compete, so output weights of future classes are never trained. MNIST images are normalised and mapped to 16 dimensions by a fixed random orthonormal projection (never fitted to data) before the explicit polynomial map (17, 153, 969 features at degrees 1, 2, 3); each degree group is divided by the square root of its size. Forgetting is the best accuracy an experience ever had minus its final accuracy, averaged over experiences 1-4 (the last has no later evaluation). `notebooks/03_class_incremental.ipynb` runs the same code and plots the accuracy matrices.

Read these results with two caveats (numbers in [`REPORT.md`](REPORT.md) section 6):

* **Training budget.** The defaults (3 epochs, learning rate 0.1) are far from convergence. Always repeat with several `--epochs` values: conclusions that change with the budget are about optimisation speed, not capacity.
* **The ring is linearly solvable.** Equal isotropic Gaussian classes have linear Bayes boundaries, so this dataset cannot reward extra capacity. It checks the machinery; the binary sinusoid task and Split MNIST are the capacity tests. Also, with the default 160 training samples per class the replay memory (200 per class) holds *all* earlier data on the ring, whereas it is a genuine 200-example budget on MNIST.
