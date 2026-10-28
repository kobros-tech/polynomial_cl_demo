# polynomial-cl-demo

A small NumPy-only research instrument for asking one question:

> Can incremental polynomial feature expansion (zero-initialised new coefficients, optionally with a preserved old function and replay) help a model learn sequential experiences without forgetting earlier ones?

It does **not** assume the answer is yes. Every mechanism (zero-init expansion, freezing, per-coefficient learning rates, replay, preserved old function) can be switched on and off independently, and every coefficient, score and prediction change is recorded and plottable.

> This project studies *explicit* polynomial feature expansion, which is related to polynomial kernels. It does not implement an implicit-kernel SVM.

## Setup

Only Python, NumPy, Matplotlib and Jupyter are used (no PyTorch / TensorFlow / scikit-learn / pretrained models). CPU only. The synthetic experiments and tests are lightweight; the full multi-seed Split MNIST expansion ablation takes longer.

```bash
cd polynomial-cl-demo
python -m venv .venv && source .venv/bin/activate     # optional
pip install -e ".[dev]"                                # numpy, matplotlib + pytest, jupyter, nbformat, nbconvert
python -m pytest                                       # test suite
jupyter lab notebooks/                                 # or: jupyter notebook
```

To re-execute both notebooks headlessly:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_joint_training.ipynb notebooks/02_continual_learning.ipynb
```

The notebooks fall back to `../src` if the package is not installed. They are committed without outputs (keeps the repository small and diffs clean); run them to regenerate every figure and table.

## Matched class-incremental comparison

The shared runner compares a ten-class synthetic 2D dataset with Split MNIST under five
experiences (two new classes each), fixed per-class replay examples, and the same three
capacity conditions. During each experience, both the softmax loss and evaluation restrict
class competition to classes introduced so far; future output weights receive no training
gradients. A separate diagnostic matrix evaluates with all ten classes eligible. The package owns the reusable dataset, feature construction, replay/task planning, training,
evaluation, and metric aggregation. The CLI in `experiments/continual_comparison.py` only
selects the dataset and run settings, invokes the package, and writes reproducible JSON.
Notebook `02_continual_learning.ipynb` runs the shared runner directly, keeps results in
memory, and generates its plots without reading generated JSON files. It requires all four
MNIST archives to be present locally under `data/mnist`; the notebook explicitly disables
downloads. The original sinusoidal binary experiment remains a separate research case; it is
not replaced by this multiclass dataset.

The older `experiments.split_mnist_expansion` command is retained only as a **legacy
global-ten-class protocol** for historical reproducibility. New comparisons should use
`experiments.continual_comparison`. The legacy script writes to
`results/split_mnist_expansion_legacy.json` by default, keeping new runs separate from any
historical file you may have retained locally. Generated files under `results/` are not
committed to the repository.

Run a quick synthetic smoke experiment:

```bash
python -m experiments.continual_comparison \
  --dataset synthetic_2d --seeds 0 --epochs 1 --memory-per-class 5 \
  --output results/synthetic_2d_smoke.json
```

Run the matched five-seed experiments (Split MNIST downloads data into `data/mnist`):

```bash
python -m experiments.continual_comparison \
  --dataset synthetic_2d --seeds 0 1 2 3 4 --epochs 3 \
  --output results/synthetic_2d_comparison.json

python -m experiments.continual_comparison \
  --dataset split_mnist --seeds 0 1 2 3 4 --epochs 3 \
  --output results/split_mnist_comparison_v2.json
```

The new result files use protocol version `class-incremental-v1`; existing JSON files are
historical results and are intentionally not overwritten. Synthetic 2D classes are Gaussian
clusters on a ring, so class overlap depends on the recorded separation and noise settings.
The image model uses a fixed orthonormal random projection to 16 dimensions before applying
the same explicit polynomial-feature machinery; this is a controlled comparison, not an
assertion that the projected image distribution is equivalent to 2D geometry.

## Layout

```text
src/polynomial_cl/
  data.py       ground-truth rule, per-experience train/test generation (explicit seeds)
  features.py   explicit polynomial map, deterministic ordering, names, counts
  model.py      logistic regression, stable sigmoid/BCE, zero-init expansion, invariant check
  training.py   binary gradient descent, continual loop, strategy presets, Strategy B constraint
  multiclass.py shared multiclass softmax-head trainer for class-incremental experiments
  multiclass_data.py ten-class synthetic 2D generator and task order
  multiclass_features.py fixed image projection, polynomial map, zero-init expansion
  multiclass_experiment.py shared task planning, replay, training, evaluation and aggregation
  mnist.py      MNIST IDX parsing and download/loading
  replay.py     train-only replay buffer and balanced batch construction
  metrics.py    accuracy, forgetting, old-score change, prediction flips, parameter changes
  plotting.py   regions, predictions, correctness maps, before/after, coefficients, history
tests/          unit and integration tests (see below)
notebooks/      01 joint training reference, 02 continual learning + strategy comparison
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
| `strategy_b_all` / `strategy_b_new` | as A, plus **preserved old predictions**: `g(x_ref) = 0` enforced exactly on reference inputs |

Two facts worth knowing before reading results (both are tested):

* The unconstrained composed variants are not separate presets: `composed=True, scope="all"` duplicates `ordinary`, while `composed=True, scope="new"` duplicates `frozen_old`. These aliases are intentionally omitted from the comparison.
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

`test_features` (ordering, counts, known values), `test_expansion` (exact copy, zeros, identical scores/probabilities on train/eval/random/outside inputs, invariant checked every step of the loop), `test_training` (finite-difference gradient, loss decrease, reproducibility, frozen coefficients, zero-init learning, per-coefficient lr, **preservation tests**: a frozen old component vs. a preserved combined classifier, constraint satisfied at reference points but not elsewhere, over-constrained g cannot learn, preset-override guard), `test_replay` (no evaluation data, valid labels, reproducibility, composition, insufficient-sample policy), `test_split_mnist.py` (IDX parsing and validation), and `test_split_mnist_expansion.py` (projected polynomial basis, zero-init logit invariance, deterministic paired replay plan, summary statistics, and a small no-download run of all three image conditions). The suite also includes `test_data.py` (fixed evaluation sets, noise, distributions) and `test_plotting.py` (plot functions run and never modify the model).

## Reproducibility

Everything is driven by explicit seeds (`np.random.default_rng([seed, experience, stream])`). Re-running a notebook gives the same numbers.
Hyper-parameters (`lr=0.5`, 300 epochs per experience, replay `m=100`, capacity 100) were fixed before the continual experiments were run and were not tuned afterwards; all strategies share them.

## Research follow-up: paired degree ablation and Split MNIST

The strategy comparison intentionally keeps only distinct presets. Do not add
`composed=True, scope="all"` as another name for `ordinary`, or
`composed=True, scope="new"` as another name for `frozen_old`: those are
implementation aliases, not additional experimental conditions.

Run a paired-seed synthetic ablation (same generated experiences, seed, replay
budget, epochs, and training settings within each pair):

```bash
PYTHONPATH=src python -m experiments.synthetic_degree_ablation \
  --seeds 0 1 2 3 4 \
  --n-experiences 10 --epochs 300 \
  --m-current 100 --m-replay 100 --capacity 100 \
  --output results/synthetic_degree_ablation.json
```

The script runs three conditions on the **same generated experience stream per
seed**:

* `expanding`: polynomial degree grows with each experience (degrees 1 through 10).
* `fixed_initial_degree`: degree 1 throughout; this is the literal no-expansion baseline.
* `fixed_capacity`: degree 10 from the start; this control has the same final capacity
  as `expanding` and tests whether delaying feature availability helps.

All conditions use identical replay settings, epoch counts, and seeds. The JSON
contains per-seed metrics and mean/sample-standard-deviation summaries. Interpret
`expanding` versus `fixed_initial_degree` as the value of adding capacity at all;
interpret `expanding` versus `fixed_capacity` as the value of *when* capacity is
introduced. The latter is the more stringent expansion-method comparison.

Run the separate, NumPy-only Split MNIST class-incremental reference baseline:

```bash
PYTHONPATH=src python -m experiments.split_mnist \
  --seed 0 --epochs 3 --memory-per-class 200 \
  --output results/split_mnist_seed0_reproduced.json
```

The script downloads the canonical MNIST IDX archives to `data/mnist/` on
first use. Five experiences introduce digit pairs `(0,1)`, `(2,3)`, ...,
`(8,9)`. A fixed 784-input, 10-output linear softmax classifier is trained
sequentially with a fixed per-class replay budget; evaluation reports the
accuracy matrix, final mean accuracy, and mean forgetting. This is the **raw
pixel linear reference baseline**. It does not test feature expansion.

### Controlled Split MNIST feature-expansion ablation

Run the controlled image experiment on five seeds:

```bash
PYTHONPATH=src python -m experiments.split_mnist_expansion \
  --seeds 0 1 2 3 4 \
  --epochs 3 --memory-per-class 200 \
  --output results/split_mnist_expansion_legacy_reproduced.json
```

This experiment projects each normalized 784-pixel image into a fixed,
seeded 16-dimensional representation and constructs explicit polynomial
features of total degree up to 3. The fixed projection is shared by all
conditions; it is not learned from the data. The degree groups are scaled by
the square root of their feature counts to keep higher-degree groups from
dominating solely because they contain more monomials. Feature counts are 17
(degree 1), 153 (degree 2), and 969 (degree 3), including the constant.

Three paired conditions use the same task order, training examples, per-class
replay indices, seeds, learning rate, batch size, and epochs:

* `fixed_linear`: degree 1 for every task.
* `expanding`: degree 1 on task 1, degree 2 on task 2, and degree 3 from task
  3 onward. New classifier coefficients are initialized to zero.
* `fixed_capacity`: degree 3 is available from task 1.

The comparison between `expanding` and `fixed_linear` tests whether adding
polynomial capacity helps on Split MNIST. The comparison between `expanding`
and `fixed_capacity` tests whether introducing that capacity gradually helps
compared with having the same final capacity from the beginning. The latter
is the more stringent test of the timing of expansion.

The experiment asserts that zero-initialized expansion leaves old logits
unchanged (within a small numerical tolerance) and records the maximum change
for each expansion. It reports per-task accuracy matrices, final mean accuracy,
final current-task accuracy, mean forgetting, feature counts, and mean/sample
standard-deviation summaries across seeds. It is a test of polynomial
expansion on a **fixed random projection of image inputs**, not a full
polynomial expansion of all 784 pixels and not evidence that the method
generalizes to all image datasets. Compare results with the raw-pixel
reference baseline as context, but use the three projected conditions for the
controlled causal comparison.

The diagnostic record already includes loss history, number of trainable
coefficients, coefficient updates, score changes and prediction flips on old
test splits, reference-input preservation, and Strategy B's remaining null-space
dimension. The plot `plot_learning_history` shows loss, accuracy/forgetting,
old-score changes, and trainable/free dimensions. Inspect the JSON summaries
alongside the saved figures before interpreting Strategy B: a low reference
score change alone is not evidence that the full old decision boundary is
preserved.


### Auditing historical results and running the corrected comparison

Historical `results/split_mnist_seed*.json` and
`results/split_mnist_expansion.json` files are **not included in this repository**;
`results/` is ignored because it contains generated artifacts. If you retained those
files from an earlier checkout, you can audit them with the command below. If you do not
have them locally, skip this optional historical audit; the raw matrices cannot be
reconstructed from the aggregate values in this report. The old expansion matrix uses a
global ten-way head with all output classes eligible and must not be silently compared
with the corrected seen-class-masked primary matrix.

Audit the retained local files and write the audit to a new output path:

```bash
PYTHONPATH=src python -m experiments.diagnose_split_mnist \
  results/split_mnist_seed0.json results/split_mnist_seed1.json \
  results/split_mnist_seed2.json results/split_mnist_seed3.json \
  results/split_mnist_seed4.json results/split_mnist_expansion.json \
  --output results/split_mnist_historical_audit.json
```

For the corrected shared protocol, first run the full test suite and a synthetic
smoke test, then run the image experiment to a **new** output path:

```bash
PYTHONPATH=src python -m pytest
PYTHONPATH=src python -m experiments.continual_comparison \
  --dataset synthetic_2d --seeds 0 --epochs 1 --memory-per-class 5 \
  --output results/synthetic_2d_smoke.json
PYTHONPATH=src python -m experiments.continual_comparison \
  --dataset split_mnist --seeds 0 1 2 3 4 --epochs 3 \
  --memory-per-class 200 --output results/split_mnist_comparison_v2.json
```

The corrected result records both seen-class-masked and all-classes-eligible
accuracy matrices, per-experience per-class accuracy and confusion matrices,
replay batch counts and epoch exposures by task, feature counts, expansion logit
changes, and prediction-flip counts before subsequent training.
The historical audit can verify matrix-derived metrics, but it cannot recover
per-class predictions or replay/expansion diagnostics that were not saved by
the original run. The full corrected Split MNIST run and notebook execution
must complete before the research conclusions are considered final.
