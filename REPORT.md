# Report: polynomial expansion for continual learning (observed behaviour)

All numbers come from notebook `02_continual_learning.ipynb` (default configuration: 10 experiences, 200 training / 500 test samples each,
degree grows 1 -> 10, `lr = 0.5`, 300 full-batch epochs per experience, replay capacity 100 and `m = 100` per experience, no label noise).
"Seed 0" is one run; "5 seeds" is mean +/- std over seeds 0-4. Hyper-parameters were not tuned after seeing results.

## 1. What is guaranteed (by construction, and covered by tests)

* **Expansion invariant.** Old coefficients are copied bit-for-bit, new ones are zero, so `f_{d+1} = f_d` up to float rounding. Measured in the continual loop on train, test, random and out-of-region inputs at every expansion: max score difference <= 1.8e-12, max probability difference <= 1.4e-14 (the loop would raise above 1e-9).
* **Explicitly frozen coefficients never change.** Asserted exactly (bit equality) inside `train`. The frozen-coefficient maximum is now reported as unavailable when a strategy has no frozen coefficients, rather than using a vacuous zero. The report's historical tables predate this diagnostic correction.
* **Strategy B constraint.** `g(x_ref) = 0` holds at the reference inputs to rounding: max change at reference inputs <= 2.7e-15, zero prediction flips there, at every experience.
* **Representational identities.** Unconstrained `f_old + g` over all coefficients is the same optimisation problem as ordinary training, and with g restricted to new coefficients it equals `frozen_old` (tested; the former `strategy_a_*` presets were removed because they only duplicated these two rows).
* **Not guaranteed:** that preserving `f_old` at finite reference points preserves it anywhere else, or that a preserved/frozen old component keeps the combined classifier's predictions.

## 2. What was measured (seed 0, after experience 10)

| strategy | mean acc (10 exps) | acc on exp 10 | acc on exp 1 | mean forgetting | worst forgetting seen |
| --- | --- | --- | --- | --- | --- |
| joint training (notebook 01, degree 10, 5000 epochs) | 0.987 | 0.986 | 0.990 | - | - |
| `ordinary_no_replay` | 0.867 | 0.986 | 0.928 | 0.122 | 0.310 |
| `ordinary` (replay) | 0.966 | 0.970 | 0.978 | 0.007 | 0.048 |
| `frozen_old` | 0.844 | 0.860 | 0.988 | 0.002 | 0.016 |
| `small_old_lr` (0.1x) | 0.915 | 0.922 | 0.950 | 0.016 | 0.068 |
| `strategy_b_all` | 0.855 | 0.934 | 0.958 | 0.000 | 0.002 |
| `strategy_b_new` | 0.842 | 0.844 | 0.952 | 0.000 | 0.000 |

Five seeds (final mean accuracy / forgetting): `ordinary_no_replay` 0.866+/-0.008 / 0.125+/-0.008; `ordinary` 0.965+/-0.004 / 0.008+/-0.001; `frozen_old` 0.842+/-0.006 / 0.002+/-0.001; `small_old_lr` 0.916+/-0.003 / 0.019+/-0.003; `strategy_b_all` 0.840+/-0.016 / 0.000; `strategy_b_new` 0.844+/-0.004 / 0.000. Across seeds the groups separate as `ordinary` > `small_old_lr` > {`ordinary_no_replay`, `frozen_old`, `strategy_b_*`}; the last group is not cleanly separable (e.g. `strategy_b_all` is above `frozen_old` for seed 0 but not over five seeds).

The joint row is a reference only: it uses 5000 epochs on the pooled data, while the continual runs use 300 epochs per experience, so the 0.987 vs 0.966 gap mixes the effect of sequential learning with a different optimisation budget and should not be read as the cost of continual learning.

## 3. Findings

1. **Zero-initialised expansion alone does not prevent forgetting.** Without replay, expansion is invariant-preserving but the next round of training moves the old function freely: mean forgetting 0.122 (0.125+/-0.008 over seeds), worst single-experience decline 0.31, up to 11% of old test predictions flipped in one step, final accuracy on experience 1 0.928.
2. **In this setup balanced replay did most of the work.** Adding replay to the same optimiser lifts final mean accuracy from 0.867 to 0.966 and cuts mean forgetting to 0.007, while keeping current-experience accuracy comparable (0.97). Caveat: the replay budget is generous (100 of 200 samples per old experience, equal weight to every experience); I did not sweep it.
3. **Small forgetting in accuracy is compatible with large changes of the function.** Under `ordinary`, the maximum change of an old score after a step is 1.3-3.5 (raw logit units) at every experience, yet only 0.5-3.5% of old predictions flip. Accuracy and predicted labels are far coarser than "the old function stays unchanged"; the old function does **not** stay unchanged here.
4. **Freezing or slowing old coefficients buys stability at the price of learning.** `frozen_old` has the smallest forgetting among the unconstrained methods (0.002) but cannot fit later strips: accuracy on the current experience falls to 0.71, 0.65, 0.69 at experiences 7-9 and training loss rises from 0.14 to 0.36. `small_old_lr` sits in between (0.915). I did not test why frozen new coefficients are insufficient (capacity of the new monomials, conditioning, or too few epochs are all untested explanations).
5. **Strategy B gives the stability it constrains, not full preservation, and costs plasticity.** The reference inputs are preserved to ~1e-15, and forgetting is essentially zero (max 0.002). But the old function is not preserved on the full old test sets: max score change 0.8 at experience 2 and still 0.33 at experience 10 (flip fraction up to 0.6%). So "preserved at reference points" is not "preserved". Meanwhile `strategy_b_all` learns later experiences poorly (current accuracy 0.82, 0.70, 0.65, 0.74 at experiences 6-9), very similar to `frozen_old`; over five seeds its final mean accuracy (0.840+/-0.016) is indistinguishable from `frozen_old` (0.842+/-0.006). Its higher accuracy on the final experience (0.913 vs 0.849 over seeds) is a single time point, not evidence of better learning overall.
6. **Over-constrained Strategy B cannot learn, and the run reports it.** With g restricted to new coefficients and 3 reference points per old experience, the free dimensions are 0 from experience 2 onward: no coefficient changes after experience 1 (trainable L1 update 0), the model stays the experience-1 line, and final mean accuracy is 0.842. Forgetting is exactly zero because nothing is learned. Zero forgetting must not be read as success.
7. **No measurable benefit from growing the degree.** Starting at degree 10 instead of growing 1 -> 10 (same replay and budget, 5 seeds) gives final mean accuracy 0.970+/-0.004 vs 0.965+/-0.004 and mean current-experience accuracy 0.952 vs 0.936. In this experiment incremental expansion neither helped nor clearly hurt.

## 4. Conclusion supported by the evidence

* **Retaining old performance:** achieved by balanced replay (`ordinary`), and by freezing / constraining (`frozen_old`, Strategy B), but the latter only by sacrificing learning of new experiences.
* **Learning new experiences while retaining old performance at the same time:** shown only by `ordinary` with replay (mean 0.965, current 0.953, forgetting 0.008 over seeds). That success is attributable to replay and the free update of old coefficients, not to zero-initialised expansion, freezing or the preservation constraint. The proposed Strategy B did **not** show a combined advantage in this experiment.
* The result does not show that polynomial expansion solves catastrophic forgetting.

## 5. Limitations

One synthetic dataset (strips of a sinusoidal boundary), where each experience is a spatial region; no label noise runs were reported; no sweeps over learning rate, epochs, replay size, number or placement of reference points; the optimiser (plain gradient descent on monomials of `[-1,1]` inputs, 300 epochs) is far from converged for high degrees, so conclusions about "capacity" versus "optimisation" are untested. Strategy B was tested with only 3 reference points per old experience; more references reduce learnability further (fewer free dimensions), fewer preserve less.

## 6. Follow-up: degree ablation and class-incremental comparison

Produced by `experiments/synthetic_degree_ablation.py` and `experiments/class_incremental.py` (five seeds, defaults, final mean accuracy / mean forgetting, mean +/- sample std). Seeds change task-internal shuffling, replay selection and (binary task only) the generated data; for the class-incremental runs the data, test sets and projection are fixed, so their spread is small.

| experiment | no expansion | expanding | fixed final capacity |
| --- | --- | --- | --- |
| binary sinusoid, 300 epochs (degree 1 / grow 1..10 / degree 10) | 0.910+/-0.004 / 0.043 | 0.965+/-0.004 / 0.008 | 0.970+/-0.005 / 0.011 |
| ring, 3 epochs (degree 1 / 1,2,3 / 3) | 0.709+/-0.002 / 0.008 | 0.909+/-0.001 / 0.023 | 0.908+/-0.001 / 0.025 |
| ring, 30 epochs, seeds 0-2 | 0.906 | 0.910 | 0.908 |
| Split MNIST, 3 epochs | 0.176+/-0.000 / 0.926 | 0.246+/-0.002 / 0.858 | 0.256+/-0.001 / 0.846 |
| Split MNIST, 30 epochs, seed 0 | 0.337 | 0.481 | 0.484 |

* **Binary task: capacity matters, gradual growth does not.** Degree 1 cannot represent the sinusoidal boundary (0.910); growing and fixed degree 10 are within one standard deviation of each other (the same conclusion as finding 7).
* **Ring: no capacity effect.** The 3-epoch gap (0.709 vs 0.909) disappears with training (0.906 / 0.910 / 0.908 at 30 epochs), as expected for a linearly solvable task: it reflects how fast each model learns a new class from zero-initialised weights in a fixed number of steps.
* **Split MNIST: expansion helps, gradual growth does not matter.** The ordering `fixed_linear` < `expanding` ~ `fixed_capacity` holds from 1 to 30 epochs, but every number is far from converged, forgetting is severe in all conditions (mostly because 200 replay examples per old class compete with about 12,000 fresh examples per new experience), and absolute values depend strongly on the epoch budget. Only five seeds that share one projection and one training set were run.

## 7. Strategy questions at default shared rates (superseded in part by section 8)

Five seeds, 3 epochs for Split MNIST (`--experiment strategies`), 300 epochs for the binary task; mean over seeds. `old_flip` is the fraction of old-class (binary: old-experience) test inputs whose prediction changed during the latest experience.

| rule | binary: final mean acc | binary: old-function change | Split MNIST: final mean acc | MNIST: current-exp acc | MNIST: forgetting | MNIST: old_flip |
| --- | --- | --- | --- | --- | --- | --- |
| `ordinary_no_replay` | 0.866 | - | 0.192 | 0.902 | 0.925 | 0.139 |
| `ordinary` | **0.965** | 2.470 | 0.246 | 0.902 | 0.858 | 0.385 |
| `small_old_lr` | 0.916 | 1.134 | 0.260 | 0.897 | 0.838 | 0.078 |
| `frozen_old` | 0.842 | 0.707 | 0.274 | 0.893 | 0.820 | 0.021 |
| Strategy B, 3 refs (default) | 0.840 | 0.302 | **0.538** | 0.683 | **0.405** | 0.074 |

Strategy B versus the number of reference inputs (Split MNIST: per old class; binary: per old experience):

| refs | 1 | 2 / 3 | 5 / 10 | 30 | 100 | 200 |
| --- | --- | --- | --- | --- | --- | --- |
| binary final mean acc (1, 2, 3, 5 refs) | 0.912 | 0.848 / 0.840 | 0.847 | - | - | - |
| Split MNIST final mean acc | 0.417 | 0.538 (3) | 0.622 (10) | 0.594 | 0.278 | 0.199 |
| Split MNIST current-exp acc | 0.828 | 0.683 (3) | 0.434 (10) | 0.151 | 0.004 | 0.000 |

* **Aliases.** `strategy_a_all` / `strategy_a_new` were identical to `ordinary` / `frozen_old` and have been removed.
* **Why Strategy B fits the sine curve badly.** Not optimisation: 10x the epochs gives 0.839 +/- 0.031 versus 0.840. It is the constraint: pinning `g` at 3 points per old experience leaves `null_dim` growing (3 -> 39) but almost none of it usable in the current strip (`loss_reduction` is ~8e-3 at experiences 2-3 and never exceeds 4e-3 afterwards, 1e-4 at experience 4). Accuracy falls monotonically with the number of references (0.965 ordinary, 0.912, 0.848, 0.840), and at 5 references the old function is preserved almost exactly (change 0.014) at the same accuracy as `frozen_old` (0.842), which moves the old function far more (0.707). `strategy_b_new` has `null_dim = 0` from experience 2 on: zero loss reduction and zero update, i.e. it does not learn after experience 1.
* **Learning indicators.** `null_dim` is only defined for Strategy B (it is `None`/`n/a` elsewhere, not "not learning"). For every rule use `loss_reduction`, `trainable_l1_update` / `n_trainable`, current-experience accuracy and the old-function change columns, in notebook 02 and in the class-incremental records.
* **Expansion versus no expansion is not a bug.** Degree 1 throughout is clearly worse on the binary task (0.910 vs 0.965) and on Split MNIST (0.176 vs 0.246). Growing 1..10 versus fixed degree 10 differ early (after experience 1: 0.959 vs 0.988, the fixed model is better) and converge by experiences 8-10, because zero-initialised expansion preserves the function and replay covers all experiences: once every coefficient exists the end state is set by the data, not the growth path.
* **Split MNIST at default rates (3 epochs, lr 0.1) favoured Strategy B** (0.538 vs 0.246), but section 8 shows this was an under-training artifact: with tuned rates the gap vanishes. Strategy B remains a stability-plasticity dial (more references -> less forgetting, less learning; none left to learn when every stored example is a reference).

## 8. Per-feature learning rates change the conclusions

Rate of feature i = `lr * (max_rms / rms_i) ** p` (rms on the experience's training set; `p = 0` is one shared rate, `p = 1` diagonal preconditioning). Implemented as `lr_feature_power` (binary) and `feature_power` (class-incremental). A latent crash for integer `lr` combined with fractional factors was fixed on the way.

**Binary task (five seeds, 300 epochs).** The shared rate is far from converged (ordinary: 0.967 at 300, 0.979 at 1,000, 0.991 at 10,000 epochs). With `p = 1, lr = 4` ordinary reaches 0.992 in 300 epochs with forgetting 0.0017 (vs 0.0080).

| rule | shared rate (p=0, lr=0.5) | per-feature (p=1, lr=4) |
| --- | --- | --- |
| degree 1 throughout (no expansion) | 0.910 | 0.910 |
| grow 1..10 | 0.965 | 0.992 |
| fixed degree 10 | 0.970 | 0.991 |
| `ordinary_no_replay` | 0.866 | 0.892 |
| `small_old_lr` (old factor 0.3 / 0.1) | 0.940 / 0.916 | 0.987 / 0.977 |
| `frozen_old` | 0.842 | 0.826 |

**Split MNIST (3 epochs).** For each condition or rule, `(p, lr)` was selected on a 10% validation split of the training data (grid p in {0,1,2,3}, lr in {0.01..1}, seeds 0-1), then evaluated on the test set (seeds 0-4); `experiments/rate_selection.py` reproduces this. Validation and test agree (e.g. 0.630 vs 0.644 for `expanding`).

| condition / rule | default rates | selected (p, lr) | test accuracy | forgetting |
| --- | --- | --- | --- | --- |
| `fixed_linear` | 0.176 | 3, 0.3 | 0.442 +/- 0.024 | 0.586 |
| `expanding` (= `ordinary`) | 0.246 | 3, 0.01 | 0.644 +/- 0.013 | 0.374 |
| `fixed_capacity` | 0.256 | 3, 0.01 | 0.645 +/- 0.011 | 0.377 |
| Strategy B, 3 refs | 0.538 | 2, 0.3 | 0.650 +/- 0.013 | 0.369 |
| `small_old_lr` | 0.260 | 2, 1 | 0.635 +/- 0.040 | 0.370 |
| `frozen_old` | 0.274 | 3, 0.01 | 0.578 +/- 0.014 | 0.453 |
| `ordinary_no_replay` | 0.192 | 1, 0.01 | 0.212 | 0.899 |

* **Expansion helps, gradual growth does not matter.** Without expansion the model is capacity-limited, not rate-limited: 0.910 on the binary task and 0.442 on Split MNIST even with tuned rates, versus 0.99 and 0.644 with expansion. `expanding` and `fixed_capacity` are tied in accuracy on both tasks; growing has lower forgetting on the binary task (0.0017 vs 0.0043) and none on Split MNIST.
* **Strategy B has no demonstrated advantage.** With rates tuned for each rule, B (0.650) and `ordinary` (0.644) differ by less than one standard deviation. The earlier B advantage (0.538 vs 0.246) came from comparing against an under-trained shared-rate baseline. On the binary task B is worse (0.840 vs 0.965 at shared rates; more epochs do not help).
* **`frozen_old` is worse** on both tasks even when tuned (binary 0.826, Split MNIST 0.578). `small_old_lr` matches `ordinary` but with more seed variance and its selected rate sits at the upper edge of the grid. Replay is the dominant ingredient (0.212 without it).
* **Decision.** Keep `ordinary` with per-feature rates as the default. Do not refine Strategy B for accuracy; keep it only as a controllable preservation diagnostic. Treat the earlier shared-rate tables (sections 6-7) as optimiser-limited.

**Limits.** Split MNIST is still 3 epochs and one dataset with one random projection (seeds change shuffling and replay only); the selected settings are close to others on the grid (e.g. `expanding` 0.630 at p=3, lr 0.01 vs 0.625 at p=2, lr 0.1); B was selected with 3 references only; Strategy B with per-feature rates was not tested on the binary task (that trainer requires one shared rate), where 10x more epochs left it unchanged.
