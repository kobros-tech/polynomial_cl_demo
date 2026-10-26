# Report: polynomial expansion for continual learning (observed behaviour)

All numbers come from notebook `02_continual_learning.ipynb` (default configuration: 10 experiences, 200 training / 500 test samples each,
degree grows 1 -> 10, `lr = 0.5`, 300 full-batch epochs per experience, replay capacity 100 and `m = 100` per experience, no label noise).
"Seed 0" is one run; "5 seeds" is mean +/- std over seeds 0-4. Hyper-parameters were not tuned after seeing results.

## 1. What is guaranteed (by construction, and covered by tests)

* **Expansion invariant.** Old coefficients are copied bit-for-bit, new ones are zero, so `f_{d+1} = f_d` up to float rounding. Measured in the continual loop on train, test, random and out-of-region inputs at every expansion: max score difference <= 1.8e-12, max probability difference <= 1.4e-14 (the loop would raise above 1e-9).
* **Explicitly frozen coefficients never change.** Asserted exactly (bit equality) inside `train`. The frozen-coefficient maximum is now reported as unavailable when a strategy has no frozen coefficients, rather than using a vacuous zero. The report's historical tables predate this diagnostic correction.
* **Strategy B constraint.** `g(x_ref) = 0` holds at the reference inputs to rounding: max change at reference inputs <= 2.7e-15, zero prediction flips there, at every experience.
* **Representational identities.** Unconstrained `f_old + g` over all coefficients is the same optimisation problem as ordinary training (observed: identical rows in every table); with g restricted to new coefficients it equals `frozen_old`.
* **Not guaranteed:** that preserving `f_old` at finite reference points preserves it anywhere else, or that a preserved/frozen old component keeps the combined classifier's predictions.

## 2. What was measured (seed 0, after experience 10)

| strategy | mean acc (10 exps) | acc on exp 10 | acc on exp 1 | mean forgetting | worst forgetting seen |
|---|---|---|---|---|---|
| joint training (notebook 01, degree 10, 5000 epochs) | 0.987 | 0.986 | 0.990 | - | - |
| `ordinary_no_replay` | 0.867 | 0.986 | 0.928 | 0.122 | 0.310 |
| `ordinary` (replay) | 0.966 | 0.970 | 0.978 | 0.007 | 0.048 |
| `frozen_old` (= `strategy_a_new`) | 0.844 | 0.860 | 0.988 | 0.002 | 0.016 |
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
