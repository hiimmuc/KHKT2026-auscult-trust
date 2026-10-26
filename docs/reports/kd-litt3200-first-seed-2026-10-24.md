# First Litt3200 distillation result

[Preliminary] The 15:00 local sync on 24 October adds seed 0 of the Litt3200 KD pilot. A now holds completion markers for 27/100 new dynamic-SC units and 4/6 KD units. The three official KD seeds were already reported in the [earlier matched analysis](synced-three-seed-lodo-progress-2026-10-24.md). Litt3200 seeds 1-2 remain pending; this is a one-seed observation, with no pilot verdict or seed-averaged LODO result.

## Matched classification and conformal results

| Model | Sp | Se | Score | HS | Coverage | Set size | Singleton |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KD student | 60.61 | 47.68 | 54.14 | 53.37 | 0.990 | 3.389 | 0.008 |
| PAFA/static control | 63.94 | 44.37 | 54.16 | 52.39 | 0.988 | 3.337 | 0.027 |
| U_BEATs3/static teacher | 37.88 | 61.59 | 49.73 | 46.91 | 0.973 | 2.740 | 0.048 |

Each model retains its frozen fixed epoch; all use the same held-out-device cohort and source-patient calibration cohort. Teacher predictions are the raw equal-weight softmax mean; no k-rate adjustment is applied. Conformal uses source-calibrated LAC at alpha 0.1. All three have zero empty sets. SC retains the existing full-unlabelled-target-pool coefficients, so this is not a finite-target-budget result.

| Contrast | Score difference | Paired patient 95% interval | Set-size difference | Paired patient 95% interval |
| --- | ---: | --- | ---: | --- |
| Student minus control | -0.01 | [-4.06, +3.63] | +0.052 | [-0.091, +0.259] |
| Student minus teacher | +4.41 | [-2.20, +10.41] | +0.649 | [+0.264, +1.059] |

Intervals use 1,000 paired whole-patient draws of calibration and evaluation cohorts, independently resampled, with recalibration in every draw. Every model uses the same sampled rows within a draw. All 1,000 draws are estimable for the reported contrasts. These are pointwise exploratory intervals conditional on one trained seed; they exclude model-seed uncertainty. The held-out cohort has only ten patients.

[Interpretation] The student has approximately the same Score as its hard-label control in this seed. Its higher descriptive Score than the teacher comes with higher specificity and lower sensitivity: student minus teacher is +22.73 Sp points and -13.91 Se points. The corresponding patient intervals are [+14.02, +32.61] and [-25.64, -5.56]. This is a substantial decision tradeoff, not a uniform improvement. High conformal coverage is achieved with sets averaging more than three of four classes and less than 1% singletons for the student; coverage alone is not evidence of useful device robustness.

The proposed teacher-gap engineering target is met by the seed-0 point estimate, but its Score interval extends below -1 point. Together with the incomplete three-seed fold and [unmeasured runtime/VRAM](kd-runtime-preflight-2026-10-24.md), this cannot establish the pilot target, superiority or equivalence. Continue the frozen pilot; do not choose a teacher, loss, epoch or SC mode using this held-out result. The selected serving model remains BEATs + PAFA + static SC.

## Integrity and reproduction

[Fact] Reconstructed patient partitions contain 92 training, 22 calibration and 10 test patients, with 4,773/1,378/481 cycles. The three patient groups are disjoint. Annotation-order unique cycle IDs, labels and patient order match across student, control and every teacher; calibration-device order also matches. Teacher configuration/report/SC hashes match the previously frozen teacher-input manifest. Student and control recipe arguments differ only in cell identity, soft-target input and loader workers (2 versus 4). Exact initialization/numerical trajectories were not replayed.

The new snapshot audits all 31 currently completed local units. Present completion-bound bytes were verified; absent checkpoint and original-input paths remain explicit in the audit. Checkpoints and teacher-target exports are still required for full KD reload/export certification. Predictions alone are sufficient for this conditional prediction analysis.

Artifacts are in `outputs/evaluation/report-20261024/synced-progress-1500/`: `summary.json`, `comparisons.json`, `lodo-kd.json`, scripts and logs. Earlier snapshots are preserved. Reproduce in a new artifact directory, updating its snapshot paths before execution; `lodo_kd.py` refuses an existing result:

```bash
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 PYTHONPATH=. .venv-train/bin/python \
  outputs/evaluation/report-20261024/synced-progress-1500/analyse.py
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 PYTHONPATH=. .venv-train/bin/python \
  outputs/evaluation/report-20261024/synced-progress-1500/comparisons.py
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 PYTHONPATH=. .venv-train/bin/python \
  outputs/evaluation/report-20261024/synced-progress-1500/lodo_kd.py
```

No incomplete-device dynamic-SC macro is reported. The two previously complete five-seed AST Patch-Mix CL folds retain their earlier opposite-direction contrasts against static SC: Litt3200 +3.35 Score points, LittC2SE -2.88, both with intervals spanning zero. Newly available partial arms stay in the artifact without a five-seed headline.
