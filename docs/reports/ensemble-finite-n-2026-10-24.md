# Exploratory ensemble and finite-target-batch evidence, 24 October 2026

[Fact] Reanalysis of existing fixed-epoch predictions; no new model fitting or model selection. The selected AuscultTrust remains BEATs + PAFA trained/evaluated with static SC. Registered ensemble identities follow Amendment 4; user-group identities are distinct. These are exploratory findings, not a new confirmatory test.

## Registered equal-weight ensembles

Five seeds, four LODO folds. Macro is the equal-device average within seed, then mean and sample SD across five seeds. Score, Sp, Se and HS are percentage points. Coverage and set statistics are fractions. No member weights or epochs are fitted.

| Group | Macro Score +/- seed SD | Macro without AKGC417L | dScore vs PAFA | Paired patient/seed 95% interval |
| --- | ---: | ---: | ---: | ---: |
| R_E1_all11 | 60.51 +/- 1.45 | 61.14 | +1.72 | [-0.72, +4.26] |
| R_E2_beats5 | 61.11 +/- 1.43 | 61.76 | +2.32 | [-0.13, +4.88] |
| R_E3_ast3 | 57.85 +/- 1.12 | 58.55 | -0.94 | [-3.63, +1.77] |
| R_E4_htsat3 | 56.99 +/- 1.94 | 57.16 | -1.80 | [-4.61, +1.37] |

[Interpretation] The BEATs-only five-member ensemble has the highest descriptive registered-group macro. Every paired patient/seed interval includes zero. Seed-only intervals in the original summary omit patient uncertainty and must not support claims of established superiority. These comparisons are exploratory, pointwise, and not corrected for multiple comparisons.

Bootstrap: 1,000 percentile draws, RNG seed 20261024. Resample the five model seeds with replacement, shared across the four folds. Use one shared multinomial draw over the union of target-patient IDs across folds/seeds; preserve all cycles of each sampled patient. Pair each ensemble and PAFA on the same patient/seed draws. This retains repeated-patient dependence across folds. The estimand is the fixed-four-device macro Score; it does not generalize to a population of devices. Both normal and abnormal groups were present in every evaluated draw. This Score bootstrap does not retrain models or quantify conformal uncertainty.

### Per-fold means over five seeds

| Group | Device | Sp | Se | Score | HS | Coverage | Mean abs(Delta) | Set size | Singleton | Empty |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R_E1_all11 | Meditron | 89.277 | 31.445 | 60.361 | 46.445 | 0.942 | 0.024 | 2.002 | 0.273 | 0.000 |
| R_E1_all11 | LittC2SE | 94.928 | 37.085 | 66.006 | 53.209 | 0.933 | 0.035 | 1.966 | 0.356 | 0.000 |
| R_E1_all11 | Litt3200 | 56.727 | 57.351 | 57.039 | 55.494 | 0.956 | 0.042 | 2.544 | 0.067 | 0.000 |
| R_E1_all11 | AKGC417L | 89.958 | 27.309 | 58.634 | 41.861 | 0.772 | 0.031 | 1.898 | 0.415 | 0.000 |
| R_E2_beats5 | Meditron | 85.703 | 33.353 | 59.528 | 47.930 | 0.952 | 0.035 | 2.357 | 0.193 | 0.000 |
| R_E2_beats5 | LittC2SE | 92.853 | 39.757 | 66.305 | 55.536 | 0.938 | 0.042 | 2.188 | 0.308 | 0.000 |
| R_E2_beats5 | Litt3200 | 66.970 | 51.921 | 59.445 | 56.017 | 0.985 | 0.079 | 3.183 | 0.025 | 0.000 |
| R_E2_beats5 | AKGC417L | 85.021 | 33.342 | 59.181 | 47.869 | 0.811 | 0.027 | 2.093 | 0.360 | 0.000 |
| R_E3_ast3 | Meditron | 84.375 | 32.197 | 58.286 | 46.555 | 0.940 | 0.026 | 2.312 | 0.230 | 0.000 |
| R_E3_ast3 | LittC2SE | 95.562 | 34.332 | 64.947 | 50.376 | 0.926 | 0.050 | 2.229 | 0.323 | 0.000 |
| R_E3_ast3 | Litt3200 | 51.333 | 53.510 | 52.422 | 50.858 | 0.956 | 0.064 | 2.811 | 0.050 | 0.000 |
| R_E3_ast3 | AKGC417L | 90.429 | 21.050 | 55.740 | 34.104 | 0.794 | 0.022 | 2.366 | 0.260 | 0.000 |
| R_E4_htsat3 | Meditron | 84.941 | 32.890 | 58.916 | 47.159 | 0.942 | 0.021 | 2.180 | 0.207 | 0.000 |
| R_E4_htsat3 | LittC2SE | 91.585 | 35.223 | 63.404 | 50.739 | 0.934 | 0.036 | 2.161 | 0.248 | 0.000 |
| R_E4_htsat3 | Litt3200 | 38.303 | 60.000 | 49.152 | 43.775 | 0.911 | 0.028 | 2.673 | 0.114 | 0.000 |
| R_E4_htsat3 | AKGC417L | 88.128 | 24.878 | 56.503 | 38.624 | 0.743 | 0.054 | 1.860 | 0.442 | 0.000 |

### Fold-level null gate

Use the existing LODO convention: average abs(Delta) and the 200-re-split null p95 over the five seeds, then compare at device level. A single seed exceeding its null is not a tested fold. The source-patient re-split null is the existing implementation; exact target-size matching and class-support sensitivity have not been independently validated here.

| Group | Device | Mean abs(Delta) | Mean null p95 | Tested fold |
| --- | --- | ---: | ---: | --- |
| R_E1_all11 | Meditron | 0.0242 | 0.1395 | no |
| R_E1_all11 | LittC2SE | 0.0352 | 0.1452 | no |
| R_E1_all11 | Litt3200 | 0.0422 | 0.1197 | no |
| R_E1_all11 | AKGC417L | 0.0315 | 0.1092 | no |
| R_E2_beats5 | Meditron | 0.0347 | 0.1496 | no |
| R_E2_beats5 | LittC2SE | 0.0420 | 0.1434 | no |
| R_E2_beats5 | Litt3200 | 0.0788 | 0.1122 | no |
| R_E2_beats5 | AKGC417L | 0.0274 | 0.1084 | no |
| R_E3_ast3 | Meditron | 0.0257 | 0.1216 | no |
| R_E3_ast3 | LittC2SE | 0.0503 | 0.1212 | no |
| R_E3_ast3 | Litt3200 | 0.0640 | 0.1088 | no |
| R_E3_ast3 | AKGC417L | 0.0217 | 0.0955 | no |
| R_E4_htsat3 | Meditron | 0.0215 | 0.1253 | no |
| R_E4_htsat3 | LittC2SE | 0.0363 | 0.1301 | no |
| R_E4_htsat3 | Litt3200 | 0.0283 | 0.1026 | no |
| R_E4_htsat3 | AKGC417L | 0.0545 | 0.1176 | no |

[Fact] No registered ensemble has a tested fold. This analysis supplies no H2 verdict and does not demonstrate that SC improves coverage. The registered AST study and its historical results remain separate.

### Conformal uncertainty with recalibration

[Fact] Added 1,000 paired hierarchical bootstrap draws per ensemble/device: resample five model seeds, then calibration and target patients within each selected seed. Recompute the LAC threshold and calibration prior in every draw; ensemble and PAFA use identical resampled rows. These are pointwise exploratory per-device intervals. The tested-fold gate above remains fixed; these all-fold descriptive contrasts are not an H2 statistic or simultaneous coverage guarantee.

| Group | Device | Coverage, 95% interval | Set size, 95% interval | abs(Delta) reduction vs PAFA, 95% interval |
| --- | --- | --- | --- | --- |
| R_E1_all11 | Meditron | 0.942 [0.908, 0.962] | 2.002 [1.722, 2.279] | 0.011 [-0.029, 0.049] |
| R_E1_all11 | LittC2SE | 0.933 [0.891, 0.955] | 1.966 [1.672, 2.202] | 0.083 [0.007, 0.127] |
| R_E1_all11 | Litt3200 | 0.956 [0.897, 0.984] | 2.544 [2.125, 2.908] | 0.001 [-0.060, 0.091] |
| R_E1_all11 | AKGC417L | 0.772 [0.669, 0.855] | 1.898 [1.433, 2.360] | 0.001 [-0.027, 0.024] |
| R_E2_beats5 | Meditron | 0.952 [0.921, 0.972] | 2.357 [1.971, 2.707] | 0.001 [-0.035, 0.050] |
| R_E2_beats5 | LittC2SE | 0.938 [0.896, 0.958] | 2.188 [1.856, 2.441] | 0.076 [0.006, 0.122] |
| R_E2_beats5 | Litt3200 | 0.985 [0.953, 0.996] | 3.183 [2.654, 3.470] | -0.035 [-0.054, 0.088] |
| R_E2_beats5 | AKGC417L | 0.811 [0.713, 0.907] | 2.093 [1.568, 2.767] | 0.005 [-0.024, 0.028] |
| R_E3_ast3 | Meditron | 0.940 [0.908, 0.961] | 2.312 [2.003, 2.619] | 0.010 [-0.033, 0.053] |
| R_E3_ast3 | LittC2SE | 0.926 [0.880, 0.948] | 2.229 [1.859, 2.575] | 0.068 [0.009, 0.111] |
| R_E3_ast3 | Litt3200 | 0.956 [0.884, 0.990] | 2.811 [2.241, 3.200] | -0.020 [-0.071, 0.074] |
| R_E3_ast3 | AKGC417L | 0.794 [0.684, 0.882] | 2.366 [1.695, 2.960] | 0.011 [-0.025, 0.033] |
| R_E4_htsat3 | Meditron | 0.942 [0.911, 0.963] | 2.180 [1.916, 2.492] | 0.014 [-0.028, 0.053] |
| R_E4_htsat3 | LittC2SE | 0.934 [0.892, 0.960] | 2.161 [1.878, 2.421] | 0.082 [0.005, 0.135] |
| R_E4_htsat3 | Litt3200 | 0.911 [0.819, 0.956] | 2.673 [2.232, 2.972] | 0.015 [-0.068, 0.102] |
| R_E4_htsat3 | AKGC417L | 0.743 [0.654, 0.822] | 1.860 [1.491, 2.234] | -0.022 [-0.055, 0.024] |

[Interpretation] AKGC417L remains a substantial pooled-coverage failure even when prior-matched abs(Delta) is small. Report both quantities: matching the source class mix can hide a different pooled target coverage question. No per-device distribution-free guarantee follows under shift.

### Secondary official-split fixed-epoch comparison

All rows are raw equal-weight softmax means. The full official benchmark contains 2,756 test cycles/49 patients; patients 156 and 218 also occur in training. The unseen-patient view excludes their 120 test cycles, leaving 2,636 cycles/47 patients and no training/evaluation patient overlap. The existing weights and epochs are unchanged; this is a secondary subset analysis, not a new split or a newly untouched test. It does not remove full-pool SC transduction.

| Group | Full official Score +/- seed SD | Unseen-patient Score +/- seed SD | Unseen-patient Sp | Unseen-patient Se |
| --- | ---: | ---: | ---: | ---: |
| PAFA_noSC | 61.73 +/- 0.67 | 61.49 +/- 0.59 | 73.44 | 49.54 |
| PAFA_static | 62.27 +/- 0.86 | 62.03 +/- 0.85 | 75.77 | 48.29 |
| R_E1_all11 | 62.64 +/- 0.38 | 62.61 +/- 0.45 | 73.85 | 51.38 |
| R_E2_beats5 | 61.90 +/- 0.62 | 61.71 +/- 0.65 | 70.90 | 52.53 |
| R_E3_ast3 | 59.51 +/- 0.73 | 59.66 +/- 0.71 | 72.08 | 47.25 |
| R_E4_htsat3 | 60.12 +/- 1.07 | 60.35 +/- 1.07 | 71.96 | 48.74 |
| U_beats3_static | 63.14 +/- 0.56 | 62.99 +/- 0.59 | 74.68 | 51.30 |
| U_beats3_htsat2_static | 63.19 +/- 0.74 | 63.21 +/- 0.72 | 76.06 | 50.35 |

[Interpretation] The registered all-11 group leads the registered groups on the full official split, while BEATs-5 leads their LODO macro. A higher official Score does not identify the best device-shift model. Static user ensembles are separate exploratory extensions with different membership.

Reproduce these additions:

```bash
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 PYTHONPATH=. .venv/bin/python \
  outputs/evaluation/report-20261024/audit/registered_coverage.py
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 PYTHONPATH=. .venv/bin/python \
  outputs/evaluation/report-20261024/audit/official_fixed.py
```

Artifacts: `audit/registered-coverage.json` (16 fold/group bootstrap summaries, including size and coverage contrasts) and `audit/official-fixed.json` (80 group/seed/cohort rows, fixed member epochs, patient boundaries and source hashes), below `outputs/evaluation/report-20261024/`.

## Finite-N k-rate with static SC held fixed

Each seed/device has 50 random patient partitions: 30% adaptation patients and the disjoint remainder for evaluation. Sample nested cycle budgets from adaptation patients without labels. Within a repetition, all N use the same evaluation patients. Estimate the PAFA normal-prediction rate and ensemble ratio quantile on adaptation cycles only; freeze k before evaluation. Full-adaptation means all cycles in that adaptation cohort, not the whole target cohort.

All contrasts below use controls on the exact same evaluation cohort as the finite-N arm. Average repetitions within device, then devices within model seed. The +/- terms are sample SD across five model-seed summaries, not patient confidence intervals. Repetitions are Monte Carlo draws and are not independent patient samples.

| Group | N | Score +/- seed SD | dScore vs raw | dScore vs PAFA | dScore vs full adaptation | dSp vs PAFA | dSe vs PAFA | Missing / 1000 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| U_beats3_static | 1 | 53.76 +/- 1.23 | -8.03 +/- 0.86 | -6.43 +/- 1.48 | -8.06 +/- 1.43 | -29.48 +/- 3.57 | 16.63 +/- 4.01 | 0 |
| U_beats3_static | 2 | 58.00 +/- 0.82 | -3.79 +/- 0.75 | -2.19 +/- 1.18 | -3.82 +/- 1.02 | -12.21 +/- 2.18 | 7.83 +/- 2.63 | 0 |
| U_beats3_static | 5 | 60.20 +/- 0.31 | -1.60 +/- 0.96 | 0.01 +/- 0.63 | -1.62 +/- 0.41 | -2.20 +/- 1.34 | 2.22 +/- 2.05 | 0 |
| U_beats3_static | 10 | 60.67 +/- 0.36 | -1.13 +/- 0.85 | 0.48 +/- 0.57 | -1.15 +/- 0.32 | -0.22 +/- 0.94 | 1.18 +/- 1.49 | 0 |
| U_beats3_static | 20 | 61.15 +/- 0.50 | -0.64 +/- 0.95 | 0.96 +/- 0.36 | -0.67 +/- 0.15 | -0.15 +/- 0.64 | 2.08 +/- 0.93 | 0 |
| U_beats3_static | 50 | 61.58 +/- 0.47 | -0.22 +/- 0.91 | 1.39 +/- 0.40 | -0.25 +/- 0.17 | -0.20 +/- 0.42 | 2.98 +/- 0.55 | 0 |
| U_beats3_static | 100 | 61.74 +/- 0.49 | -0.05 +/- 0.98 | 1.56 +/- 0.40 | -0.07 +/- 0.07 | 0.03 +/- 0.50 | 3.08 +/- 0.57 | 22 |
| U_beats3_static | all | 61.82 +/- 0.54 | 0.03 +/- 1.04 | 1.63 +/- 0.41 | 0.00 +/- 0.00 | 0.13 +/- 0.50 | 3.14 +/- 0.70 | 0 |
| U_beats3_htsat2_static | 1 | 54.25 +/- 0.69 | -7.68 +/- 0.92 | -5.94 +/- 1.27 | -7.96 +/- 1.11 | -29.11 +/- 3.67 | 17.23 +/- 3.11 | 0 |
| U_beats3_htsat2_static | 2 | 58.08 +/- 0.60 | -3.85 +/- 0.87 | -2.11 +/- 1.11 | -4.13 +/- 0.96 | -12.60 +/- 1.84 | 8.38 +/- 1.98 | 0 |
| U_beats3_htsat2_static | 5 | 60.12 +/- 0.33 | -1.80 +/- 0.86 | -0.07 +/- 0.72 | -2.08 +/- 0.70 | -2.81 +/- 1.19 | 2.68 +/- 1.85 | 0 |
| U_beats3_htsat2_static | 10 | 60.84 +/- 0.43 | -1.09 +/- 0.70 | 0.65 +/- 0.39 | -1.37 +/- 0.31 | -0.83 +/- 0.80 | 2.13 +/- 1.06 | 0 |
| U_beats3_htsat2_static | 20 | 61.59 +/- 0.63 | -0.34 +/- 0.78 | 1.40 +/- 0.22 | -0.62 +/- 0.17 | -0.30 +/- 0.80 | 3.09 +/- 0.64 | 0 |
| U_beats3_htsat2_static | 50 | 61.99 +/- 0.66 | 0.07 +/- 0.71 | 1.80 +/- 0.33 | -0.21 +/- 0.07 | 0.16 +/- 0.58 | 3.45 +/- 0.29 | 0 |
| U_beats3_htsat2_static | 100 | 62.12 +/- 0.68 | 0.19 +/- 0.72 | 1.94 +/- 0.34 | -0.08 +/- 0.02 | 0.36 +/- 0.63 | 3.52 +/- 0.32 | 22 |
| U_beats3_htsat2_static | all | 62.21 +/- 0.68 | 0.28 +/- 0.70 | 2.02 +/- 0.34 | 0.00 +/- 0.00 | 0.61 +/- 0.69 | 3.43 +/- 0.55 | 0 |

[Fact] For N=100, 22 of 1,000 fold/repetitions per group have insufficient adaptation cycles; they are missing, not imputed. The N=100 row therefore has a different feasibility subset. Its paired control differences remain matched; absolute cross-N comparisons with that row need caution.

[Interpretation] One or two cycles produce unstable adjustment with substantial specificity losses. Fifty-cycle estimates descriptively approach the full-adaptation control, but this is not a validated minimum budget or evidence of equivalence. Predicted-normal-rate matching does not guarantee specificity preservation.

[Limitation] SC coefficients here were already estimated from the full target pool, including evaluation-patient audio without labels. Only k estimation is patient-disjoint and finite-N. These tables cannot establish performance of an entirely patient-disjoint or N-total-cycle SC+k pipeline. Joint finite-N SC remains pending. Patient/seed bootstrap intervals with adaptation-only k refitting are now in the [uncertainty update](sc-finite-n-uncertainty-2026-10-24.md); it uses a separately identified Monte Carlo reanalysis point. Adjustment distributions and a separately labelled whole-target transductive reference follow below. No k-adjusted conformal claim is made.

### Replayed adjustment diagnostics

[Fact] Replayed all 15,956 feasible finite-N rows with the original training environment (NumPy 2.2.6), reproducing Sp/Se/Score/HS and k within numerical tolerance. Reconstructed patient partitions are disjoint for every repetition; all 44 infeasible entries across two groups are explicitly listed in the diagnostic artifact. Analysis-environment NumPy 2.5.3 produces slightly different float32 quantiles, so use `.venv-train` for this replay.

Intervals below are pooled 2.5th/50th/97.5th percentiles of log(k) across devices, seeds and random partitions, not confidence intervals. Devices have different k distributions; these pooled diagnostics do not establish a universal threshold. Changed is the median fraction of evaluation predictions differing from raw k=1.

| Group | N | log(k), 2.5% / median / 97.5% | Median changed vs raw | Median changed vs full adaptation | Zero k |
| --- | --- | --- | ---: | ---: | ---: |
| U_beats3_static | 1 | -7.98 / -2.59 / 8.20 | 0.267 | 0.286 | 0 |
| U_beats3_static | 2 | -7.46 / -0.02 / 7.96 | 0.166 | 0.146 | 0 |
| U_beats3_static | 5 | -5.22 / 0.67 / 7.22 | 0.112 | 0.085 | 0 |
| U_beats3_static | 10 | -1.66 / 0.73 / 6.59 | 0.089 | 0.059 | 0 |
| U_beats3_static | 20 | -1.12 / 0.69 / 4.90 | 0.077 | 0.038 | 0 |
| U_beats3_static | 50 | -0.66 / 0.66 / 2.64 | 0.058 | 0.018 | 0 |
| U_beats3_static | 100 | -0.57 / 0.66 / 1.48 | 0.054 | 0.010 | 0 |
| U_beats3_static | all | -0.45 / 0.65 / 1.35 | 0.052 | 0.000 | 0 |
| U_beats3_htsat2_static | 1 | -7.83 / -1.36 / 7.91 | 0.273 | 0.284 | 0 |
| U_beats3_htsat2_static | 2 | -6.77 / -0.18 / 7.59 | 0.168 | 0.152 | 0 |
| U_beats3_htsat2_static | 5 | -3.09 / 0.43 / 6.90 | 0.127 | 0.095 | 0 |
| U_beats3_htsat2_static | 10 | -1.62 / 0.49 / 6.00 | 0.100 | 0.060 | 0 |
| U_beats3_htsat2_static | 20 | -1.22 / 0.42 / 3.75 | 0.077 | 0.041 | 0 |
| U_beats3_htsat2_static | 50 | -0.87 / 0.39 / 1.74 | 0.056 | 0.021 | 0 |
| U_beats3_htsat2_static | 100 | -0.63 / 0.38 / 1.57 | 0.047 | 0.012 | 0 |
| U_beats3_htsat2_static | all | -0.51 / 0.36 / 1.61 | 0.041 | 0.000 | 0 |

### Whole-target transductive reference

This reference estimates k on the same target patients it evaluates. It uses every target cycle and is not a deployable finite-batch control. Its absolute Score uses a different evaluation cohort from the patient-split finite-N rows, so do not subtract those scores to estimate an adaptation effect.

| Group | Full-target macro Score +/- seed SD |
| --- | ---: |
| U_beats3_static | 62.05 +/- 0.57 |
| U_beats3_htsat2_static | 62.36 +/- 0.82 |

Reproduce the replay and diagnostics:

```bash
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 PYTHONPATH=. .venv-train/bin/python \
  outputs/evaluation/report-20261024/audit/krate_diagnostics.py
```

Artifact: `outputs/evaluation/report-20261024/audit/krate-diagnostics.json` includes per-repetition patient cohorts, normal-rate and prediction-change distributions, the missing-entry list, transductive rows and SHA-256 provenance. Preserve the script and artifacts with the report archive.

## Artifact checks and reproduction

[Fact] Audit checked 400 finite-N input-file hashes and 880 distinct registered member-file hashes against saved manifests. Recomputed all 80 registered unit classification metrics; checked unique exported cycle IDs and PAFA/ensemble label, patient and calibration-device alignment. Calibration/evaluation patient overlap was absent. Historical member exports lack independent cycle IDs: ensemble IDs come from deterministic annotation reconstruction, so these checks cannot detect an undocumented within-patient, same-label permutation in a historical export. Exported ID arrays currently have object dtype; the local audit reads only locally generated ensemble artifacts with pickle enabled. Do not enable pickle for untrusted NPZ inputs.

```bash
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 PYTHONPATH=. .venv/bin/python \
  outputs/evaluation/report-20261024/audit/analyse.py
.venv/bin/python -m pytest -q tests/evaluation/test_ensemble_study.py
```

Generated evidence (ignored local artifacts, preserve with report archive): `outputs/evaluation/report-20261024/{registered,krate}/summary.json`, `audit/analyse.py`, `audit/analysis.json`, and `audit/audit.log`. The analysis JSON records input-summary and audit-source SHA-256 hashes. The source summaries retain all per-seed/per-fold rows. Tests: 3 passed.

## Remaining execution and research order

[Fact] Last recorded remote inspection was 24 October 10:00 local: B/D/E schedulers reported live, with 2/1/2 dynamic units partial and none certified complete; D also had one distillation unit partial. A fresh SSH attempt in this session failed before connecting (`socket: Operation not permitted`). This is an observation failure, not evidence the jobs stopped; do not restart or duplicate them from this record.

1. Revalidate the existing remote process handles and collect certified dynamic-SC/KD outputs when connectivity permits. Preserve running jobs and the API. Complete scheduled missing cells before adding another training search.
2. Finish uncertainty and missing diagnostics in the CPU analyses, then compare all no-SC/static/dynamic diagonal arms with explicit source-run mapping. Keep inference-only correction separate.
3. For the broader SOTA objective, first audit published protocols and reference implementations against patient splits, fixed-vs-test-selected epochs, seed count and external pretraining. Existing Score results do not establish SOTA. Run any next candidate selection on source training-patient CV; repeated official-test/LODO inspection makes new selections exploratory.
4. Evaluate a frozen candidate in LODO, then assess conformal coverage and set size with a freshly computed null gate. Conformal can reveal uncertainty; adding it does not by itself prove device robustness. Retain negative results and complete phantom H1/H3/co-primary H2 when hardware exists.

No new model-quality or hardware result is claimed by this update. The deadline is not evidence of completion.
