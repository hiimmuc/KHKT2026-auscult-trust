# Existing SC arms and finite-N uncertainty, 24 October 2026

Exploratory analysis of frozen predictions. Selected serving model remains BEATs + PAFA trained/evaluated with static SC. No new training, target-label fitting, ensemble member search or model promotion occurred. Read with the [earlier ensemble/finite-N report](ensemble-finite-n-2026-10-24.md) and [benchmark protocol audit](benchmark-protocol-audit-2026-10-24.md).

## Patient/seed uncertainty for finite-N k-rate

[Fact] Added 1,000 hierarchical bootstrap draws, 50 adaptation/evaluation partitions per draw, five model seeds and four fixed devices. Draw whole target patients with replacement using one global patient-weight vector shared across folds and seeds. Draw five model seeds with replacement, shared across folds. In each fold, partition distinct resampled patient identities into 30% adaptation and disjoint evaluation cohorts before selecting adaptation cycles. Keep all copies of one patient on one side. Recompute the PAFA predicted-normal rate and ensemble k using adaptation cycles only, with nested cycle budgets. Evaluate all N on the same evaluation cohort within a repetition. Use paired controls and equal device weights.

Bootstrap patient multiplicities replicate all cycles of a selected patient; finite-N sampling is without replacement within that bootstrap pool. Repeated original cycle identities can occur through patient multiplicity only. The procedure asserts patient separation in every partition. It never resamples cycles as independent patients.

The displayed reanalysis point uses shared random partitions across model seeds, rather than the original seed-specific random partitions. Both average 50 draws of the same adaptation policy; finite Monte Carlo differences remain. Original point estimates are retained in the earlier report. Do not attach these intervals to a different point estimate or count the repeated partitions as extra subjects/seeds. The intervals are pointwise, exploratory, and not adjusted across the two ensembles/eight budgets.

| Group | N | Reanalysis Score [95% interval] | dScore vs PAFA [95% interval] | dSp vs PAFA [95% interval] | dSe vs PAFA [95% interval] | Estimable bootstrap draws |
| --- | --- | --- | --- | --- | --- | ---: |
| U_beats3_static | 1 | +53.90 [+51.53, +56.78] | -6.28 [-8.60, -3.45] | -28.65 [-33.49, -23.02] | +16.08 [+10.52, +23.30] | 1000 / 1000 |
| U_beats3_static | 2 | +58.07 [+55.34, +60.79] | -2.12 [-4.15, -0.23] | -12.70 [-17.60, -8.94] | +8.46 [+4.04, +14.75] | 1000 / 1000 |
| U_beats3_static | 5 | +60.43 [+57.29, +62.96] | +0.24 [-1.31, +1.33] | -1.49 [-5.21, -0.14] | +1.98 [-0.77, +6.52] | 1000 / 1000 |
| U_beats3_static | 10 | +60.76 [+57.65, +63.64] | +0.58 [-0.57, +1.68] | +0.54 [-2.49, +0.97] | +0.62 [-0.80, +4.63] | 1000 / 1000 |
| U_beats3_static | 20 | +61.11 [+57.94, +64.16] | +0.92 [-0.13, +1.98] | +0.34 [-1.77, +0.68] | +1.51 [+0.24, +4.47] | 1000 / 1000 |
| U_beats3_static | 50 | +61.56 [+58.28, +64.56] | +1.37 [+0.24, +2.40] | +0.20 [-1.87, +0.61] | +2.54 [+1.13, +5.21] | 999 / 1000 |
| U_beats3_static | 100 | +61.71 [+58.08, +64.59] | +1.54 [+0.36, +2.55] | +0.35 [-1.95, +0.76] | +2.73 [+1.37, +5.40] | 959 / 1000 |
| U_beats3_static | all | +61.79 [+58.40, +64.94] | +1.61 [+0.35, +2.57] | +0.48 [-1.56, +0.83] | +2.73 [+1.31, +5.24] | 1000 / 1000 |
| U_beats3_htsat2_static | 1 | +54.16 [+52.00, +57.10] | -6.03 [-8.48, -2.90] | -28.70 [-33.55, -22.99] | +16.64 [+11.05, +24.23] | 1000 / 1000 |
| U_beats3_htsat2_static | 2 | +58.08 [+55.72, +60.81] | -2.11 [-4.17, +0.34] | -13.03 [-17.51, -9.30] | +8.81 [+3.98, +15.79] | 1000 / 1000 |
| U_beats3_htsat2_static | 5 | +60.50 [+57.70, +62.86] | +0.31 [-1.60, +1.97] | -2.30 [-5.78, -0.78] | +2.92 [-0.48, +8.19] | 1000 / 1000 |
| U_beats3_htsat2_static | 10 | +61.10 [+58.16, +63.78] | +0.91 [-0.72, +2.41] | -0.50 [-3.01, +0.44] | +2.32 [-0.14, +6.49] | 1000 / 1000 |
| U_beats3_htsat2_static | 20 | +61.59 [+58.49, +64.48] | +1.40 [-0.15, +2.85] | -0.23 [-2.19, +0.55] | +3.04 [+0.67, +6.38] | 1000 / 1000 |
| U_beats3_htsat2_static | 50 | +61.96 [+58.71, +64.93] | +1.78 [+0.17, +3.14] | +0.28 [-1.88, +0.93] | +3.28 [+1.06, +6.57] | 999 / 1000 |
| U_beats3_htsat2_static | 100 | +62.09 [+58.68, +64.80] | +1.92 [+0.34, +3.31] | +0.52 [-1.72, +1.20] | +3.32 [+1.04, +6.94] | 959 / 1000 |
| U_beats3_htsat2_static | all | +62.13 [+58.76, +65.10] | +1.95 [+0.22, +3.22] | +0.69 [-1.30, +1.37] | +3.20 [+0.81, +6.36] | 1000 / 1000 |

[Fact] 30 of 200,000 bootstrap fold/partition combinations could not support both normal and abnormal evaluation groups (or a usable patient partition). Undefined cases are excluded within the partition average, not assigned zero Score. A bootstrap draw is non-estimable for an N if any device has no feasible partition at that N; that device is never dropped from the macro. N=50 has 999/1,000 estimable draws, N=100 has 959/1,000, and other budgets have 1,000. Those intervals are conditional on estimability. Original finite-N infeasible cases remain recorded separately.

[Interpretation] N=1 causes substantial specificity and Score loss. N=50 gives an exploratory positive Score contrast against PAFA for both groups, with sensitivity gains, but specificity preservation is not established. Its lower specificity limits permit losses of about 1.9 points; no prospective non-inferiority margin was registered. Larger N does not justify an equivalence claim from overlapping intervals. These findings support further validation of larger-batch adaptation, not a deployed minimum-N rule.

[Limitations] SC and model predictions remain fixed at their existing full-target-pool estimates. This resamples target patients and model seeds and refits k, but does not refit the classifier/SC, quantify training-patient uncertainty, validate new devices or establish joint N-total-cycle SC+k performance. No k-adjusted conformal result is claimed.

## LODO comparison of existing SC arms

[Fact] Verified 880 available group/seed/fold rows from 620 underlying units, with 2,481 source/input file hashes recorded. Groups are the three user ensembles and their seven individual members. Every group is equal-weight mean softmax, paired by seed and held-out device. No k adjustment is used. Source/calibration/test metadata alignment and fixed epochs were checked; inference metadata identifies target-only correction of weights trained without SC. Remote checkpoint bytes were not independently reread here.

No-SC, static-trained/static-inference and dynamic-trained/dynamic-inference are diagonal arms. Inference-only static/dynamic applies correction only to target cycles of no-SC weights; their calibration predictions stay on the source regime. These have a different intervention and are shown separately. The diagonal comparisons do not isolate training from inference effects. Cross-mode combinations are not run.

Trained dynamic exists only for BEATs here. Four AST/HTS-AT cells x four LODO folds x five seeds = 80 missing member units, plus 20 official-split units, make the scheduled 100-unit completion. The 120 missing rows below include their two ensemble groups; they are not 120 distinct training jobs. No incomplete member ensemble is averaged.

| Group | Arm | Macro Score +/- seed SD | dScore vs same-group no SC [patient/seed 95% interval] | Coverage | Mean abs(Delta) | Set size |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| U_beats3 | no_SC | 60.52 +/- 1.52 | -0.00 [+0.00, +0.00] | 0.925 | 0.042 | 2.430 |
| U_beats3 | trained_static | 61.92 +/- 1.15 | +1.40 [-0.24, +3.03] | 0.912 | 0.049 | 2.440 |
| U_beats3 | trained_dynamic | 60.30 +/- 1.29 | -0.22 [-2.45, +1.82] | 0.901 | 0.046 | 2.428 |
| U_beats3 | inference_only_static | 60.88 +/- 1.24 | +0.36 [-1.22, +1.91] | 0.918 | 0.052 | 2.551 |
| U_beats3 | inference_only_dynamic | 60.13 +/- 1.39 | -0.39 [-1.76, +1.04] | 0.906 | 0.057 | 2.493 |
| U_ast2 | no_SC | 58.08 +/- 0.71 | +0.00 [+0.00, +0.00] | 0.904 | 0.046 | 2.483 |
| U_ast2 | trained_static | 58.71 +/- 0.91 | +0.63 [-1.00, +2.20] | 0.876 | 0.051 | 2.256 |
| U_ast2 | trained_dynamic | Pending | Pending | Pending | Pending | Pending |
| U_ast2 | inference_only_static | 58.86 +/- 0.59 | +0.78 [-0.80, +2.26] | 0.897 | 0.045 | 2.489 |
| U_ast2 | inference_only_dynamic | 58.42 +/- 0.76 | +0.33 [-1.30, +2.02] | 0.900 | 0.044 | 2.534 |
| U_htsat2 | no_SC | 57.19 +/- 1.81 | +0.00 [+0.00, +0.00] | 0.884 | 0.036 | 2.264 |
| U_htsat2 | trained_static | 58.68 +/- 0.87 | +1.49 [-0.21, +3.20] | 0.882 | 0.040 | 2.139 |
| U_htsat2 | trained_dynamic | Pending | Pending | Pending | Pending | Pending |
| U_htsat2 | inference_only_static | 58.22 +/- 0.69 | +1.03 [-0.39, +2.75] | 0.882 | 0.040 | 2.208 |
| U_htsat2 | inference_only_dynamic | 58.23 +/- 1.26 | +1.05 [-0.37, +2.56] | 0.879 | 0.040 | 2.217 |
| beats+ft+pafa | no_SC | 58.79 +/- 1.39 | +0.00 [+0.00, +0.00] | 0.887 | 0.057 | 2.229 |
| beats+ft+pafa | trained_static | 60.30 +/- 0.84 | +1.51 [-0.39, +3.31] | 0.864 | 0.069 | 2.132 |
| beats+ft+pafa | trained_dynamic | 57.87 +/- 1.58 | -0.92 [-3.15, +0.95] | 0.886 | 0.038 | 2.333 |
| beats+ft+pafa | inference_only_static | 58.52 +/- 1.19 | -0.27 [-1.52, +1.03] | 0.875 | 0.047 | 2.304 |
| beats+ft+pafa | inference_only_dynamic | 57.97 +/- 1.52 | -0.82 [-2.13, +0.61] | 0.882 | 0.040 | 2.339 |
| beats+ft+sg_scl | no_SC | 59.02 +/- 1.65 | +0.00 [+0.00, +0.00] | 0.913 | 0.046 | 2.569 |
| beats+ft+sg_scl | trained_static | 60.53 +/- 1.50 | +1.51 [-0.17, +3.34] | 0.892 | 0.043 | 2.498 |
| beats+ft+sg_scl | trained_dynamic | 59.47 +/- 0.81 | +0.45 [-1.75, +2.68] | 0.874 | 0.041 | 2.474 |
| beats+ft+sg_scl | inference_only_static | 58.84 +/- 0.73 | -0.18 [-1.70, +1.48] | 0.893 | 0.051 | 2.549 |
| beats+ft+sg_scl | inference_only_dynamic | 58.78 +/- 1.16 | -0.24 [-1.50, +1.17] | 0.886 | 0.044 | 2.517 |
| beats+ft+patchmix_cl | no_SC | 60.12 +/- 1.75 | -0.00 [+0.00, +0.00] | 0.933 | 0.050 | 2.769 |
| beats+ft+patchmix_cl | trained_static | 60.07 +/- 1.08 | -0.04 [-1.95, +1.77] | 0.915 | 0.048 | 2.766 |
| beats+ft+patchmix_cl | trained_dynamic | 59.88 +/- 1.60 | -0.23 [-2.63, +1.96] | 0.902 | 0.052 | 2.663 |
| beats+ft+patchmix_cl | inference_only_static | 60.20 +/- 1.28 | +0.08 [-1.43, +1.55] | 0.921 | 0.049 | 2.809 |
| beats+ft+patchmix_cl | inference_only_dynamic | 59.41 +/- 1.64 | -0.70 [-2.27, +0.70] | 0.905 | 0.054 | 2.742 |
| ast+ft+patchmix_cl | no_SC | 56.35 +/- 1.23 | +0.00 [+0.00, +0.00] | 0.919 | 0.044 | 2.713 |
| ast+ft+patchmix_cl | trained_static | 56.70 +/- 0.87 | +0.35 [-1.46, +2.19] | 0.885 | 0.041 | 2.425 |
| ast+ft+patchmix_cl | trained_dynamic | Pending | Pending | Pending | Pending | Pending |
| ast+ft+patchmix_cl | inference_only_static | 57.18 +/- 1.38 | +0.83 [-0.82, +2.44] | 0.907 | 0.039 | 2.665 |
| ast+ft+patchmix_cl | inference_only_dynamic | 57.47 +/- 1.03 | +1.12 [-0.64, +2.89] | 0.906 | 0.041 | 2.689 |
| ast+ft+pafa | no_SC | 57.05 +/- 1.04 | +0.00 [+0.00, +0.00] | 0.889 | 0.049 | 2.394 |
| ast+ft+pafa | trained_static | 58.46 +/- 1.12 | +1.42 [-0.36, +3.06] | 0.862 | 0.055 | 2.220 |
| ast+ft+pafa | trained_dynamic | Pending | Pending | Pending | Pending | Pending |
| ast+ft+pafa | inference_only_static | 57.71 +/- 1.11 | +0.67 [-1.03, +2.38] | 0.879 | 0.051 | 2.486 |
| ast+ft+pafa | inference_only_dynamic | 57.19 +/- 0.78 | +0.14 [-1.61, +2.04] | 0.889 | 0.046 | 2.581 |
| htsat+ft+patchmix_cl | no_SC | 56.88 +/- 2.00 | -0.00 [+0.00, +0.00] | 0.885 | 0.043 | 2.270 |
| htsat+ft+patchmix_cl | trained_static | 58.13 +/- 0.92 | +1.25 [-0.68, +3.04] | 0.882 | 0.036 | 2.146 |
| htsat+ft+patchmix_cl | trained_dynamic | Pending | Pending | Pending | Pending | Pending |
| htsat+ft+patchmix_cl | inference_only_static | 57.94 +/- 0.92 | +1.05 [-0.49, +2.92] | 0.883 | 0.041 | 2.211 |
| htsat+ft+patchmix_cl | inference_only_dynamic | 57.69 +/- 1.15 | +0.81 [-0.57, +2.31] | 0.879 | 0.042 | 2.219 |
| htsat+ft+sg_scl | no_SC | 56.60 +/- 1.51 | +0.00 [+0.00, +0.00] | 0.879 | 0.037 | 2.405 |
| htsat+ft+sg_scl | trained_static | 58.10 +/- 1.20 | +1.50 [-0.54, +3.35] | 0.871 | 0.050 | 2.293 |
| htsat+ft+sg_scl | trained_dynamic | Pending | Pending | Pending | Pending | Pending |
| htsat+ft+sg_scl | inference_only_static | 57.59 +/- 1.20 | +0.98 [-0.24, +2.29] | 0.880 | 0.038 | 2.353 |
| htsat+ft+sg_scl | inference_only_dynamic | 57.34 +/- 1.51 | +0.74 [-0.56, +2.03] | 0.877 | 0.038 | 2.353 |

Score intervals: 1,000 paired patient/seed draws, using a shared global target-patient cluster draw and a shared model-seed draw across four fixed devices. These are pointwise exploratory intervals, not a multiplicity-adjusted superiority analysis. Coverage/set-size entries here are descriptive means; their paired effects must not inherit the classification interval. All per-seed/per-fold Sp, Se, Score, HS, coverage, Delta, singleton and empty rates are in `sc-existing.json`.

[Interpretation] Static training gives the three-BEATs group a descriptive +1.40 Score over no SC, whereas dynamic training gives -0.22; both contrasts include zero after patient/seed uncertainty. The selected single PAFA/static gains 1.51 Score over PAFA/no-SC but its coverage falls from 0.887 to 0.864. That is not evidence that SC improves coverage. Do not replace the selected model based on these post hoc comparisons.

## Existing source-device-robust threshold comparator

This comparator uses the already implemented `robust_coverage`: maximum LAC quantile over eligible source devices, requiring at least 30 calibration cycles per device. It uses source calibration labels only. It changes prediction sets, not classifier argmax predictions, model weights or k. No target labels select the threshold.

| Group | Arm | Ordinary macro coverage | Robust macro coverage | Ordinary set size | Robust set size |
| --- | --- | ---: | ---: | ---: | ---: |
| PAFA | no_SC | 0.887 | 0.930 | 2.229 | 2.639 |
| PAFA | trained_static | 0.864 | 0.915 | 2.132 | 2.484 |
| U_beats3 | no_SC | 0.925 | 0.958 | 2.430 | 2.782 |
| U_beats3 | trained_static | 0.912 | 0.945 | 2.440 | 2.763 |
| U_ast2 | no_SC | 0.904 | 0.948 | 2.483 | 2.953 |
| U_ast2 | trained_static | 0.876 | 0.907 | 2.256 | 2.575 |
| U_htsat2 | no_SC | 0.884 | 0.913 | 2.264 | 2.560 |
| U_htsat2 | trained_static | 0.882 | 0.916 | 2.139 | 2.459 |

### Selected PAFA/static by held-out device

| Device | Ordinary coverage | Robust coverage | Ordinary set size | Robust set size |
| --- | ---: | ---: | ---: | ---: |
| Meditron | 0.902 | 0.940 | 1.771 | 2.178 |
| LittC2SE | 0.880 | 0.906 | 1.859 | 2.145 |
| Litt3200 | 0.975 | 0.983 | 3.007 | 3.222 |
| AKGC417L | 0.700 | 0.830 | 1.890 | 2.390 |

[Interpretation] The robust threshold lifts PAFA/static macro coverage to 0.915, with set size 2.484 instead of 2.132. AKGC417L still has only 0.830 coverage, below the 0.90 nominal level. A macro above nominal does not establish coverage on every unseen device. This source-only comparator offers no distribution-free guarantee on arbitrary unseen devices, and larger sets are a cost. It is not evidence for SC-only H2 because the conformal rule also changes. All 160 unit/group threshold comparisons are retained, including eligible calibration-device counts.

## Reproduction and remaining work

```bash
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 PYTHONPATH=. .venv-train/bin/python \
  outputs/evaluation/report-20261024/audit/krate_bootstrap.py \
  --out outputs/evaluation/report-20261024/audit/krate-bootstrap-replay
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 PYTHONPATH=. .venv-train/bin/python \
  outputs/evaluation/report-20261024/audit/sc_existing.py
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 PYTHONPATH=. .venv-train/bin/python \
  outputs/evaluation/report-20261024/audit/conformal_existing.py
```

Artifacts under `outputs/evaluation/report-20261024/audit/`: `krate-bootstrap/{summary.json,draws.npz,progress.json}`, `sc-existing.json`, `conformal-existing.json`, the three analysis scripts and their logs. Preserve these ignored artifacts with the report archive. The finite-N script refuses an existing output directory and verifies original prediction hashes before and after calculation. Its assert-based self-check tests vector/scalar quantiles and metric parity and 100 patient-bootstrap partitions. RNG seed is 20261026; NumPy 2.2.6; 1,000 draws and 50 partitions/draw. No GPU is used.

Joint finite-N SC, missing dynamic training completion, KD quality/runtime/VRAM, source-patient-CV optimization and hardware experiments remain pending. Current-session SSH attempts fail before connecting (`socket: Operation not permitted`), and local GPU-driver communication is unavailable. Historical scheduler records are not evidence of current job state. Do not launch duplicate jobs or overwrite historical results. SOTA and device-robust coverage remain unproven.
