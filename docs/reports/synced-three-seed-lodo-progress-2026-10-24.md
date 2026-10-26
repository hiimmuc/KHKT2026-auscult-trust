# Three-seed official KD and completed dynamic LODO folds

[Preliminary] Snapshot of results synced to A around 14:30 local on 24 October 2026. The tracker at 14:29 records 23/100 dynamic-SC units and 3/6 KD units complete. The official three-seed KD pilot is available; its Litt3200 half and the complete dynamic matrix are not yet available. Earlier seed-0-only results remain in the [first synced report](synced-campaign-progress-2026-10-24.md).

## Integrity and protocol

Verified prediction/completion-bound files for 26 completed local units and 2117 currently available original input files. Unavailable checkpoint or original-input paths are retained in the audit; no missing bytes are treated as certified. Recipe comparisons allow identifiers, loader-worker counts and inactive PC-MCL defaults where those fields were absent in older runs. The selected CE+PAFA and Patch-Mix CL methods do not consume those PC-MCL options. Exact initialization/numerical trajectories were not replayed.

Full official evaluation retains 2,756 test cycles/49 patients for comparability. The unseen-patient view excludes patients 156 and 218 (120 test cycles), leaving 2,636 cycles/47 patients absent from training. This is a separately labelled subset of the existing benchmark, not a newly untouched test. SC remains full-pool/transductive.

## Official KD pilot: seeds 0-2

| Cohort | Model | Sp | Se | Score +/- seed SD |
| --- | --- | ---: | ---: | ---: |
| official_all | student | 75.83 | 49.67 | 62.75 +/- 0.76 |
| official_all | control | 74.90 | 50.10 | 62.50 +/- 1.09 |
| official_all | teacher | 75.13 | 51.80 | 63.47 +/- 0.47 |
| unseen_patients | student | 76.00 | 48.85 | 62.43 +/- 0.79 |
| unseen_patients | control | 74.98 | 49.44 | 62.21 +/- 1.13 |
| unseen_patients | teacher | 75.21 | 51.46 | 63.33 +/- 0.48 |

| Cohort | Contrast | dScore | Paired patient/seed 95% interval |
| --- | --- | ---: | --- |
| official_all | student_minus_control | +0.25 | [-0.67, +1.27] |
| official_all | student_minus_teacher | -0.71 | [-3.15, +2.17] |
| unseen_patients | student_minus_control | +0.22 | [-0.70, +1.22] |
| unseen_patients | student_minus_teacher | -0.91 | [-3.55, +2.13] |

Intervals use 1,000 paired whole-patient/model-seed draws with the same resampled patients/seeds across models. They are pointwise exploratory intervals based on three seeds, not proof of superiority or equivalence. The student has a small descriptive gain over the matched control and a lower mean than its three-member teacher. Its official mean gap of -0.71 meets the proposed -1-point engineering target only as a point estimate; the interval extends well below -1, so the target is not statistically established. Litt3200 and runtime/VRAM checks remain required for the pilot.

### Official-distribution conformal view

E0 uses the same 20 clean-patient calibration/evaluation partitions for each model within each seed, recalibrating LAC in each partition. The following are descriptive means over the 60 seed/partition pairs, which reuse patients and are not independent experiments. This is not LODO or H2.

| Model | Coverage | Set size | Singleton | Empty |
| --- | ---: | ---: | ---: | ---: |
| student | 0.889 | 2.373 | 0.267 | 0.000 |
| PAFA_static_control | 0.890 | 2.285 | 0.309 | 0.000 |
| teacher_U_beats3_static | 0.888 | 2.028 | 0.329 | 0.000 |

## Completed five-seed dynamic LODO folds

Compare the same model/recipe and seed in no-SC, static-trained/static-inference and dynamic-trained/dynamic-inference arms. Only the two folds with all five new dynamic seeds are summarized below; partial folds remain per-seed rows in the artifact. No macro over an incomplete device set is reported.

| Cell / held-out device | Arm | Sp | Se | Score +/- seed SD | Coverage | Set size |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ast+ft+patchmix_cl+lodo_litt3200 | dynamic | 81.21 | 36.56 | 58.88 +/- 2.55 | 0.976 | 2.758 |
| ast+ft+patchmix_cl+lodo_litt3200 | static | 68.42 | 42.65 | 55.54 +/- 3.31 | 0.969 | 2.760 |
| ast+ft+patchmix_cl+lodo_litt3200 | none | 53.58 | 52.05 | 52.81 +/- 4.68 | 0.953 | 2.970 |
| ast+ft+patchmix_cl+lodo_littc2se | dynamic | 87.26 | 38.22 | 62.74 +/- 2.72 | 0.961 | 2.732 |
| ast+ft+patchmix_cl+lodo_littc2se | static | 96.25 | 34.98 | 65.62 +/- 3.31 | 0.941 | 2.334 |
| ast+ft+patchmix_cl+lodo_littc2se | none | 94.70 | 31.26 | 62.98 +/- 2.80 | 0.926 | 2.489 |

| Cell / held-out device | Contrast | dScore | Paired patient/seed 95% interval |
| --- | --- | ---: | --- |
| ast+ft+patchmix_cl+lodo_litt3200 | dynamic_minus_none | +6.07 | [-1.69, +12.84] |
| ast+ft+patchmix_cl+lodo_litt3200 | dynamic_minus_static | +3.35 | [-2.73, +8.60] |
| ast+ft+patchmix_cl+lodo_littc2se | dynamic_minus_none | -0.24 | [-4.21, +3.94] |
| ast+ft+patchmix_cl+lodo_littc2se | dynamic_minus_static | -2.88 | [-6.09, +0.62] |

[Interpretation] AST Patch-Mix CL dynamic SC descriptively improves Litt3200 over static SC while worsening LittC2SE. Intervals include zero and the complete macro is pending. These observations do not justify selecting SC mode per held-out device or altering the selected serving model. Retain the opposite-direction result alongside the improvement.

## Reproduction and current schedule

```bash
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 PYTHONPATH=. .venv-train/bin/python \
  outputs/evaluation/report-20261024/synced-progress-1430/analyse.py
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 PYTHONPATH=. .venv-train/bin/python \
  outputs/evaluation/report-20261024/synced-progress-1430/comparisons.py
```

Archive `summary.json`, `comparisons.json`, the two scripts and logs in that directory. They contain all unit and per-seed rows, E0 partitions, source/input hashes, unavailable-file lists and recipe differences. Preserve this snapshot before rerunning against later synced results.

The current `logs/report-20261024/resource-reallocation.json` records A joining the dynamic queue while protecting the API, revised B/A job ownership and an admission deadline of 26 October 08:00 local. This supersedes the earlier proposed 24 October deadline for those active queues; it does not guarantee completion by report submission. Current tracker manifests and allocation records take precedence over historical machine tables. The proposed [warm-up CV comparison](../updates/pafa-warmup-cv-preflight-2026-10-24.md) passed preflight but has not been launched. Finish/validate the active campaigns first.
