# First synced dynamic-SC and distillation results, 24 October 2026

[Preliminary] Analysed completed prediction artifacts already synced to machine A. The 11:00 local tracker snapshot records 5/100 dynamic-SC units and 1/6 KD pilot units complete; subsequent jobs remain partial/pending. This is not an all-campaign result. No job was restarted or new model selected.

## Local integrity and scope

[Fact] All five locally available completion-bound files were SHA-256 verified for each of six completed units: train arguments, report, prediction NPZ, input manifest and SC state. Verified 1,850 locally available original input files from the training manifests. Another 294 original manifest paths are unavailable locally, principally remote frozen snapshot/teacher-target inputs; these were recorded as unavailable rather than silently remapped. Checkpoint files named by all six completion markers are absent from their corresponding paths on A. Thus prediction-based analysis is available, but full checkpoint/input certification, reload, teacher-target replay and student GPU cost measurements are pending.

Predictions are finite/non-negative and sum to one within stored float16 precision. Report Sp/Se/Score/HS were recomputed from the stored hard predictions, using the stored fixed epoch. Labels, patients and device order were matched across arms and against annotations. No stored-hard-prediction versus rounded-probability argmax disagreements were found in the completed dynamic/KD units. Raw data and historical outputs were not changed.

## AST Patch-Mix CL: complete official-split dynamic cell

Five seeds 0-4, fixed epoch 50. These are the official-split units; the four LODO folds and other dynamic cells are not yet available from this campaign. The dynamic/static active recipe settings match except SC mode. Identifiers/config-file lists, loader workers (2 versus 4), and previously absent inactive PC-MCL defaults differ; the PC-MCL options are not consumed by Patch-Mix CL. This check does not replay identical initialization/numerical trajectories.

| Evaluation cohort | Arm | Sp | Se | Score +/- seed SD |
| --- | --- | ---: | ---: | ---: |
| official_all | none | 71.72 | 44.71 | 58.21 +/- 1.27 |
| official_all | static | 70.75 | 45.83 | 58.29 +/- 1.05 |
| official_all | dynamic | 73.45 | 45.74 | 59.60 +/- 0.50 |
| unseen_patients | none | 71.79 | 44.29 | 58.04 +/- 1.25 |
| unseen_patients | static | 71.01 | 45.84 | 58.42 +/- 1.00 |
| unseen_patients | dynamic | 73.76 | 45.28 | 59.52 +/- 0.73 |

| Cohort | Contrast | Mean dScore | Paired patient/seed 95% interval |
| --- | --- | ---: | --- |
| official_all | dynamic_minus_none | +1.39 | [-1.03, +3.91] |
| official_all | dynamic_minus_static | +1.31 | [-0.97, +3.30] |
| unseen_patients | dynamic_minus_none | +1.47 | [-1.14, +4.21] |
| unseen_patients | dynamic_minus_static | +1.09 | [-1.15, +3.19] |

Intervals use 1,000 paired draws of whole test patients and five model seeds, shared across arms. They are pointwise exploratory intervals. Both contrasts include zero; the descriptive gain is not established superiority or evidence for LODO robustness.

## Three-teacher KD pilot: first seed only

Teacher is the raw mean softmax of the same-seed static-SC BEATs PAFA, SG-SCL and Patch-Mix CL members. PAFA uses epoch 100, the other two epoch 50. Student is BEATs + PAFA/static-SC with the frozen pilot soft-target loss at epoch 100. No k-rate is distilled or used here.

[Fact] Teacher argument/report/SC-state hashes match the prior teacher-input manifest for this seed. The student and hard-label PAFA control saved arguments differ only in cell identifier, soft-target path and loader-worker count (2 versus 4). The teacher-target NPZ/provenance and student checkpoint are not locally present, so full training-target alignment and exact initialization parity were not reverified from those bytes.

| Cohort | Model | Sp | Se | Score |
| --- | --- | ---: | ---: | ---: |
| official_all | student | 78.53 | 48.68 | 63.61 |
| official_all | PAFA_static_control | 79.42 | 47.58 | 63.50 |
| official_all | teacher_U_beats3_static | 74.73 | 51.15 | 62.94 |
| unseen_patients | student | 78.78 | 47.68 | 63.23 |
| unseen_patients | PAFA_static_control | 79.55 | 46.47 | 63.01 |
| unseen_patients | teacher_U_beats3_static | 74.81 | 50.74 | 62.78 |

| Cohort | Student contrast | dScore | Paired patient 95% interval |
| --- | --- | ---: | --- |
| official_all | student minus PAFA_static_control | +0.11 | [-0.78, +1.18] |
| official_all | student minus teacher_U_beats3_static | +0.67 | [-2.51, +4.05] |
| unseen_patients | student minus PAFA_static_control | +0.22 | [-0.65, +1.32] |
| unseen_patients | student minus teacher_U_beats3_static | +0.45 | [-2.70, +3.76] |

These intervals resample patients for a single fixed model seed; they contain no model-seed uncertainty. One completed seed cannot establish a three-seed pilot outcome, five-seed trained-arm result, superiority, or equivalence with the teacher. The full official benchmark includes shared training/test patients 156 and 218; the separate unseen-patient view excludes their 120 test cycles, leaving 2,636 cycles/47 patients absent from training. Existing SC remains full-pool/transductive.

### Preliminary official-test conformal view

E0: exclude patients 156/218, then use the same 20 device-stratified calibration/evaluation patient partitions for student, control and teacher. Assert no patient overlap in every partition. Recalibrate LAC at alpha=0.1 for each model and partition. Report means over partitions descriptively; partitions reuse patients and are not 20 independent experiments. This is official-distribution calibration, not LODO or evidence for H2.

| Model | Coverage | Set size | Singleton | Empty |
| --- | ---: | ---: | ---: | ---: |
| student | 0.892 | 2.231 | 0.285 | 0.000 |
| PAFA_static_control | 0.892 | 2.103 | 0.324 | 0.000 |
| teacher_U_beats3_static | 0.891 | 1.975 | 0.348 | 0.000 |

[Interpretation] The student scores slightly higher in seed 0, but patient intervals include zero and its sets are larger than the control/teacher at similar observed coverage. It is not yet evidence for a quality or conformal-efficiency improvement. Student latency, VRAM, memory, size and reload checks still require the checkpoint and a usable execution device; model count alone is not a measured speedup.

## Reproduction and next admission gate

```bash
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 PYTHONPATH=. .venv-train/bin/python \
  outputs/evaluation/report-20261024/synced-progress/analyse.py
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 PYTHONPATH=. .venv-train/bin/python \
  outputs/evaluation/report-20261024/synced-progress/dynamic_intervals.py
```

Archive the scripts, `summary.json`, `dynamic-intervals.json` and logs below `outputs/evaluation/report-20261024/synced-progress/`. They record completion/file checks, unavailable paths, arguments, per-seed/cohort metrics, all 60 E0 model/partition rows and SHA-256 input/source hashes. Preserve this snapshot before rerunning on additional synced units.

Next: admit newly completed synced units after the same checks; keep the fixed six-unit KD pilot and 100-unit dynamic campaign identities. Compare LODO only when its calibration/test predictions exist. Obtain checkpoint and teacher-target artifacts for full certification/cost measurements. Do not retrain or duplicate jobs because direct SSH observation fails; the existing tracker is continuing to sync completed results to A. The selected model remains unchanged.
