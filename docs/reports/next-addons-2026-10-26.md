# Bounded fusion and target spectrum adaptation

[Fact] gain6cv66 checkpoints, seeds 0-4; source-selected static SC. One seed-20261026 patient split: 33 adaptation / 78 evaluation patients; all three KAUH filters share the same patient split. Adaptation uses unlabelled audio only. Baseline uses the identical trained checkpoint and evaluation audio without target-derived SC. Adapted uses a per-filter spectrum estimated only on adaptation patients.

| Filter | Raw Score mean ± SD | SC Score mean ± SD | Delta | Paired 95% CI |
|---|---:|---:|---:|---:|
| Bell | 61.94 ± 2.89 | 65.32 ± 2.29 | +3.38 | -4.58, 11.38 |
| Diaphragm | 69.77 ± 1.50 | 69.03 ± 3.20 | -0.74 | -6.99, 4.59 |
| Extended | 71.76 ± 2.10 | 71.20 ± 2.43 | -0.56 | -6.32, 4.15 |

[Interpretation] Mixed result; every confidence interval crosses zero. No general gain established and no deployment changed. Classification only, recording binary abnormal-sound screening at threshold 0.5; mean window abnormal probability. No target-label calibration or conformal guarantee is assessed. Paired patient/model-seed bootstrap (1000 draws) conditions on the frozen adaptation cohort. Software filters are not separate physical devices; target dataset was previously observed.

[Fact] Learned fusion not fitted: original five complementary model OOF predictions are missing in the scanned roots. Existing gain6 anchor@66 and LLRD@21 OOF exports align over 4142 cycles / 79 source patients in three patient-disjoint folds, but epochs were selected using these OOF outcomes and an independent fusion-validation bundle is absent.

[Pending] Clean improved distillation needs outerfold teacher weights frozen before outer-validation outcomes. Retained best.pth files are per-fold validation-selected; exact66/21 weights were not retained. Existing OOF probabilities are outer-validation targets, not outer-training targets. Use explicitly in-sample hard+soft outer-training targets from new fold-only teachers, with temperature fixed a priori or chosen on inner source validation. Do not train outer-CV students using full-source teachers.

[Fact] Focused tests: 5 passed on B. freeze.json binds the initial source and input hashes; inference-source.py retains that exact source. execution-receipt.json records the report-only bug fix and final source hashes. All 10 prediction exports were reused unchanged for summary assembly.
