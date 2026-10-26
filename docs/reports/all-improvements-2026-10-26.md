# Nine-improvement programme: completed results

[Fact] Completed on 26 October 2026 using A (RTX 5000 Ada), B/D (RTX 5090), and F (RTX PRO 6000, project VRAM cap64GB). E was not used. No deployed weights were changed.

## Execution and timing

- 30 initial source-CV fits: eight planned new candidates, a source-motivated recording/PCSL refinement, and matched anchors; three folds, seed0.
- 6 confirmation fits: LLRD and matched anchor, three folds, seed1.
- 25 final fits: LLRD, five seeds on official and four LODO cohorts.
- All three frozen-source preprocessing audits completed. The source audit service initially lacked PYTHONPATH and repeatedly failed; this delayed reporting after GPU training finished. The import environment was fixed, then an explicit FP32 CUDA audit path passed CPU/GPU parity and completed each fold in34–39seconds.
- Approximate wall times: source screening3h32m, confirmation37m, final evaluation48m; total GPU programme about5h. Source/audit and final receipts were verified.

## Source selection

LLRD passed the two-seed source gate: +0.553 Score points, positive fold means in3/3 folds; source-selected final epoch21. Patient cap, classification top-k/attention, class-aware alignment, bandwidth/noise, consistency, and both recording variants did not pass their gates. No stack was formed because only one candidate survived. Original recording results were retained; disabling PCSL was a separately named source-only diagnostic.

Alignment attention and gain6 were already implemented and screened in the preceding programme; their prior evidence was reused rather than rerun. All trained-arm source epoch choices used patient-disjoint source validation.

## Frozen final comparison

| Endpoint | Static control, epoch21 | Mean+gain6, epoch67 | LLRD, epoch21 |
| --- | ---: | ---: | ---: |
| official | 61.044 ± 2.001 | 62.416 ± 0.633 | 62.313 ± 1.344 |
| AKGC417L | 55.012 ± 1.112 | 55.293 ± 1.735 | 54.648 ± 1.004 |
| Litt3200 | 60.775 ± 6.898 | 61.697 ± 7.648 | 59.683 ± 6.275 |
| LittC2SE | 67.086 ± 2.600 | 67.385 ± 2.002 | 67.321 ± 1.176 |
| Meditron | 58.843 ± 4.162 | 59.999 ± 2.998 | 60.789 ± 2.635 |

LODO macro means (equal device weights): control 60.429, gain6 61.093, LLRD 60.610. This descriptive macro has no newly computed macro interval.

LLRD versus gain6: official delta-0.102points (paired seed/patient95% interval−2.799 to+2.195), LODO macro delta−0.483. Device-specific contrasts all have intervals including zero. This is not evidence that LLRD improves on gain6. The reused historical controls retain their previously frozen epochs21/67; the fresh source anchor chose epoch66 in confirmation and was not refit as an additional final arm. These are exploratory comparisons against the prespecified historical benchmarks, not a newly matched epoch66 control experiment.

KAUH mean-window screening with LLRD raises referral sensitivity in these results but reduces specificity and increases uncertainty relative to gain6. No uniform KAUH improvement is established. No positive LODO/ICBHI superiority or physical-device robustness claim follows.

## Preprocessing audit

| Source recording variant | Mean binary Score over3folds |
| --- | ---: |
| registered8 | 71.769 |
| duration5 | 68.751 |
| scipy8 | 71.276 |
| no_recording_fade8 | 71.292 |

The registered8-second rule is retained. These diagnostics completed after the frozen target evaluations; no target inference rule or model was changed using their results. CPU/CUDA FP32 probability parity was checked on source audio (max difference<4.1e-9 in the pilot); this does not imply bit-identical results for every input.

## Artifacts and limits

- `outputs/evaluation/improvements-20261026/{summary.json,report.md,complete.json}`; complete source gates and paired intervals are in summary.json.
- `outputs/screening/improvements-20261026/` and `outputs/kauh/improvements-20261026/`: separate comparisons against control/gain6.
- `logs/improvements-20261026/final-verification.json`: source/final completion and receipt verification.
- Immutable training source and canonical worker outputs: `build/campaigns/improvements-20261026/`; root training/checkpoint links expose collected units without duplicating large weights.
- F replicas were reclaimed only after verifying canonical A and B copies. Active resume checkpoints, raw data and historical runs were preserved.
- Failed source candidates were not promoted to five-seed target evaluation. Target outcomes did not select survivors; new LODO results use a shared source-selected recipe rather than fully nested device-wise recipe selection.
- No distillation, external pretraining or target adaptation was added in this nine-idea programme.
