# 48-hour improvement sprint

Supersedes the large programme for execution. Allocation: four dedicated RTX 5090 GPUs for 48 h (192 GPU-hours), plus one RTX 6000 PRO for 24 h. No new jobs are launched by this plan.

## Keep two improvements

| Condition | Recipe | Purpose |
| --- | --- | --- |
| Control | Current BEATs + PAFA + static SC | Matched reference |
| A | Token-attention PAFA + static SC | Restore released alignment pooling; implemented and mechanically tested |
| B | A + smooth random spectral gain, existing 6 dB setting | Test device-response robustness using existing augmentation; no extra backbone pass |

Keep input length, optimizer, batch 32 and loss weights fixed. No warm-up, new class-aware loss, pretraining, new recording head, learned fusion or extra distillation in this sprint. Those routes are deferred, not silently counted as completed.

## Experiments and acceptance

- Screen all three conditions on the same three official-training-patient folds, seeds 0–1: **18 CV fits**. Choose each condition's epoch from the earliest maximum mean source-validation curve; preserve the original 100-epoch cosine horizon during refit. Use distinct run IDs per CV seed.
- Before any SC-CV run, prove its reference uses training patients only, excluding validation/calibration patients. Estimate validation correction only under the declared unlabelled-pool protocol. The previous no-SC attention preflight does not certify this SC-CV path. Any required partition/reference correction is a launch gate.
- Freeze recipes/epochs before target evaluation. Evaluate all three conditions at five seeds on official ICBHI and all four LODO folds: **75 fixed-epoch fits**, trained/evaluated with static SC. Reuse existing controls only if epoch, initialization protocol, split and preprocessing match; budget assumes fresh controls.
- This is the existing Stage-1-source-CV → fixed-recipe LODO transfer protocol. Strict per-outer-fold nested hyperparameter selection is deferred and must not be claimed. No LODO or KAUH result chooses a recipe or SC mode.
- Run frozen KAUH inference for the three official models × five seeds. Report recording balanced accuracy/Se/Sp, filter consistency, coverage, size and uncertain fraction. Keep the established recording/window rule; no KAUH tuning.
- Produce paired patient/seed intervals, checkpoint/prediction hashes and per-device tables. Select deployment candidates from source validation, not the highest test row. If gains fail to materialize, retain the current model.

## Allocation and schedule

| Elapsed time | Four RTX 5090s | RTX 6000 PRO |
| --- | --- | --- |
| 0–4 h | Verify source/configs and SC-CV boundaries; representative real GPU timing/memory tests | If available early, prepare exports/cache; otherwise reserve for day 2 |
| 4–12 h | 18 CV fits; four workers, initially one job/GPU | Optional preprocessing |
| 12–38 h | 75 final fits, balanced across conditions/folds/seeds | Prefer its 24 h allocation during hours 24–48: export/infer as official checkpoints arrive |
| 38–48 h | Finish admitted jobs; artifact verification; no late scope expansion | KAUH, inference/runtime profiling and remaining exports |

CPU patient bootstrap and report assembly run alongside GPU work. Protect the existing API. This allocation assumes dedicated worker availability; unfinished old campaigns must finish or have their resource allocation explicitly resolved before T0. Their GPU time is not free or hidden in this budget.

## Time budget and fallback

Historical RTX 5090 PAFA smoke extrapolation: 0.602 h/100 epochs and 7.06 GB allocator peak (`logs/grid_smoke_B.json`). This is not an end-to-end measurement of A/B. Budget approximately **100–140 RTX-5090 GPU-hours including overhead/recovery**, leaving **52–92 GPU-hours** of the 192-hour allocation uncommitted. Reserve **8–18 RTX-6000-PRO GPU-hours** for exports/inference/profile, within its 24 h; no equivalence to RTX 5090 throughput or unverified VRAM specification is assumed.

The 48-hour schedule requires an observed mean full-fit completion time of about **1.2 GPU-hours or less**, including export overhead. At 1.2 h, 75 fits cost 90 GPU-hours; 18 smaller CV fits at an assumed 0.6 h cost 10.8, with roughly 25% extra allowance giving about 126. These are planning assumptions that must be replaced by the initial timing samples. Spread late expensive folds early enough to avoid a long tail.

If measured throughput cannot meet the time budget, **drop B before target evaluation**, retaining control+A: 12 CV fits and 50 final fits. Keep five final seeds and all four LODO folds. If SC-CV reference isolation fails, fix it before those runs; dropping B does not repair a broken split. Report a reduced/incomplete scope if that gate consumes the available window. Do not trade away valid calibration/partitioning to keep another candidate. Stop admitting training by hour 38 unless its measured finish time leaves verification time before hour 48. Preserve incomplete units explicitly.

Deliverable by hour 48: complete paired tables for the accepted conditions across ICBHI/LODO/KAUH, uncertainty/efficiency and runtime evidence, plus a validation-based recommendation. No promise of a higher Score; no claim that software-filter consistency proves physical-device robustness. No new hardware or data purchases are required for this shortlist.
