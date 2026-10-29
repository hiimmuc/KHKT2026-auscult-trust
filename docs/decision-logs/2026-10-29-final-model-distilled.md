# AuscultTrust redefined as the distilled student (29 October 2026)

| Type | Date | Status |
| --- | --- | --- |
| decision | 2026-10-29 | current; supersedes the AuscultTrust row of `2026-10-23-final-model.md`; exploratory evidence |

[Fact] The user decided on 29/10/2026 to define AuscultTrust as the highest-scoring single model of the evidence cut-off (28/10 23:18): the distilled
student `kd_g4_k1`. This replaces "BEATs + PAFA trained and evaluated with static SC, epoch 100" as the project model.

| Role | Configuration |
| --- | --- |
| **AuscultTrust (new)** | BEATs fine-tuned with PAFA and trained static SC, gain6 recipe, epoch 66, trained with soft targets only (K1: no hard-label term, temperature 1) from the equal-weight softmax of the G4 teacher group (anchor `gain6-cv66`, PCSL-5 control, class-aware light, multi-scale), plus the split conformal layer. One model at serving time |
| Teacher ensemble G4 | The four members above, equal-weight mean softmax; used for training only and as an exploratory reference (64.11 official, 62.22 LODO macro) |
| Reference for the distillation gain | Anchor `gain6-cv66` (epoch 66): 63.13 official, 61.62 LODO macro |
| Project baseline (23/10 decision, not an assumption) | BEATs + PAFA, fine-tuned, no SC, epoch 100 (official 61.7 ± 0.7, KAUH flip 0.445). Unchanged |
| Baseline for the student's SC contrast | The student's no-SC twin (same recipe and soft targets, SC off); training and evaluation in progress 29/10. The BEATs PAFA pair (61.7 → 62.3 official) is not evidence about the student |
| Previous AuscultTrust | BEATs + PAFA + trained static SC, epoch 100: official 62.3 ± 0.9, LODO macro 60.3 ± 0.8. Kept as a reference row |

## Evidence behind the choice (all exploratory)

Source: `docs/reports/2026-10-28-target-distillation-results.md`, `build/campaigns/distill-20261028/`.

| Endpoint | kd_g4_k1 | Anchor | Previous AuscultTrust | G4 ensemble |
| --- | ---: | ---: | ---: | ---: |
| Official Score, seeds 0-4 | 64.15 | 63.13 (+1.02; 4 of 5 seeds above) | 62.3 ± 0.9 | 64.11 |
| LODO macro, seeds 0-4 | 62.2 ± 2.0 | 61.6 ± 1.4 | 60.3 ± 0.8 | 62.26 (seeds 0-2 only) |
| KAUH flip rate (Bell, Diaphragm, Extended) | 0.228 | 0.204 | 0.209 (frozen model, 23/10) | not run |
| KAUH flip reduction vs the 23/10 AuscultTrust (SC) [95% bootstrap over patients and seeds; the 28/10 report calls that model "frozen"] | -0.019 [-0.082, +0.038] | +0.004 [-0.040, +0.045] | 0 | |
| Recording screening, Bell, mean aggregation | coverage 0.790, uncertain 0.311 | 0.805, 0.471 | 0.831, 0.464 | |
| KAUH B2 | no pair passes the null gate | same | same | |

## What the decision does not claim

- [Fact] The 1.02-point gain over the anchor and the LODO gain (LODO macro +0.6 ± 1.0 over five seeds, 3 of 5 positive) depend on the seed:
  per-seed official deltas are +0.8, +1.8, -0.5, +1.5, +1.4. The student keeps about half of the ensemble's LODO gain.
- [Fact] The teacher group G4 was chosen on the same test scores it is evaluated on. The seed-split check (chosen on seeds 0-1, scored on 2-4) gives 64.15 / 62.60
  against the anchor's 63.56 / 62.45, a small gain on the same split.
- [Fact] The student shows no KAUH flip-rate improvement and no coverage claim; H2 (conformal coverage under device shift) stays untested in LODO and
  needs the phantom. Fewer "uncertain" recordings come with lower coverage (0.79 against 0.81-0.83 for the Bell filter).
- [Fact] There is no patient-level bootstrap for the student, and its KAUH and screening results cover the official unit only (not the LODO units).
- The registered criteria of v9 are unchanged. The 23/10 frozen evaluation (`final-pafa-static`) was run for the previous AuscultTrust and has not been rerun.

## Required before the new definition is final

1. [Pending] Frozen final evaluation (`scripts/final_evaluation.py`) for the student: five paired seeds, official, LODO, KAUH, screening, with weights and
   `s_ref` frozen and hashed before any phantom recording.
2. [Pending] Serving bundle: `app/` still serves the previous model (`app/bundle-final-pafa-static`). Student checkpoints exist in the `distill-20261028` units on
   the training machines (epoch 66); activate only after bundle validation (hash, health check).
3. [Pending] Whether the ensemble extensions E1 and E1+E3 of the 23/10 log are replaced by G4; whether the 5-seed student ensemble used in the 29/10 replay report is only an exploratory variant.
4. [Pending] Phantom H1, H3, co-primary H2 on the frozen student.

## Frozen evaluation, first part (29/10, run `final-kd-g4-k1`)

[Fact] Five paired seeds, current code, hashes of weights and `s_ref` frozen with serving seed 0; no bundle built or activated. Baseline and previous AuscultTrust
reproduce their 23/10 KAUH flip rates (0.445, 0.209) and screening, so the inference path is unchanged.

| Condition | KAUH-A flip | Reduction vs baseline [95% CI] | B2 D vs baseline (2 tested pairs) | ICBHI screening uncertain |
| --- | ---: | --- | --- | ---: |
| baseline | 0.445 | | | 0.570 |
| previous AuscultTrust | 0.209 | +0.236 [+0.177, +0.305] | +0.113 [+0.072, +0.153] | 0.603 |
| anchor | 0.204 | +0.240 [+0.196, +0.285] | +0.113 [+0.070, +0.146] | 0.559 |
| **student** | 0.228 | +0.217 [+0.166, +0.266] | +0.100 [+0.059, +0.130] | 0.525 |

[Fact] The student is not more invariant across filters than the previous model; its gain is in Score (section 17.7 of the comprehensive report) and in fewer "uncertain"
recordings with slightly lower KAUH coverage. Still running: the no-SC twin (official split and four held-out devices, 5 seeds), LODO seeds 3-4 of the student, and the
analysis (LODO five-seed table, null gate and H2 per fold, unseen-patient view, best-on-test). Until the twin is evaluated, no "SC helps the student" claim stands.
