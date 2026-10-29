# Target evaluation (target-20261027) and distillation (distill-20261028): results

| Type | Date | Status |
| --- | --- | --- |
| report | 2026-10-28 | current (exploratory; no registered criterion changed) |

Question: do the stage-12 recipes (SWA, multi-scale, class-aware light, LoRA, LoRA + class-aware, PCSL-5 control) add to the
anchor on the official split, the four-device LODO and KAUH, and can an equal-weight ensemble of them be distilled into one
BEATs + PAFA + static-SC student? Evidence cut-off: 28/10/2026 23:18 (+07). All numbers are Score = (Sp + Se) / 2 on ICBHI
4-class, mean over seeds unless stated. Everything here is exploratory: the teacher and ensemble groups were chosen on the
same test data they are scored on.

Sources: `build/campaigns/target-20261027/` (175 fits), `build/campaigns/distill-20261028/` (39 fits: 15 official, 24 LODO),
`outputs/kauh/target-20261027/kauh_summary.md`, `outputs/kauh/distill-20261028/kauh_summary.md`,
`outputs/screening/{target,distill}-20261028/screening_summary.md`, ensemble script output (`/tmp` analysis, reproducible from
`test_preds.npz` of each unit with equal-weight softmax mean over the last epoch).

## 1. Target evaluation: single arms (5 seeds)

| Arm | Official | LODO macro (4 devices) | Notes |
| --- | ---: | ---: | --- |
| anchor (`gain6-cv66`, epoch 66) | 63.13 | 61.62 | reference |
| SWA (epoch 43) | 63.32 | 61.77 | +0.2, within seed noise |
| multi-scale (epoch 68) | 62.45 | 61.41 | |
| PCSL-5 control (epoch 78) | 62.16 | 60.72 | non-class-aware control |
| class-aware light (epoch 4) | 58.66 | 60.07 | source-CV gain did not transfer |
| LoRA (epoch 83) | 57.05 | 57.81 | source-CV gain did not transfer |
| LoRA + class-aware (epoch 43) | 54.84 | 55.11 | worst |

[Fact] Source-CV gains of class-aware light and LoRA did not transfer to the official split or to LODO.

## 2. Equal-weight ensembles of the target arms

Softmax mean of the members' last-epoch `test_preds.npz`, paired by seed. Ensembles over all 2^7 - 1 subsets were scored.

| Ensemble | Official | LODO macro |
| --- | ---: | ---: |
| all seven arms | 64.21 ± 0.74 | 62.09 ± 1.36 |
| anchor + control + class-aware light + multi-scale (G4) | 64.11 ± 0.65 | 62.22 ± 1.21 |
| anchor + class-aware light + SWA + multi-scale | 64.08 ± 0.27 | 62.25 ± 1.23 |
| anchor + control + class-aware light + SWA + multi-scale | 63.99 ± 0.72 | 62.19 ± 1.54 |

[Fact] Ensembles add about +1.0 official and +0.5 to +0.6 LODO macro over the anchor. Many subsets reach the same level,
so the best subset is not stable.

[Fact] Selection check: the subset chosen on seeds 0-1 (G4) scores 64.15 official / 62.60 macro on seeds 2-4, against the
anchor's 63.56 / 62.45 on the same seeds. The subset chosen on seeds 2-4 gives 63.16 / 60.98 on seeds 0-1, against the
anchor's 62.50 / 60.37. The gain holds on held-out seeds, but it is small.

[Interpretation] Most of the ensemble gain comes from diversity among members that are each about as good as the anchor.
Adding the weak LoRA arms does not help the official score and lowers the LODO macro.

## 3. KAUH (frozen models, seeds 0-4)

Base for paired contrasts: `frozen` is the 23/10 AuscultTrust: BEATs + PAFA with trained static SC, epoch 100 (its `sc` flag is true and its checkpoint hashes equal `final-pafa-static/auscult`). [Correction 29/10] Earlier text called it "no SC"; the no-SC baseline of the 23/10 decision (KAUH flip 0.445) is `final-pafa-static/baseline`, so every "vs frozen" contrast in this section is against an SC model. Flip rate and TV distance across the three vendor filters (windows).
Full table: `outputs/kauh/target-20261027/kauh_summary.md`, `outputs/kauh/distill-20261028/kauh_summary.md`.

| Condition | Flip rate | TV distance | Flip reduction vs frozen [95% bootstrap] |
| --- | ---: | ---: | --- |
| frozen | 0.209 | 0.206 | |
| anchor | 0.204 | 0.205 | +0.004 [-0.040, +0.045] |
| PCSL-5 control | 0.212 | 0.206 | -0.003 [-0.046, +0.046] |
| class-aware light | 0.186 | 0.143 | +0.023 [-0.043, +0.093] |
| LoRA + class-aware | 0.072 | 0.086 | +0.137 [+0.076, +0.205] |
| LoRA | 0.040 | 0.062 | +0.169 [+0.122, +0.216] |
| SWA | 0.198 | 0.191 | +0.011 [-0.038, +0.055] |
| multi-scale | 0.241 | 0.225 | -0.032 [-0.086, +0.018] |
| distill kd_g4_k1 | 0.228 | 0.191 | -0.019 [-0.082, +0.038] |
| distill kd_g4_k2 | 0.213 | 0.190 | -0.004 [-0.055, +0.046] |
| distill kd_g7_k2 | 0.191 | 0.152 | +0.018 [-0.045, +0.068] |

[Fact] LoRA and LoRA + class-aware reduce the flip rate with intervals above zero. Their official and LODO scores are the
worst of the set.

[Interpretation] The flip reduction is more likely a sign of low-confidence collapse than of robustness; not tested.

[Fact] B2 (conformal coverage across filter pairs): 0 of 6 ordered pairs passes the null p95 gate for the frozen base (an SC model, whose |Delta| is already below the gate) and for
every arm, so no D statistic is tested. [Correction 29/10] With the no-SC baseline as base, 2 of 6 pairs are tested (23/10 and 29/10 frozen evaluations); the 0 of 6 here follows from using an SC model as base.

## 4. Recording screening (ICBHI calibration, applied to KAUH)

Summary: `outputs/screening/target-20261027/screening_summary.md`, `outputs/screening/distill-20261028/screening_summary.md`.
Columns: coverage, Δ, size, refer, uncertain, rerecord, screen_se/sp, argmax_se/sp, abnormal share; aggregation mean or max over
cycles of a recording; 20 calibration splits; 5 seeds.

[Fact] With mean aggregation, KAUH Bell coverage for the anchor is 0.805 (uncertain 0.471), against the frozen base (the 23/10 AuscultTrust with SC) 0.831
(uncertain 0.464). Distilled kd_g4_k1 gives 0.790 with uncertain 0.311; kd_g4_k2 0.800 / 0.327; kd_g7_k2 0.813 / 0.350.
Differences are small and not paired-tested.

[Interpretation] The distilled students send fewer recordings to "uncertain" on Bell and Diaphragm, at the cost of lower
coverage. This is the same direction as the PCSL-5 control and SWA, and it is not a clear improvement on its own.

## 5. Distillation (students at epoch 66, reference recipe, soft targets from the teacher group)

Teacher targets: in-sample ensemble softmax on each training cycle of the same cohort and seed, from the members' own
units (G4 = anchor, control, class-aware light, multi-scale; G7 = all seven). Losses: K1 = soft targets only;
K2 = hard CE weight 0.5 + soft KL at T = 2 (KL x T²).

Gate, fixed before results: official mean Δ ≥ +0.3 vs the anchor and ≥ 3 of 5 seeds above the anchor. Phase 2 (LODO, seeds 0-2)
only for configurations that pass.

| Config | Official mean | Δ vs anchor | Per-seed Δ (seeds 0-4) | Gate |
| --- | ---: | ---: | --- | --- |
| kd_g4_k1 | 64.15 | +1.02 | +0.8, +1.8, −0.5, +1.5, +1.4 | pass |
| kd_g4_k2 | 63.47 | +0.33 | −0.6, +1.8, −1.2, +0.6, +1.0 | pass |
| kd_g7_k2 | 62.75 | −0.38 | −0.6, +0.5, −1.8, −0.3, +0.3 | fail (no LODO) |

LODO (held-out device Score, seeds 0-2; anchor and teacher on the same seeds):

| Model | AKGC417L | Litt3200 | LittC2SE | Meditron | Macro | Δ vs anchor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| anchor | 54.92 | 60.85 | 68.16 | 60.98 | 61.22 | |
| teacher ensemble (G4) | 55.05 | 63.47 | 69.06 | 61.46 | 62.26 | +1.04 |
| kd_g4_k1 | 54.63 | 61.15 | 70.58 | 60.55 | 61.72 | +0.50 |
| kd_g4_k2 | 54.29 | 61.70 | 69.56 | 61.51 | 61.77 | +0.54 |

Per-seed LODO macro Δ vs anchor: kd_g4_k1 −0.41 / +2.12 / −0.21; kd_g4_k2 −0.46 / +2.37 / −0.28.

[Fact] The distilled K1 student recovers the full official gain of the G4 ensemble (64.15 vs 64.11) but only about half of its
LODO macro gain (+0.50 vs +1.04). Gains are on LittC2SE and Litt3200; AKGC417L is slightly lower.

[Fact] K2 (temperature and hard-label mix) does not beat K1 on LODO and is lower on official. G7 teachers (with the weak LoRA
arms) give no gain.

[Interpretation] Teacher targets are in-sample on the optimizer training cycles, so they are close to one-hot. The student
mostly learns the labels plus a small smoothing, which is consistent with the 24/10 pilot (+0.25 official, interval includes
zero) and with the gain being about half on LODO.

[Pending] Not done: KAUH and screening for the students cover the official units only (sections 3 and 4), not the LODO units;
no patient-level bootstrap; no distillation with out-of-fold or augmented-view targets.

## 6. Findings

1. [Fact] The best single-model result on the official split is distilled K1 (64.15), about the same as the G4 ensemble it
   copies (64.11) and +1.0 over the registered anchor (63.13). This is exploratory: G4 was picked on the test data.
2. [Fact] On LODO the student keeps about half of the ensemble gain (+0.5 of +1.0). The result is seed-dependent (seed 1 +2.1,
   seeds 0 and 2 negative).
3. [Fact] No arm improves KAUH invariance in a way that survives the bootstrap, and no B2 pair is tested.
4. [Interpretation] SWA is the most stable single addition (official +0.2, LODO +0.2); the class-aware and LoRA recipes from
   source CV do not transfer to target evaluation.
5. [Pending] The registered AuscultTrust is unchanged. Any change to the serving model needs the registered criteria and a
   separate decision.

## 7. Caveats

- One split, 5 seeds (official), 3 seeds (LODO); bootstrap over seeds only for Score differences (no patient-level bootstrap); the KAUH and B2 intervals are bootstrapped over patients and seeds.
- G4 selection and the ensemble subset choice used the same test scores; the seed-split check (section 2) is the only held-out
  check and it is on the same split.
- The LODO students used only the teacher members trained without that device; the teacher LODO files for four devices were
  synced from B to A, C, D and E after an initial stall (see `docs/handoff/HANDOFF.md`, incident of 28/10 20:50).
- KAUH filters are vendor software transformations of the same recordings, not physical stethoscopes.
