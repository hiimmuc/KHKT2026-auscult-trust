# Distilling the ensembles into one BEATs model: plan for A, B, D, E (and F if allowed)

| Type | Date | Status |
| --- | --- | --- |
| plan | 2026-10-28 | proposed, not launched |

Question: can a single BEATs + PAFA + static-SC + gain6 student recover the +1.0 official and +0.5 LODO-macro gain of the equal-weight
ensembles of the target-evaluation arms? Evidence cut-off: 28/10/2026, campaign `target-20261027` (175 fits complete).

## 1. What is compared

| Item | Choice |
| --- | --- |
| Student | anchor recipe (`base, encoder_beats, grid/pafa_beats, sc, sc_static, sc_gain`), fixed epoch 66, seeds 0-4; reference = `gain6-cv66` finals (official 63.13, LODO macro 61.62) |
| Teacher group G4 | anchor + control PCSL-5 + class-aware light + multiscale (ensemble 64.11 official / 62.22 macro, exploratory) |
| Teacher group G7 | all seven arms (64.21 / 62.09) |
| KD variant K1 | soft targets only, temperature 1 (the 24/10 pilot: `--soft_targets` alone) |
| KD variant K2 | hard weight 0.5 plus soft KL at temperature 2, KL x T^2 (`--kd_hard_weight 0.5 --kd_temperature 2`) |
| Cohorts | official + AKGC417L, Litt3200, LittC2SE, Meditron held out; a LODO student uses only teachers trained without that device |
| Pairing | student seed s uses the ensemble of the seed-s members, so every contrast is paired by seed with the anchor and with the teacher |

4 configurations x 25 fits = 100 student fits.

## 2. Cost, from measured times

Measured median wall hours of one final fit at epoch 66 with two jobs per GPU: 0.56 official, 0.50 LODO (anchor finals). Teacher export of one
member is estimated at 2-3 min (4,142 training cycles of inference plus model build; not measured).

| Block | Work | Slot-hours |
| --- | --- | ---: |
| Teacher export, official | 5 seeds x 11 member passes (G4 4 + G7 7) | ~2.5 |
| Teacher export, LODO | 20 cohort-seeds x 11 member passes | ~10 |
| Phase 1: students, official | 4 configs x 5 seeds = 20 fits x 0.56 h | ~11 |
| Phase 2: students, LODO | 4 configs x 20 fits x 0.50 h | ~40 |
| Total | | ~64 |

Capacity: B, D, E two slots each, A one slot at about half speed: 6.5 effective slots; F adds about 2.4 if allowed (30 GB cap).
- Phase 1 answer (official, all four configurations): about 2 h wall on A, B, D, E (about 1.5 h with F).
- Everything: about 10 h wall without F, about 7 h with F. With the gate below, Phase 2 shrinks to the configurations that pass.
- Speed-ups: export each member once and average per group (needs a short combiner, about 40 lines plus a test) instead of re-running members per group
  (saves 4 of 11 passes); start LODO teacher exports while Phase 1 students train, so Phase 2 does not wait.

## 3. Order and gate

1. Teacher exports for the official cohort (both groups), launch Phase 1 students as soon as each target file exists.
2. In parallel, export LODO teacher targets.
3. Gate on Phase 1 (fixed now, before any result): a configuration continues to LODO only if its official mean Score is at least the anchor's
   + 0.3 and at least 3 of 5 seeds are above the anchor's same seed. Others stop and the negative result is reported.
4. Phase 2 for the surviving configurations; then paired analysis (official, four LODO, macro, KAUH A, screening) as for the other arms.

## 4. Expected outcome and risks

- [Interpretation] Teacher targets on the optimizer training cycles are in-sample (the members were trained on them), so they are close to one-hot. The
  student then gets little more than the labels, and the 24/10 pilot gained only +0.25 official (62.75 vs 62.50) with intervals including zero.
  The refined K2 loss failed its 3-fold source gate (+0.009). A realistic outcome is a student between the anchor and the ensemble,
  perhaps +0.2 to +0.5 official; a result above +1.0 is unlikely with these targets.
- Temperature in K2 softens in-sample targets and may help; K1 is the control for that.
- A stronger option, not included for speed: out-of-fold teacher targets (members trained on other patients) or augmented-view targets. Both need new teacher training.
- Selection of G4 was made on the same test data as the evaluation (exploratory, optimistic); G7 uses every arm and needs no selection.
- Distilled students must be evaluated exactly as the other arms; the serving model and the registered criteria do not change.

## 5. Decisions needed

1. Include F (shared, slower, 30 GB cap) for Phase 1? About 0.5 h faster.
2. Approve the gate in section 3, or relax it to run LODO for all four configurations regardless (about 10 h, no early stop).
