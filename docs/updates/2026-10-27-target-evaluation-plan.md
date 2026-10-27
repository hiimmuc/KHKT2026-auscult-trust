# Target evaluation of the stage-12 recipes: plan for machines A, B, D, E (27/10/2026)

| Type | Date | Status |
| --- | --- | --- |
| plan | 2026-10-27 | proposed, not launched |

Status: **proposed, not launched.** Nothing in this plan changes the frozen AuscultTrust (BEATs + PAFA + trained static SC),
the serving bundle or any registered criterion. Context: `report.md` section 17 (stage 12) and
`docs/reports/2026-10-27-findings.md`.

## 1. Goal

Five recipes passed or approached the source gate in five-fold source CV but have no official, LODO or KAUH result:
SWA, LoRA, multi-scale, class-aware light (PCSL 5) and LoRA + class-aware. Measure the three missing endpoints with
paired seeds, without using any target outcome to choose a recipe or epoch.

| Endpoint | Unit | Output |
| --- | --- | --- |
| Official | retrain on the official training split to the source-selected epoch, 5 seeds | Score / Sp / Se / HS, per device and pooled |
| LODO | retrain with each of AKGC417L, Litt3200, LittC2SE, Meditron held out, 5 seeds | per-device Score, calibration split for conformal |
| KAUH | frozen official-split checkpoint, cycle and recording inference with static SC | A flip rate, B2 coverage error, ICBHI recording screening |

Each endpoint is produced by the existing final pipeline (`scripts/improvement_unit.py`: fixed-epoch fit, `export_probs`,
`screening infer`, `kauh_infer`, receipts with SHA-256), i.e. 25 fits per arm: 5 official + 20 LODO; the 5 official units also
produce screening and KAUH. This is exactly the shape of the completed LLRD and gain6-cv66 finals.

## 2. Arms and fixed epochs

Epochs are the source-selected epochs of the 150-fit run (`build/campaigns/fivefold-20261026/outputs/evaluation/fivefold-20261026/summary.json`).
The training horizon stays 100 epochs, so the schedule matches the source run, and the fit stops at the selected epoch.

| Arm | Epoch | Tier | Why |
| --- | ---: | --- | --- |
| class-aware light | 4 | 1 | largest source gain (+1.458); epoch 4 is cheap (see caveat 1) |
| LoRA + class-aware | 43 | 1 | second gain (+0.695) |
| LoRA | 83 | 1 | gain +0.607 |
| PCSL-5 control (PAFA, no class-aware pairing, PCSL 5) | needs its own source CV | 0 | separates pairing from the PCSL change; **has no source result yet** |
| gain6 anchor | 66 | reference | official / LODO already complete (cell `gain6-cv66`, section 4) |
| SWA | 43 | 2 (optional) | gain +0.076, not above seed noise |
| multi-scale | 68 | 2 (optional) | gain -0.321; most expensive arm (16 GB, 2.0x cost) |

Tier 0 has to run first because its epoch determines the control's final fits. Tier 2 is listed so the user can decide;
by the source results, SWA and multi-scale are the least likely to repay their cost.

## 3. Cost, from measured times

Measured: median wall time of the 150 source-CV fits (100 epochs, two jobs per GPU, 4/5 of the training patients),
from the `train_args.json` to `completion.json` file times: anchor 0.62 h, SWA 0.63, LoRA 0.52, multi-scale 0.82,
class-aware light 0.77, LoRA + class-aware 0.60 (maxima 0.93-1.59 h). A final unit trains on more patients (about 1.25x) and
stops early, so cost per unit is roughly `h_cv * epoch / 100 * 1.25`, plus about 0.1 h for the three exports on official
units. These are estimates (+-30%) in **slot-hours** (one slot = one of two concurrent jobs on a GPU).

| Block | Fits | Estimate (slot-h) |
| --- | ---: | ---: |
| Tier 0: PCSL-5 control, 5 folds x 5 seeds, 100 epochs | 25 | 16 |
| Tier 0: PCSL-5 control final at its selected epoch (if about 66) | 25 | 13 |
| Tier 1: class-aware light, epoch 4 | 25 | 3 (startup dominates) |
| Tier 1: LoRA + class-aware, epoch 43 | 25 | 8 |
| Tier 1: LoRA, epoch 83 | 25 | 14 |
| Tier 2: SWA, epoch 43 | 25 | 9 |
| Tier 2: multi-scale, epoch 68 | 25 | 17 |
| Reference: gain6 anchor re-run in the new snapshot (only if reuse fails, section 4) | 25 | 13 |

Totals: tier 0 + 1 = **about 54 slot-h** (125 fits); with tier 2 = **about 80 slot-h** (175 fits).

Capacity: B 2 slots, D 2, E 2 (28 GB admission each), A 1 slot (weight 0.55; the RTX 5000 Ada also serves the demo API, which
uses about 1.2 GB). About 6.5 effective slots, so tier 0 + 1 takes about 8-9 h wall and the full set about 12-13 h wall,
including the control's dependency (the 25 CV fits, about 3 h on all slots, gate the control's finals). Machine F is not
used: it is shared with another user.

## 4. Anchor reference

`gain6-cv66` has 25 completed finals (5 official + 20 LODO, epoch 66) on B under
`build/campaigns/followup-20261026/outputs/train/followup-20261026-final/`. Reuse them as the paired reference only if all hold:
identical `train_args.json` apart from run / cell names, same cycle IDs and patient lists, same SC state hashes, same
code digests for the files on the training path. Otherwise re-run the 25 fits inside the new snapshot (+13 slot-h).
The KAUH and screening exports of the official anchor units must also exist; if not, run them through frozen inference only.

## 5. Preflight (before any training)

1. **Frozen inference parity for the new arms.** For one existing five-fold CV unit each of LoRA, SWA and multi-scale (and
   class-aware light), load `best.pth` through `src/inference/model.py`, re-predict the validation fold and compare with
   the saved `val_preds.npz`. This catches LoRA weights dropped by a non-strict load and the SWA-averaged weights
   not being the exported ones. Stop on any mismatch above float tolerance.
2. **New control config.** Add `experiments/train/pcsl5_control.yaml` (`method: ce_pafa`, `lambda_pcsl: 5.0`,
   `lambda_gpal: 0.0005`) and register the arm; unit-test that it builds a plain PAFA with PCSL 5.
3. **Snapshot.** Build `build/campaigns/target-20261027/` with the existing `prepare` mechanism (hashes of source, configs and
   environment), from the current tree (which already contains the 27/10 `BeatsEncoder` default fix). Verify hashes on
   every worker before the first launch; never edit a frozen snapshot afterwards (the stale E service shows the failure mode).
4. **Disk.** A BEATs checkpoint is about 0.37 GB (measured on one SWA unit), about 50 GB for tier 0 + 1. A's `checkpoints/` and `outputs/`
   are already symlinks to the HDD (`/storage/hdd/disk0/namdp15/auscult-trust-A`; SSD 255 GB free); a new snapshot's
   `outputs`/`checkpoints` should be created there too. B, D, E have 144-356 GB.
   B stays canonical storage; E the verified archive; collect and hash as in the five-fold run before any deletion.
5. **Services.** All `auscult-*` campaign services on A, B, D, E were stopped and disabled on 27/10 (only `auscult-inference-api` on A runs);
   confirm with `systemctl --user list-unit-files 'auscult*'` before launching.
6. **No Codex session editing `src/`, `scripts/`, `experiments/` or `report.md` while a snapshot is being built.**

## 6. Execution

1. Preflight, then launch tier 0 CV (25 fits) together with tier 1 finals (75 fits) so no slot waits.
2. When the control's CV completes: select its epoch from its own mean curve (same rule), freeze, launch its 25 finals.
3. The controller (`scripts/improvement_campaign.py` core, as in `fivefold_campaign.py`) runs each fit, verifies receipts and
   checkpoint SHA-256, and mirrors to B and E. Resume only from complete units; a cut-off unit reruns from epoch 0.
4. Tier 2 starts only on the user's decision, after tier 1.

## 7. Analysis and stop rules

Paired by seed and patient against the anchor (and against the PCSL-5 control for class-aware arms); patient / seed bootstrap
with recalibration in each draw for conformal; KAUH A / B2 with the existing null gates. Report official, each LODO device
and the macro, worst-device Score, set size and coverage, KAUH flip rate and screening Se / Sp. No arm is promoted on a source
result: promotion to a candidate for the serving model needs a positive paired interval on official **and** LODO macro and no
worse worst-device Score, and would still be reported as exploratory because the recipes were chosen after the earlier
benchmark inspection. Stop rule: if the anchor reuse check or the frozen-inference parity check fails, stop and report.

## 8. Decisions needed from the user

1. Tiers: tier 0 + 1 only (about 54 slot-h, 9 h wall) or also SWA and multi-scale (about 80 slot-h, 13 h wall)?
2. PCSL-5 control: evaluate at its own source-selected epoch (recommended, 25 CV fits first) or match class-aware light's
   epoch 4 directly (cheaper, 3 slot-h, but it uses a non-selected epoch and answers only "is epoch 4 itself the cause")?
3. Run one worker on A's GPU (checkpoints already on the HDD; services already disabled)?

## 9. Caveats that apply to every result of this plan

1. Class-aware light is selected at epoch 4 of a 100-epoch cosine schedule (an early-stopped model); the
   anchor reaches 55.8 at that epoch. A gain at epoch 4 may reflect shallow fine-tuning rather than the recipe.
2. The 3-fold screen and the 5-fold run disagree in sign for LoRA (-3.25 vs +0.61); the source gains are within the
   fold-to-fold spread (fold 1 contributes +4.2 / +5.1). The same anchor recipe, folds and seed was selected at epoch 21 in
   one programme and 73 in another, so selected epochs are noisy.
3. Recipe and epoch selection used source validation curves of the same data; the surrounding benchmark was inspected earlier.
4. KAUH filters are software, not hardware; H2 on LODO has no tested fold, so conformal results are descriptive.
