# AuscultTrust: consolidated results report

| | |
| --- | --- |
| Report version | 6.0 (27/10/2026): restructured by plan stage; adds stage 12 "Additional improvement trials" |
| Project definition (provisional) | **AuscultTrust = BEATs + PAFA, fine-tuned, trained static spectrum correction (SC) + split conformal layer.** Baseline = the same BEATs + PAFA recipe without SC. Exploratory extensions: equal-weight ensembles E1 and E1+E3. Decision log: `docs/decision-logs/2026-10-23-final-model.md` |
| Evidence cut-off | 27/10/2026 16:30 (+07). Nothing from stage 12 changes the deployed model or the registered criteria |
| Proposal / prereg | v10 (`docs/updates/research-proposal-v10-en.md`, 23/10; model and evidence updated, registered criteria of v9 unchanged); registered study v9 (`docs/updates/research-proposal-v9-en.md`) with `docs/updates/prereg-v8.md` and Amendments 1-5 (`docs/updates/prereg-v9-amendment*.md`) |
| Plans | `docs/updates/stage-plan-v1.md` (grid), `docs/updates/lodo-plan-v1.md` (LODO, ensembles; Amendment 4 wins where they differ), `docs/updates/2026-10-27-target-evaluation-plan.md` (official / LODO / KAUH for the five stage-12 recipes; proposed, not launched) |
| Detailed sources | `docs/reports/` (date-first names): `2026-10-09-registered-core-report`, `2026-10-19-grid-v1-results`, `2026-10-20-pcmcl-v1-results`, `2026-10-21-lodo-v1-results`, `2026-10-22-ensemble-v1-results`, `2026-10-23-final-pafa-static-execution` (frozen final evaluation), 24/10 audits (`benchmark-protocol-audit`, `ensemble-finite-n`, `sc-finite-n-uncertainty`, `class-conditional-conformal`, `pafa-pooling-audit`), `2026-10-27-improvement-results`, `2026-10-27-findings` |
| Document index and conventions | `docs/README.md` (naming, template, list of removed interim documents) |
| State and commands | `docs/handoff/HANDOFF.md`, `docs/QUICKSTART.md`, `docs/SETUP.md`, `app/README.md` |

Tags: [Fact] measured in the repo, [Interpretation] reasoned, [Assumption], [Pending] not run, [Preliminary] computed on
the units finished so far (may change; no verdict). Score = (Sp + Se) / 2 on ICBHI 4-class, official split unless
stated. All numbers are mean ± SD over seeds 0-4 unless a seed count is given.

## 1. Status and key findings

**Where the project stands [Fact].** Every training run of the main study is finished and verified. The selected
comparison is BEATs + PAFA without SC versus BEATs + PAFA with trained static SC (fixed epoch 100, seeds 0-4). A
second line of work (stage 12) searched for training improvements on source-patient cross-validation. It produced one
candidate that is not yet evaluated on any target endpoint, and several negative results that are kept.

Key findings, in the order of the plan stages:

1. [Fact] Encoder choice dominates method choice: BEATs leads on the official split (stage 2) and under device shift
   (stage 3); most methods stay within one seed SD of CE on the same encoder.
2. [Fact] BEATs PAFA is the one consistent method gain (+2.6 Score, +8.5 worst-device Score over BEATs CE on the official
   split) and the best cell on the hardest unseen device (Litt3200).
3. [Fact] H2 cannot be tested in LODO v1: no held-out device shows a coverage error above chance once models are trained
   to their last epoch (stage 7). The registered AST study (models stopped at epoch 2) did show one.
4. [Fact] In LODO v1, SC helps Score, with no established coverage improvement: test-time SC raises Litt3200 Score for every
   cell (stage 6); trained SC raises the LODO macro Score for 6 of 7 cells at no cost on the official split (stage 4).
5. [Fact] Selected comparison: official 61.7 ± 0.7 (no SC) vs 62.3 ± 0.9 (static SC); LODO macro 58.8 ± 1.4 vs
   60.3 ± 0.8. The gain is uneven across devices (+5.5 LittC2SE, +1.9 Litt3200, -1.8 AKGC417L; section 15.1). The recipe
   and the SC variant were chosen after the official-split results were seen, so these are exploratory estimates.
6. [Fact] Ensembles raise Se at PAFA's Sp only with a target-side anchor: equal-weight E1+E3 or E1 with a label-free rule
   that calls as many cycles normal as PAFA does on the same unlabelled cycles keep Sp and gain 1.4-3.1 Se (+1.1 to +2.1
   Score); learned member weights never beat equal weights (stage 9).
7. [Fact] The dynamic SC bound helps SG-SCL and Patch-Mix CL on the official split but lowers the LODO macro for PAFA
   (-0.9); static is the better bound for BEATs PAFA. PAFA + PC-MCL with static SC has the best LODO macro (60.7)
   and the smallest sets (stage 8).
8. [Fact] Both KAUH positive controls pass on the frozen final model (stage 11): flip rate 0.445 -> 0.209 (reduction
   +0.236 [+0.177, +0.305]); coverage-error reduction +0.113 [+0.072, +0.153] on the 2 of 6 pairs above the null gate.
   Software filters only: no hardware claim follows. Recording screening keeps a high uncertain rate (60.3%).
9. [Fact] Stage 12, source CV only: a lighter class-aware PAFA (PCSL 5) gained +1.46 Score points over a matched anchor
   in 5 folds x 5 seeds, LoRA +0.61, LoRA + class-aware +0.70, SWA +0.08, multi-scale -0.32. The same recipes failed the
   source gate in an earlier 3-fold screen (section 17.5), the selected epoch of class-aware light is 4 of 100, and its
   PCSL change is confounded with the pairing change. [Pending] official, LODO and KAUH for all five.
10. [Fact] Stage 12 negatives: LLRD (official 62.31 vs 62.42 for gain6), patient cap, class top-k/attention, band noise,
    recording head, event heads (-1.64 source Score) and the refined hard + soft KD (+0.01, 3-fold) did not pass the
    source gate; target-spectrum adaptation moved KAUH Score by +3.38 / -0.74 / -0.56 with intervals crossing zero.
11. [Pending] Phantom (H1, H3, co-primary H2) needs hardware. The registered v9 criteria are unchanged.

## 2. Overall plan

The pipeline: encoder x method grid on the official split, then LODO without SC, then SC (trained SC on 7 cells, and
test-time SC on the LODO models), then the final configuration; ensembles and distillation are optional. KAUH controls,
the screening view and the phantom tests close the project. Stage 12 collects the training-improvement trials that
followed the final decision; they never replace stage 10 without a paired target evaluation.

| # | Stage | What | Status |
| --- | --- | --- | --- |
| 0 | Phase 0 (v4-v8) | Frozen-encoder ladders, Part II shift study, Stage-1 screening of device variants | Done (section 5); `docs/reports/2026-10-07-phase0-report.md` |
| 1 | Registered core (v9) | Gate G on AST Patch-Mix CL vs AuscultTrust (AST + SC); LODO on AST (baseline, Freq-MixStyle, trained and test-time SC, A2, TTA-EQ); KAUH control A | Done 09/10 (section 6) |
| 2 | Grid v1 | 5 encoders x 14 methods (8 fine-tuned, 6 frozen) x 5 seeds, official split | Done 18/10 (section 7) |
| 3 | LODO v1 without SC (L1) | 11 cells x 4 held-out devices x 5 seeds = 220 units | Done 21/10 (section 8) |
| 4 | Trained SC, static bound | 7 cells, official split + 4 LODO folds, 175 units | Done 21/10 (section 9) |
| 5 | PC-MCL (Amendment 5) | BEATs PC-MCL and PAFA + PC-MCL, official split | Done 20/10; rule not met (section 10) |
| 6 | Inference arms (L2) | Test-time SC (static, dynamic), A2, TTA-EQ on every LODO unit (`src.evaluation.lodo_ttsc`) | Done 21/10 (section 11) |
| 7 | Analysis (L3) | Per-fold Sp/Se/Score/HS/2c; method vs CE; H2 (prior-matched coverage, null gate, two-level bootstrap); SC vs no SC paired by seed | Done 21/10: H2 not testable (section 12) |
| 8 | Trained SC, dynamic bound | BEATs PAFA, SG-SCL, Patch-Mix CL (75 units) + BEATs PAFA + PC-MCL with SC static and dynamic (50 units) | Done 23/10 (section 13) |
| 9 | Ensembles, distillation | Registered E1-E4; user groups by encoder with equal and fitted weights | User groups done 22/10; registered E1-E4 computed 24/10 (sections 14, 12.2); KD pilot and refined KD screen in section 17.4 |
| 10 | Final AuscultTrust | Choose encoder + method + SC with the user | Decided 23/10 (section 15) |
| 11 | After LODO | Frozen KAUH A/B2/B1 and recording screening; phantom H1/H3 | Computation done 23/10; phantom pending hardware (section 16) |
| 12 | Additional improvement trials | Source-CV search for training improvements (25/10-27/10): attention, gain6, LLRD, event heads, KD, SWA, LoRA, multi-scale, class-aware PAFA; adaptation, fusion, audio audit | Source CV done; target evaluation of 5 recipes [Pending] (section 17; plan `docs/updates/2026-10-27-target-evaluation-plan.md`) |
| - | Demo app (`app/`) | FastAPI + web demo with live recording; serves the BEATs + PAFA static-SC bundle | Cloudflare Pages front end, GPU API behind Tailscale Funnel (`app/README.md`) |

## 3. Decisions of the user

| Date | Decision |
| --- | --- |
| 15/10 | Stop earlier campaigns; run the method x encoder grid (Lung-SRAD as a reference row only, no noise arm). |
| 16/10 | SG-SCL with the DANN ramp (`grl_schedule: progress`) after the reference schedule collapsed BEATs. |
| 17/10 | After the power cut, rerun the 13 resumed units cleanly and swap them in. |
| 19/10 | LODO cells: AST ce, pafa, patchmix_cl; HTS-AT ce, patchmix_cl, sg_scl; BEATs ce, patchmix_cl, sg_scl, pafa, qlung (BEATs CE replaces ce_pafa_cfg). 80/20 split stage dropped. Ensembles and distillation optional. |
| 19/10 | Trained SC cells: AST patchmix_cl, pafa; HTS-AT patchmix_cl, sg_scl; BEATs patchmix_cl, pafa, sg_scl. Static bound first, dynamic later. PC-MCL and PAFA + PC-MCL on BEATs on machine D (Amendment 5). |
| 20/10 | PC-MCL rule not met: SC cell list unchanged. Machine C dropped after the third kill by its user. Cut-off units always rerun from epoch 0. |
| 20/10 | Cleanup: unused checkpoints and caches deleted (about 175 GB); grid-v1, lodo-v1, lodo-ast, lodo-bench checkpoints and all outputs kept. |
| 21/10 | Dynamic-bound SC only on BEATs PAFA, SG-SCL, Patch-Mix CL; add BEATs PAFA + PC-MCL with SC static and dynamic. |
| 22/10 | Final AuscultTrust is BEATs + PAFA, fine-tuned (replaces the AST Patch-Mix CL definition of v9). Ensembles by encoder group; report best epoch on test as an optimistic column only. |
| 23/10 | Baseline = BEATs + PAFA without SC; AuscultTrust = BEATs + PAFA with trained static SC; E1 and E1+E3 as exploratory extensions. |
| 24/10 | Improvement work limited to a 48-hour sprint on the available GPUs; the larger programme is deferred. |
| 26/10 | Follow-up on B/D/E/F only (A busy with the demo API): recover E results, refit the gain6 anchor, test event heads, fixed ensemble additions. No new architectures, no target-based selection. |
| 26/10 | Five-fold, five-seed source CV of SWA, LoRA, multi-scale, class-aware light and LoRA + class-aware against a matched anchor. External pretraining skipped. |
| 27/10 | Provisional project definition kept as BEATs + PAFA + static SC. Plan official / LODO / KAUH for the stage-12 recipes on A, B, D, E (this report, section 19). |

## 4. Compute, timeline and incidents

### 4.1 Machines (27/10)

| Machine | Address | GPU | Role |
| --- | --- | --- | --- |
| A (local) | 10.148.182.134 | RTX 5000 Ada 32 GB | analysis, serving (demo API on port 8000); outputs and checkpoints live on the 9.1 TB HDD (`/storage/hdd/disk0/namdp15/auscult-trust-A`, symlinked into the repository; moved 27/10) next to the mirrors in `auscult-remote-mirror`; SSD 255 GB free |
| B | 10.148.183.8 | RTX 5090 32 GB | training; canonical campaign storage (`build/campaigns/*`) |
| D | 10.148.183.171 | RTX 5090 32 GB | training |
| E | 10.148.182.179 | RTX 5090 32 GB | training; independent verified checkpoint archive |
| F | 10.148.182.137 | RTX PRO 6000 96 GB | shared with another user; used in stage 12 with a 64 GB project cap; not in the next plan |
| C | 10.148.183.119 | RTX PRO 6000 96 GB | dropped 20/10 (section 4.2); never start anything there |

- [Fact] `scripts/lodo_move.py` moved the remaining static-SC units between machines (stop a scheduler by PID while its
  jobs finish, set cut-off units aside to rerun from epoch 0, split by measured throughput, copy outputs to A).
- [Fact] Throughput in B-smoke hours per wall hour: A 0.80, B 1.94, E 1.96; B, D and E ran the last static-SC units at
  1.57-1.73. Stage-12 campaigns weight A at 0.55 and F at 0.25 relative to a 5090.
- [Assumption] +-20% on every estimate.

### 4.2 Incidents

- [Fact] Power cut 17/10 10:43 (both machines, 9.6 h): 13 resumed grid units rerun cleanly and swapped in (resumed vs rerun
  Score: mean abs 0.61, largest 2.5).
- [Fact] Machine C: our jobs were killed by its user at 19/10 23:52, 20/10 01:30 and 20/10 12:10. 14 SC units finished
  there and were copied to A (checksums verified); the 9 cut-off units were rerun from epoch 0 elsewhere.
- [Fact] The 20/10 cleanup deleted the `screen-ast` checkpoints that the demo app served at first; the app was switched
  to BEATs units, and on 23/10 to the BEATs + PAFA static-SC bundle.
- [Fact] Inference arms, 21/10: the scheduler appended `--num_workers`, which `lodo_ttsc` does not take (fixed by ending
  each job line with `&& :`); 20 QLung units failed at model build (`class_nums` unset at inference) and were rerun.
- [Fact] Machine E rebooted on 22/10 about 14:40: one PAFA + PC-MCL unit cut off at epoch 54 was set aside
  (`lodo-v1-interrupted/...seed2.reboot-2210`) and rerun from epoch 0.
- [Fact] 26/10: the source-audio service failed repeatedly because `PYTHONPATH` was missing; GPU training had finished,
  the audit and report were delayed until the import path was fixed.
- [Fact] 26/10: the old E results were recovered to B first (3,608 files, 63.9 GB, 94/94 completed-unit receipt sets
  valid, 10 old final pipelines match canonical digests); partial files and interrupted units were preserved, not resumed.
- [Fact] 27/10: the `refactor` of `src/training` (`patchmix_cl/` -> `methods/`, tests split into subfolders) left two tests
  failing (`BeatsEncoder.multiscale` unset on instances built without `__init__`); fixed with class-level defaults. Two
  deleted test files (`test_grid_rerun.py`, `test_grid_audit.py`) were restored under `tests/infrastructure/`.
- [Fact] 27/10: on E the frozen snapshot `build/campaigns/next-improvements-20261026/scripts/improvement_campaign.py` no
  longer matches its manifest digest (file time 26/10 22:35, freeze 26/10 17:58), so the service
  `auscult-next-improvements-20261026-screen-E` restarts in a loop with an integrity error. That 3-fold programme is
  superseded by the five-fold run; the 3-fold reports used in section 17.5 were read from B's copy. All `auscult-*` campaign services on A, B, D, E were stopped and disabled on 27/10; only the inference API on A runs.
- [Fact] 0 failed training units and 0 NaN or collapsed runs in every LODO, SC and PC-MCL list.

### 4.3 Where the results are (27/10)

- [Fact] Historical runs are synced to A at `outputs/train/<run>/<cell>/seed<s>/`: `grid-v1` (350 units), `lodo-v1`
  (1,400: 220 no SC, 880 inference arms, 175 SC static, 125 stage 8), `pcmcl-v1` (10), `lodo-ast` (260), `lodo-bench`
  (95), `sc-dynamic-completion-20261024` (100), `improvement-final-20261025` (50), `distill-pilot-20261024` (6), the
  frozen evaluation `final-pafa-static` and smoke runs (all counted by `report.json` on 27/10).
- [Fact] Stage-12 campaigns keep an immutable source snapshot, outputs and checkpoints per programme under
  `build/campaigns/<programme>/` (A and B; B is canonical). The 150 five-fold fits have outputs at
  `build/campaigns/fivefold-20261026/outputs/` (150 completion receipts) and 150 `best.pth` in `checkpoints-synced-to-A`.
  The anchor final fits (25 units, cell `gain6-cv66`) are only on B under `build/campaigns/followup-20261026/`.
- [Fact] Since 27/10 17:40 `outputs/`, `checkpoints/` and `build/campaigns/<programme>/{outputs,checkpoints,archive}` are symlinks to `/storage/hdd/disk0/namdp15/auscult-trust-A/` (14 directories, 209 GB, checksum-verified before the SSD copy was deleted; log `logs/hdd-move-20261027/`). Per-machine mirrors with a file inventory are on A's HDD:
  `manifest.json` lists B 7,268 files / 143.9 GB, D 925 / 40.8 GB, E 3,110 / 60.3 GB, F 147 / 1.8 GB, 0 source files
  deleted. An 18-file SHA-256 spot check (6 random checkpoints each from the B, D and E mirrors) matched the source machines.
- [Fact] Checkpoint coverage, joined from each run's `train_args.json` to a `.pth` (not `last.pth`) on A or in a mirror on 27/10:

| Run | Trained units | Checkpoint on A | In a mirror only | Missing |
| --- | ---: | ---: | --- | ---: |
| grid-v1 | 350 | 350 | | 0 |
| lodo-v1 | 520 | 201 | B 154, E 112, D 53 | 0 |
| pcmcl-v1 | 10 | 10 | | 0 |
| lodo-ast | 80 | 80 | | 0 |
| lodo-bench | 100 | 95 | | 5 (`dass_baseline` LODO units with only `train_args.json`: started, never finished; outside the 95 reported) |
| sc-dynamic-completion-20261024 | 100 | 10 | B 30, D 20, E 40 | 0 |
| improvement-final-20261025 | 50 | 50 | | 0 |
| distill-pilot-20261024 | 6 | 0 | D 6 | 0 |
| grid-v1-rerun / -resumed | 13 / 13 | 0 / 0 | | the 13 reruns were swapped into grid-v1 (counted there); the 13 resumed copies were set aside and their checkpoints are not kept |
| lung-srad | 7 | 0 | | reference row only; checkpoints deleted in the 20/10 cleanup |

  The 880 inference-arm units of lodo-v1 have no checkpoint of their own (they reuse the no-SC units). Hashes of the
  mirror are per machine, not per file: `manifest.json` lists counts and bytes only.
- Regenerate the tables: `bash scripts/lodo_h2_all.sh`, `PYTHONPATH=. .venv-train/bin/python scripts/lodo_tables.py`,
  `scripts/ensemble.py`, `scripts/ensemble_sesp.py`, `PYTHONPATH=. .venv-train/bin/python scripts/grid.py table last`.

## 5. Stage 0: Phase 0 (before v9)

Frozen-encoder ladders, fine-tuning ladders, Part II shift study and the Stage-1 screening of device variants (P0-P8)
are in `docs/reports/2026-10-07-phase0-report.md`. Main findings: the early weak base model was the encoder;
fine-tuning did not raise decodability or the coverage deficit (F4); no device variant separated from the baseline in
3-fold CV.

## 6. Stage 1: Registered core (v9, AST), from `docs/reports/2026-10-09-registered-core-report.md`

| Item | Result |
| --- | --- |
| Patch-Mix CL reproduction (AST, best epoch on test) | 61.49 ± 0.91 vs 62.37 ± 0.61 published; last epoch 59.59 ± 0.47 |
| Refit at the CV epoch (seeds 0-4) | AuscultTrust (SC) 58.31 ± 1.08; baseline 56.80 ± 1.49; SC + random gain 59.72 (not selected by the registered rule) |
| Gate G (10 seeds) | Paired difference +1.91, 95% CI [+0.31, +3.50]: passes (lower bound > -1.5). Superiority not shown (patient bootstrap [-0.83, +4.84]) |
| LODO on AST (H2) | Only Litt3200 passes the null gate. Prior-matched coverage 0.670 (baseline) vs 0.904 (SC); D = +0.177 [+0.055, +0.326]; set size 1.92 vs 2.67 labels. Freq-MixStyle D = +0.126 [-0.006, +0.269] |
| KAUH control A | Test-time SC on baseline weights lowers the cross-filter flip rate 0.188 to 0.104 (+0.085 [+0.041, +0.132]); AuscultTrust weights do not (0.175). Control B2 not interpretable (no filter pair beyond the chance range) |
| H1, H3 | [Pending]: phantom recordings |

[Interpretation] H2 failed only on set size, so the LODO v1 analysis reads set size next to abs(Delta), and ensembles
(sharper softmax) may shrink sets.

## 7. Stage 2: Grid v1, method x encoder on the official split

### 7.1 Setup

- 70 cells x 5 seeds = 350 units, finished 18/10 23:49. Audit (`scripts/grid.py audit`): 0 problems; SHA-256 of the
  split file and every checkpoint in `outputs/train/grid-v1/provenance_grid-v1-final-{A,B}.json`.
- Encoders AST, HTS-AT, BEATs, DASS, OPERA-CT. Fine-tuned (ft) methods: CE, Patch-Mix, Patch-Mix CL, PAFA, SG-SCL,
  QLung, ADD-RSC, ce_pafa_cfg (CE with PAFA's settings: batch 32, no SpecAugment, 5 s cycles on AST/BEATs, BEATs 100
  epochs). Frozen: the same minus Patch-Mix CL and SG-SCL. Each method with its reference settings
  (`experiments/train/grid/<method>.yaml`). Lung-SRAD is a reference row only.
- Last epoch = primary. Best epoch on test = optimistic, for comparison with papers only.
- Columns (%): Sp = normal recall; Se = recall of crackle, wheeze, both; HS = harmonic mean of Sp and Se; 2c = normal
  vs abnormal view of the same predictions (2c Sp = Sp); worst-dev = lowest per-device value; Cov = LAC split-conformal
  coverage at 90% over 20 device-stratified calibration/evaluation halves of the test patients; set size = mean labels
  per set (of 4).

### 7.2 Best cell per encoder, last epoch

| Encoder | Cell | Sp | Se | Score | HS | 2c Se | 2c Score | 2c HS | Worst-dev Score | Cov | Worst-dev Cov | Set size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BEATs | ft pafa | 73.2 ± 2.7 | 50.2 ± 2.9 | **61.7 ± 0.7** | 59.5 ± 1.3 | 65.4 ± 3.8 | 69.3 ± 1.1 | 69.0 ± 1.4 | 58.9 ± 0.5 | 89.8 ± 0.3 | 85.4 ± 1.0 | 2.47 ± 0.16 |
| HTS-AT | ft sg_scl | 72.4 ± 2.2 | 46.9 ± 1.2 | 59.6 ± 0.5 | 56.9 ± 0.2 | 62.3 ± 1.5 | 67.3 ± 0.6 | 66.9 ± 0.5 | 49.9 ± 2.8 | 90.1 ± 0.1 | 88.4 ± 0.3 | 2.38 ± 0.05 |
| DASS | ft patchmix | 70.6 ± 2.8 | 48.8 ± 1.2 | 59.7 ± 0.9 | 57.7 ± 0.4 | 61.8 ± 1.9 | 66.2 ± 0.8 | 65.9 ± 0.7 | 49.8 ± 3.4 | 89.7 ± 0.3 | 87.6 ± 0.3 | 2.15 ± 0.06 |
| AST | ft ce_pafa_cfg | 71.8 ± 2.4 | 46.2 ± 1.3 | 59.0 ± 1.0 | 56.2 ± 0.9 | 62.0 ± 1.7 | 66.9 ± 0.7 | 66.5 ± 0.7 | 48.5 ± 2.7 | 89.7 ± 0.2 | 86.6 ± 1.1 | 2.29 ± 0.04 |
| OPERA-CT | ft patchmix_cl | 65.4 ± 0.9 | 38.0 ± 1.4 | 51.7 ± 0.7 | 48.0 ± 1.1 | 53.8 ± 1.3 | 59.6 ± 0.6 | 59.0 ± 0.7 | 35.3 ± 0.9 | 90.1 ± 0.2 | 72.1 ± 2.2 | 2.83 ± 0.04 |
| AST | frozen add_rsc | 70.7 ± 2.4 | 43.7 ± 1.1 | 57.2 ± 1.5 | 54.0 ± 1.3 | 59.5 ± 1.0 | 65.1 ± 1.6 | 64.6 ± 1.4 | 47.4 ± 2.4 | 88.9 ± 0.3 | 84.0 ± 0.9 | 2.45 ± 0.04 |
| DASS | frozen add_rsc | 67.2 ± 1.5 | 45.3 ± 1.6 | 56.3 ± 0.5 | 54.1 ± 0.8 | 61.6 ± 2.3 | 64.4 ± 0.9 | 64.2 ± 1.0 | 48.5 ± 2.3 | 89.3 ± 0.3 | 81.9 ± 2.3 | 2.44 ± 0.04 |
| HTS-AT | frozen pafa | 63.1 ± 0.4 | 42.8 ± 0.2 | 52.9 ± 0.3 | 51.0 ± 0.3 | 60.0 ± 0.4 | 61.6 ± 0.3 | 61.5 ± 0.3 | 38.5 ± 1.5 | 90.0 ± 0.1 | 67.5 ± 4.5 | 2.66 ± 0.07 |
| BEATs | frozen patchmix | 67.3 ± 1.4 | 36.3 ± 1.2 | 51.8 ± 0.3 | 47.2 ± 0.7 | 50.7 ± 1.8 | 59.0 ± 0.4 | 57.8 ± 0.7 | 45.8 ± 1.0 | 89.6 ± 0.1 | 86.4 ± 0.5 | 2.22 ± 0.02 |
| OPERA-CT | frozen patchmix | 79.6 ± 2.3 | 18.6 ± 1.8 | 49.1 ± 0.5 | 30.1 ± 2.2 | 25.7 ± 2.0 | 52.6 ± 0.5 | 38.7 ± 2.1 | 43.1 ± 1.3 | 89.5 ± 0.1 | 87.5 ± 0.1 | 2.50 ± 0.02 |

### 7.3 Top 5, best epoch on test (optimistic)

| Rank | Cell | Sp | Se | Score | HS | 2c Se | 2c Score | 2c HS | Worst-dev Score | Cov | Worst-dev Cov | Set size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | beats+ft+patchmix_cl | 76.4 ± 3.8 | 50.9 ± 3.4 | 63.6 ± 0.8 | 60.9 ± 1.4 | 65.5 ± 3.2 | 70.9 ± 1.0 | 70.4 ± 0.9 | 57.8 ± 3.8 | 89.2 ± 0.3 | 86.2 ± 0.7 | 1.95 ± 0.12 |
| 2 | beats+ft+sg_scl | 78.6 ± 4.5 | 47.9 ± 3.3 | 63.2 ± 0.7 | 59.3 ± 1.3 | 62.4 ± 3.5 | 70.5 ± 0.7 | 69.4 ± 0.5 | 56.2 ± 2.9 | 89.5 ± 0.4 | 87.0 ± 0.4 | 1.99 ± 0.09 |
| 3 | beats+ft+pafa | 75.5 ± 2.9 | 51.0 ± 2.8 | 63.2 ± 0.6 | 60.8 ± 1.2 | 65.8 ± 3.2 | 70.6 ± 0.4 | 70.2 ± 0.6 | 60.0 ± 0.9 | 89.7 ± 0.2 | 86.0 ± 0.8 | 2.16 ± 0.28 |
| 4 | beats+ft+ce_pafa_cfg | 76.1 ± 3.2 | 50.0 ± 2.8 | 63.1 ± 1.1 | 60.3 ± 1.4 | 64.6 ± 3.6 | 70.3 ± 0.9 | 69.7 ± 1.1 | 58.5 ± 2.7 | 89.7 ± 0.1 | 86.3 ± 0.8 | 2.11 ± 0.26 |
| 5 | beats+ft+qlung | 77.3 ± 3.3 | 48.1 ± 2.5 | 62.7 ± 0.5 | 59.2 ± 0.9 | 62.3 ± 2.5 | 69.8 ± 0.5 | 68.9 ± 0.4 | 55.9 ± 3.2 | 89.4 ± 0.3 | 86.3 ± 1.3 | 1.98 ± 0.07 |
| (7) | beats+ft+ce | 74.9 ± 4.3 | 49.4 ± 2.3 | 62.2 ± 1.1 | 59.4 ± 0.8 | 65.2 ± 2.8 | 70.1 ± 0.8 | 69.6 ± 0.4 | 53.7 ± 3.2 | 89.7 ± 0.2 | 86.2 ± 1.0 | 2.10 ± 0.21 |
| best non-BEATs | htsat+ft+patchmix_cl | 77.0 ± 4.6 | 47.1 ± 3.5 | 62.1 ± 1.5 | 58.3 ± 1.9 | 60.9 ± 3.4 | 69.0 ± 1.4 | 67.8 ± 1.4 | 51.5 ± 5.4 | 89.6 ± 0.2 | 86.5 ± 0.6 | 2.01 ± 0.06 |

[Fact] The top 5 span 1.0 Score, below one seed SD. Best per encoder on the best epoch: BEATs Patch-Mix CL 63.6, HTS-AT
Patch-Mix CL 62.1, DASS Patch-Mix 61.7, AST CE 61.3, OPERA-CT CE 52.6. Best-epoch selection adds 0.6-4.4 Score over the
last epoch. [Interpretation] Both epoch rules put BEATs first; the method order inside BEATs is within noise.

### 7.4 Cells carried into LODO v1 and the trained-SC arms (last epoch)

LODO without SC: all 11 rows (stage 3). Trained SC: rows marked SC (stage 4).

| Cell | SC | Sp | Se | Score | HS | 2c Se | 2c Score | 2c HS | Worst-dev Score | Cov | Worst-dev Cov | Set size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ast+ft+ce | | 68.4 ± 2.3 | 46.8 ± 2.5 | 57.6 ± 1.0 | 55.5 ± 1.4 | 64.5 ± 3.6 | 66.5 ± 1.2 | 66.3 ± 1.3 | 44.7 ± 0.9 | 89.5 ± 0.2 | 86.7 ± 0.4 | 2.35 ± 0.05 |
| ast+ft+patchmix_cl | SC | 71.7 ± 4.1 | 44.7 ± 1.7 | 58.2 ± 1.3 | 55.0 ± 0.5 | 60.7 ± 2.5 | 66.2 ± 0.9 | 65.7 ± 0.4 | 47.1 ± 3.2 | 89.3 ± 0.3 | 86.9 ± 0.4 | 2.25 ± 0.03 |
| ast+ft+pafa | SC | 69.3 ± 1.0 | 47.4 ± 1.2 | 58.3 ± 0.9 | 56.3 ± 1.0 | 63.2 ± 1.7 | 66.2 ± 1.1 | 66.1 ± 1.1 | 47.4 ± 1.9 | 89.7 ± 0.2 | 86.1 ± 0.7 | 2.34 ± 0.05 |
| htsat+ft+ce | | 70.0 ± 3.0 | 47.6 ± 0.8 | 58.8 ± 1.4 | 56.6 ± 0.9 | 62.4 ± 1.8 | 66.2 ± 1.2 | 65.9 ± 1.1 | 46.9 ± 3.7 | 90.0 ± 0.2 | 87.1 ± 1.2 | 2.23 ± 0.03 |
| htsat+ft+patchmix_cl | SC | 69.9 ± 2.7 | 48.3 ± 1.2 | 59.1 ± 1.0 | 57.1 ± 0.5 | 63.4 ± 1.9 | 66.7 ± 0.8 | 66.5 ± 0.7 | 46.5 ± 5.1 | 89.8 ± 0.1 | 87.6 ± 0.7 | 2.21 ± 0.02 |
| htsat+ft+sg_scl | SC | 72.4 ± 2.2 | 46.9 ± 1.2 | 59.6 ± 0.5 | 56.9 ± 0.2 | 62.3 ± 1.5 | 67.3 ± 0.6 | 66.9 ± 0.5 | 49.9 ± 2.8 | 90.1 ± 0.1 | 88.4 ± 0.3 | 2.38 ± 0.05 |
| beats+ft+ce | | 65.7 ± 3.7 | 52.7 ± 1.3 | 59.2 ± 1.5 | 58.4 ± 1.1 | 69.7 ± 1.4 | 67.7 ± 1.5 | 67.6 ± 1.5 | 50.4 ± 3.0 | 89.4 ± 0.4 | 86.6 ± 0.4 | 2.38 ± 0.06 |
| beats+ft+patchmix_cl | SC | 64.2 ± 2.6 | 55.2 ± 1.8 | 59.7 ± 1.3 | 59.3 ± 1.2 | 72.4 ± 1.5 | 68.3 ± 0.9 | 68.0 ± 1.0 | 52.1 ± 3.5 | 89.3 ± 0.2 | 86.3 ± 0.2 | 2.38 ± 0.08 |
| beats+ft+pafa | SC | 73.2 ± 2.7 | 50.2 ± 2.9 | 61.7 ± 0.7 | 59.5 ± 1.3 | 65.4 ± 3.8 | 69.3 ± 1.1 | 69.0 ± 1.4 | 58.9 ± 0.5 | 89.8 ± 0.3 | 85.4 ± 1.0 | 2.47 ± 0.16 |
| beats+ft+sg_scl | SC | 71.9 ± 4.1 | 50.2 ± 1.5 | 61.1 ± 1.3 | 59.1 ± 0.5 | 65.8 ± 2.3 | 68.9 ± 1.0 | 68.6 ± 0.8 | 52.9 ± 6.4 | 89.7 ± 0.1 | 86.6 ± 0.6 | 2.59 ± 0.21 |
| beats+ft+qlung | | 67.4 ± 2.1 | 51.2 ± 1.4 | 59.3 ± 0.8 | 58.2 ± 0.7 | 67.5 ± 1.6 | 67.4 ± 0.7 | 67.4 ± 0.7 | 51.8 ± 6.9 | 89.6 ± 0.3 | 86.9 ± 0.4 | 2.40 ± 0.05 |

### 7.5 Method effects (paired difference to CE on the same encoder and seed, last epoch)

Only the rows with abs(mean) > SD on Score, plus the LODO cells; full table in `docs/reports/2026-10-19-grid-v1-results.md`.

| Encoder | Method | ΔSp | ΔSe | ΔScore | ΔHS | Δ2c Score | ΔWorst-dev Score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BEATs | pafa | +7.5 ± 2.9 | -2.4 ± 2.4 | **+2.6 ± 1.4** | +1.1 ± 1.6 | +1.6 ± 1.8 | **+8.5 ± 2.9** |
| BEATs | sg_scl | +6.3 ± 7.2 | -2.4 ± 2.2 | +1.9 ± 2.7 | +0.7 ± 1.5 | +1.2 ± 2.3 | +2.5 ± 8.7 |
| BEATs | patchmix_cl | -1.5 ± 5.1 | +2.5 ± 2.9 | +0.5 ± 2.0 | +0.9 ± 1.7 | +0.6 ± 1.6 | +1.7 ± 6.0 |
| BEATs | qlung | +1.7 ± 3.8 | -1.4 ± 2.5 | +0.1 ± 1.3 | -0.2 ± 1.1 | -0.3 ± 1.2 | +1.4 ± 4.7 |
| AST | pafa | +0.9 ± 1.4 | +0.6 ± 3.2 | +0.7 ± 1.3 | +0.7 ± 2.0 | -0.2 ± 1.9 | +2.6 ± 2.2 |
| AST | patchmix_cl | +3.3 ± 6.2 | -2.1 ± 3.6 | +0.6 ± 2.0 | -0.5 ± 1.5 | -0.2 ± 1.7 | +2.3 ± 3.4 |
| AST | ce_pafa_cfg | +3.4 ± 2.4 | -0.7 ± 1.4 | **+1.4 ± 1.2** | +0.6 ± 1.0 | +0.5 ± 1.4 | +3.7 ± 3.1 |
| HTS-AT | patchmix_cl | -0.1 ± 1.4 | +0.7 ± 1.0 | +0.3 ± 0.5 | +0.4 ± 0.5 | +0.5 ± 0.8 | -0.4 ± 2.1 |
| HTS-AT | sg_scl | +2.4 ± 2.8 | -0.7 ± 0.6 | +0.8 ± 1.3 | +0.2 ± 0.9 | +1.1 ± 1.7 | +3.0 ± 4.0 |
| HTS-AT | pafa | -3.0 ± 2.4 | -1.4 ± 1.2 | **-2.2 ± 1.3** | -2.0 ± 1.2 | -1.4 ± 1.1 | -0.7 ± 4.9 |
| HTS-AT | ce_pafa_cfg | -3.0 ± 3.6 | -2.1 ± 3.2 | **-2.5 ± 1.1** | -2.6 ± 1.7 | -1.7 ± 1.0 | -5.2 ± 2.5 |
| DASS | patchmix | +2.5 ± 2.9 | +0.8 ± 1.6 | **+1.7 ± 1.1** | +1.4 ± 0.8 | +0.7 ± 0.4 | +3.8 ± 4.9 |
| DASS | pafa | +0.5 ± 3.3 | +1.8 ± 2.3 | **+1.2 ± 0.6** | +1.4 ± 0.6 | +1.3 ± 0.7 | +6.2 ± 2.7 |
| DASS | sg_scl | +1.9 ± 2.6 | +0.6 ± 2.0 | **+1.2 ± 1.1** | +1.0 ± 1.1 | +0.8 ± 1.0 | +2.5 ± 2.3 |
| OPERA-CT | pafa | -3.9 ± 3.2 | -0.4 ± 1.2 | **-2.2 ± 1.2** | -1.4 ± 0.7 | -1.2 ± 1.7 | +3.4 ± 1.9 |

- [Fact] The encoder matters more than the method: fine-tuned Score BEATs 58.9-61.7, HTS-AT 56.3-59.6, DASS 57.5-59.7,
  AST 56.9-59.0, OPERA-CT 49.3-51.7. Most methods gain less than one seed SD over CE.
- [Fact] Fine-tuning beats freezing (CE): AST +2.5, DASS +3.7, OPERA-CT +4.7, HTS-AT +7.0, BEATs +7.7.
- [Fact] PAFA on BEATs is the one large, consistent gain, mostly in Sp and in the worst device (+8.5). PAFA's losses vs
  its settings (pafa - ce_pafa_cfg): BEATs +1.6 ± 1.8, DASS +1.6 ± 0.4, AST -0.7 ± 0.8.
- [Fact] Frozen PAFA equals frozen ce_pafa_cfg within 0.2 Score on every encoder: its losses train a projector the
  classifier does not use. Frozen OPERA-CT nearly ignores abnormal sounds (Se 19-25%).

### 7.6 Per-class recall and ensembles (BEATs, last epoch)

| Cell | Normal (= Sp) | Crackle | Wheeze | Both | Se | Score |
| --- | --- | --- | --- | --- | --- | --- |
| beats+ft+ce | 65.7 | 62.3 | 42.6 | 35.8 | 52.7 | 59.2 |
| beats+ft+pafa | 73.2 | 60.4 | 38.7 | 35.1 | 50.2 | 61.7 |
| beats+ft+sg_scl | 71.9 | 60.6 | 39.0 | 33.6 | 50.2 | 61.1 |
| beats+ft+patchmix_cl | 64.2 | 64.5 | 45.8 | 38.2 | 55.2 | 59.7 |
| beats+ft+ce_pafa_cfg | 66.4 | 65.5 | 42.9 | 30.8 | 53.9 | 60.1 |
| beats+ft+qlung | 67.4 | 61.0 | 40.9 | 35.0 | 51.2 | 59.3 |
| beats+ft+add_rsc | 70.9 | 59.9 | 31.6 | 28.8 | 46.9 | 58.9 |

Ensembles = mean softmax at the last epoch, same seed. [test-picked] = members chosen after seeing test results.

| Ensemble (members) | Sp | Se | Score | HS | 2c Se | 2c Score | 2c HS | Worst-dev Score | Cov | Worst-dev Cov | Set size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| beats+ft+pafa alone (reference) | 73.2 ± 2.7 | 50.2 ± 2.9 | 61.7 ± 0.7 | 59.5 ± 1.3 | 65.4 ± 3.8 | 69.3 ± 1.1 | 69.0 ± 1.4 | 58.9 ± 0.5 | 89.8 ± 0.3 | 85.4 ± 1.0 | 2.47 ± 0.16 |
| 8 fine-tuned methods on BEATs (8) | 70.6 ± 1.4 | 53.3 ± 1.2 | 62.0 ± 0.2 | 60.8 ± 0.3 | 68.6 ± 1.3 | 69.6 ± 0.2 | 69.6 ± 0.2 | 55.2 ± 1.2 | 89.3 ± 0.2 | 86.3 ± 0.4 | 1.95 ± 0.03 |
| CE on AST, HTS-AT, BEATs, DASS (4) | 70.4 ± 2.1 | 50.4 ± 1.0 | 60.4 ± 0.8 | 58.7 ± 0.5 | 64.6 ± 1.5 | 67.5 ± 0.6 | 67.4 ± 0.6 | 48.6 ± 2.6 | 89.6 ± 0.2 | 87.0 ± 0.6 | 1.94 ± 0.01 |
| 8 methods x 4 encoders (32) | 72.8 ± 1.3 | 50.6 ± 1.3 | 61.7 ± 0.4 | 59.7 ± 0.5 | 64.0 ± 1.1 | 68.4 ± 0.2 | 68.1 ± 0.2 | 50.9 ± 0.7 | 89.7 ± 0.1 | 85.9 ± 0.2 | 1.81 ± 0.01 |
| [test-picked] LODO set (11) | 74.2 ± 1.5 | 51.7 ± 0.5 | 63.0 ± 0.7 | 60.9 ± 0.4 | 65.9 ± 1.0 | 70.1 ± 0.3 | 69.8 ± 0.1 | 55.8 ± 2.0 | 89.5 ± 0.1 | 86.1 ± 0.2 | 1.86 ± 0.01 |
| [test-picked] LODO BEATs cells (5) | 71.3 ± 1.7 | 53.4 ± 1.3 | 62.4 ± 0.4 | 61.1 ± 0.4 | 68.8 ± 1.3 | 70.1 ± 0.3 | 70.0 ± 0.3 | 57.0 ± 2.2 | 89.5 ± 0.2 | 86.4 ± 0.3 | 2.08 ± 0.03 |

- [Fact] PAFA and SG-SCL raise Sp; Patch-Mix CL raises all three abnormal recalls. Two BEATs methods disagree on 18-28%
  of test cycles (seed 0). The 8-method BEATs ensemble picks no cell on test: +2.8 Score over BEATs CE, seed SD 1.5 to
  0.2, and sets of 1.95 labels instead of 2.47; it does not beat PAFA alone on the worst device. Averaging the last 10
  epochs of one model changes Score by 0.0-0.2: the ensemble gain comes from different models.
- [Note] The LODO-set ensemble above includes ce_pafa_cfg; LODO v1 runs BEATs CE in its place (user decision 19/10), so
  the registered E1 has the same 11-cell size with BEATs CE.

### 7.7 Coverage, published numbers, incidents

- [Fact] Pooled coverage is at nominal (88.2-90.5%) in every cell; the worst device (Littmann 3200 in 343 of 350 units)
  falls to 85-88% in most fine-tuned cells and 67-82% in many frozen cells.
- [Fact] AST best epoch on test vs published: Patch-Mix CL 61.1 ± 0.4 vs 62.37; QLung 61.0 ± 0.9 vs 62.01; ADD-RSC
  59.6 ± 0.5 vs 65.53 (the reference resizes AST input to 1024 x 256 with its own AST; here the denoising layer runs on
  each encoder's own input). Published BEATs PC-MCL: 65.37 ± 0.73 (best epoch on test; ours in stage 5, section 10).
- Incidents: QLung's reference quality score is NaN on 71 AKGC417L cycles (leading silence), our port skips silent
  frames; SG-SCL's reference reversal schedule collapsed BEATs in 3 of 4 seeds, all SG-SCL cells use the DANN ramp;
  power cut 17/10 10:43 (9.6 h), 13 resumed units rerun cleanly and swapped in (resumed vs rerun Score: mean abs 0.61,
  largest 2.5).

### 7.8 Findings (stage 2)

- [Fact] Encoder choice dominates method choice: fine-tuned Score BEATs 58.9-61.7, HTS-AT 56.3-59.6, DASS 57.5-59.7,
  AST 56.9-59.0, OPERA-CT 49.3-51.7; most methods differ from CE on the same encoder by less than one seed SD.
- [Fact] BEATs PAFA is the one large consistent method gain (+2.6 Score, +8.5 worst-device Score), mostly through Sp;
  PAFA and SG-SCL push Sp up and Se down, Patch-Mix CL the other way.
- [Fact] Averaging methods helps more than any single method: 8 BEATs methods 62.0 ± 0.2 (SD 0.2 vs 0.7-1.5 for single
  cells) with smaller sets (1.95 vs 2.47 labels).
- [Fact] Test-selected epochs inflate Score by 0.6-4.4 points.
- [Fact] Littmann 3200 is the worst device in 343 of 350 units; pooled coverage stays at nominal while the worst device
  drops to 85-88% (fine-tuned) and 67-82% (frozen).

## 8. Stage 3: LODO v1 without SC, from `docs/reports/2026-10-21-lodo-v1-results.md`

220 units (11 cells x 4 held-out devices x 5 seeds), fixed last epoch; test = every cycle of the held-out device;
conformal LAC at 90% calibrated on the calibration patients of the source devices; Delta = 0.90 minus prior-matched
coverage.

### 8.1 Score and coverage per held-out device (5 seeds)

| Cell | Meditron | LittC2SE | Litt3200 | AKGC417L | Macro Score | Macro w/o AKGC417L | Litt3200 Cov / set | AKGC417L Cov / set |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ast+ft+ce | 56.2 | 65.1 | 50.2 | 53.7 | 56.3 ± 1.7 | 57.2 ± 2.5 | 96.3 / 3.04 | 78.1 / 2.40 |
| ast+ft+pafa | 57.0 | 64.5 | 50.5 | 56.1 | 57.0 ± 1.0 | 57.3 ± 1.2 | 97.6 / 3.22 | 74.5 / 1.92 |
| ast+ft+patchmix_cl | 55.6 | 63.0 | 52.8 | 54.0 | 56.3 ± 1.2 | 57.1 ± 1.9 | 95.3 / 2.97 | 85.2 / 2.81 |
| htsat+ft+ce | 58.1 | 63.0 | 47.3 | 55.9 | 56.1 ± 1.7 | 56.1 ± 1.9 | 88.9 / 2.67 | 77.9 / 2.13 |
| htsat+ft+patchmix_cl | 58.3 | 63.4 | 49.6 | 56.1 | 56.9 ± 2.0 | 57.1 ± 2.2 | 91.2 / 2.60 | 74.1 / 1.90 |
| htsat+ft+sg_scl | 58.2 | 63.7 | 48.8 | 55.7 | 56.6 ± 1.5 | 56.9 ± 1.7 | 89.7 / 2.87 | 74.6 / 1.89 |
| beats+ft+ce | 57.6 | 64.4 | 56.3 | 57.9 | 59.1 ± 1.3 | 59.5 ± 1.8 | 99.5 / 3.54 | 81.1 / 2.35 |
| beats+ft+patchmix_cl | 57.7 | 66.4 | 57.5 | 58.8 | 60.1 ± 1.7 | 60.5 ± 2.6 | 99.4 / 3.54 | 84.0 / 2.36 |
| beats+ft+sg_scl | 58.4 | 64.8 | 54.9 | 57.9 | 59.0 ± 1.7 | 59.4 ± 2.3 | 94.9 / 2.80 | 79.3 / 2.05 |
| beats+ft+pafa | 57.2 | 62.0 | 59.7 | 56.4 | 58.8 ± 1.4 | 59.6 ± 1.6 | 96.6 / 2.76 | 80.9 / 2.26 |
| beats+ft+qlung | 58.5 | 65.1 | 57.2 | 57.8 | 59.6 ± 0.8 | 60.3 ± 0.6 | 98.8 / 3.37 | 85.2 / 2.46 |

Sp, Se and HS per fold, and coverage on every fold, are in the detailed report. [Fact] Litt3200 is the hardest fold
for every cell; Sp falls to 33-63% there except for BEATs PAFA (88.5%). On AKGC417L raw coverage is 74-85% but the
prior-matched |Delta| is 0.03-0.06: the shortfall is class mix.

### 8.2 Method vs CE on the same encoder (paired by seed)

| Encoder | Method | dScore macro | dScore w/o AKGC417L | dScore Litt3200 |
| --- | --- | --- | --- | --- |
| AST | pafa | +0.7 ± 2.3 | +0.2 ± 3.2 | +0.3 ± 7.5 |
| AST | patchmix_cl | +0.0 ± 1.7 | -0.0 ± 1.7 | +2.6 ± 4.4 |
| HTS-AT | patchmix_cl | +0.8 ± 1.5 | +1.0 ± 1.9 | +2.3 ± 4.4 |
| HTS-AT | sg_scl | +0.5 ± 1.4 | +0.8 ± 1.8 | +1.5 ± 3.4 |
| BEATs | patchmix_cl | +1.0 ± 1.4 | +1.1 ± 1.8 | +1.2 ± 3.8 |
| BEATs | sg_scl | -0.1 ± 1.5 | -0.1 ± 2.0 | -1.4 ± 4.4 |
| BEATs | pafa | -0.3 ± 1.5 | +0.1 ± 1.3 | +3.3 ± 2.9 |
| BEATs | qlung | +0.6 ± 0.9 | +0.8 ± 1.5 | +0.8 ± 6.9 |

### 8.3 Findings (stage 3)

- [Fact] The encoder order holds under device shift: macro Score BEATs 58.8-60.1, AST 56.3-57.0, HTS-AT 56.1-56.9,
  about the official-split level for BEATs.
- [Fact] Litt3200 is the hardest held-out device for every cell (Score 47.3-59.7, Sp 33-63%); BEATs PAFA is best there
  (59.7, Sp 88.5%).
- [Fact] Methods vs CE stay within seed SD in LODO (macro -0.3 to +1.0); BEATs PAFA gains on Litt3200 (+3.3 ± 2.9).
- [Fact] Raw coverage on AKGC417L is 74-85%, but the prior-matched |Delta| is 0.03-0.06: the shortfall is class mix.

## 9. Stage 4: Trained SC, static bound

175 units: 7 cells x (official split + 4 held-out devices) x 5 seeds, paired by seed with the no-SC units.

### 9.1 Official split vs no SC (same cell and seed, last epoch)

| Cell | Sp | Se | Score | HS | 2c Score | Worst-dev Score | Cov | Worst-dev Cov | Set size | dScore vs no SC | dWorst-dev Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ast+ft+patchmix_cl | 70.8 ± 2.0 | 45.8 ± 1.3 | 58.3 ± 1.1 | 55.6 ± 1.0 | 66.1 ± 0.8 | 49.8 ± 3.0 | 89.4 ± 0.1 | 87.5 ± 0.3 | 2.25 ± 0.06 | +0.1 ± 1.3 | +2.8 ± 5.0 |
| ast+ft+pafa | 71.7 ± 2.3 | 46.4 ± 1.6 | 59.0 ± 0.7 | 56.3 ± 0.8 | 67.0 ± 0.6 | 50.3 ± 2.0 | 89.5 ± 0.2 | 85.5 ± 0.6 | 2.34 ± 0.02 | +0.7 ± 0.5 | +2.9 ± 2.5 |
| htsat+ft+patchmix_cl | 68.8 ± 3.3 | 48.1 ± 1.0 | 58.5 ± 1.2 | 56.6 ± 0.6 | 65.9 ± 0.8 | 46.3 ± 3.5 | 90.1 ± 0.3 | 87.3 ± 0.5 | 2.24 ± 0.08 | -0.6 ± 1.4 | -0.2 ± 6.1 |
| htsat+ft+sg_scl | 73.0 ± 3.0 | 45.7 ± 1.6 | 59.3 ± 1.0 | 56.2 ± 0.7 | 66.5 ± 0.9 | 49.7 ± 4.3 | 90.2 ± 0.3 | 87.7 ± 1.1 | 2.34 ± 0.03 | -0.3 ± 1.4 | -0.1 ± 5.8 |
| beats+ft+patchmix_cl | 67.4 ± 1.6 | 52.2 ± 1.3 | 59.8 ± 0.8 | 58.8 ± 0.8 | 68.3 ± 0.7 | 52.0 ± 2.6 | 89.2 ± 0.2 | 86.2 ± 0.2 | 2.30 ± 0.06 | +0.1 ± 1.2 | -0.1 ± 4.0 |
| beats+ft+pafa | 75.6 ± 3.1 | 48.9 ± 2.9 | **62.3 ± 0.9** | 59.3 ± 1.3 | 69.5 ± 0.7 | 58.9 ± 1.4 | 89.6 ± 0.3 | 85.6 ± 0.7 | 2.33 ± 0.13 | +0.5 ± 1.5 | -0.1 ± 1.2 |
| beats+ft+sg_scl | 72.0 ± 2.9 | 49.4 ± 1.1 | 60.7 ± 1.0 | 58.6 ± 0.6 | 68.5 ± 0.8 | 51.8 ± 5.5 | 89.8 ± 0.2 | 87.5 ± 0.2 | 2.43 ± 0.16 | -0.4 ± 1.6 | -1.1 ± 6.8 |

### 9.2 LODO vs no SC (paired by seed)

| Arm | dScore macro | dScore Litt3200 | Litt3200 coverage change | Set size change (macro) |
| --- | --- | --- | --- | --- |
| Trained SC, static bound (7 cells) | -0.0 to +1.5 (6 of 7 positive) | -0.4 to +2.9 | -1.4 to +1.7 | -0.29 to 0.00 |

Trained SC static per cell, macro dScore in LODO: AST PAFA +1.4 ± 0.7, AST Patch-Mix CL +0.3 ± 1.5, HTS-AT Patch-Mix CL
+1.2 ± 1.5, HTS-AT SG-SCL +1.5 ± 1.4, BEATs Patch-Mix CL -0.0 ± 1.4, BEATs PAFA +1.5 ± 1.2, BEATs SG-SCL +1.5 ± 1.1.

### 9.3 Findings (stage 4)

- [Fact] Trained SC costs nothing on the official split (-0.6 to +0.7 Score) and raises the LODO macro Score for 6 of 7
  cells (+1.2 to +1.5 for AST PAFA, HTS-AT both, BEATs PAFA and SG-SCL; +0.3 AST Patch-Mix CL; -0.0 BEATs Patch-Mix CL).
- [Fact] BEATs PAFA + trained SC reaches 62.3 ± 0.9 on the official split (best last-epoch Score so far) and about 60.3
  macro in LODO, 61.6 on Litt3200.
- [Fact] Coverage and set size barely move with trained SC (Litt3200 coverage -1.4 to +1.7 points, sets -0.29 to 0.00).

## 10. Stage 5: PC-MCL on BEATs (Amendment 5), from `docs/reports/2026-10-20-pcmcl-v1-results.md`

Official split, seeds 0-4, 100 epochs, run `pcmcl-v1` (machine D, 10 of 10 units, 0 failed).

| Cell | Epoch | Sp | Se | Score | HS | 2c Score | Worst-dev Score | Cov | Worst-dev Cov | Set size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| beats+ft+pcmcl | last | 69.3 ± 1.7 | 52.6 ± 1.5 | 60.9 ± 0.6 | 59.7 ± 0.7 | 69.3 ± 0.4 | 57.2 ± 2.6 | 89.2 ± 0.1 | 86.6 ± 0.4 | 1.95 ± 0.02 |
| beats+ft+pafa_pcmcl | last | 74.0 ± 1.7 | 48.6 ± 0.7 | 61.3 ± 1.0 | 58.6 ± 0.8 | 69.7 ± 0.9 | 58.8 ± 1.0 | 89.8 ± 0.2 | 86.8 ± 0.2 | 2.07 ± 0.05 |
| beats+ft+pafa (grid) | last | 73.2 ± 2.7 | 50.2 ± 2.9 | 61.7 ± 0.7 | 59.5 ± 1.3 | 69.3 ± 1.1 | 58.9 ± 0.5 | 89.8 ± 0.3 | 85.4 ± 1.0 | 2.47 ± 0.16 |
| beats+ft+pcmcl | best on test (optimistic) | 77.0 ± 2.8 | 51.7 ± 3.0 | 64.3 ± 1.0 | 61.8 ± 1.6 | 71.9 ± 1.4 | 58.8 ± 3.6 | 89.2 ± 0.2 | 85.5 ± 0.5 | 1.85 ± 0.03 |
| beats+ft+pafa_pcmcl | best on test (optimistic) | 80.2 ± 4.0 | 46.9 ± 2.3 | 63.6 ± 1.1 | 59.1 ± 1.1 | 71.0 ± 0.8 | 59.3 ± 2.6 | 89.4 ± 0.3 | 84.5 ± 1.7 | 2.02 ± 0.06 |

- [Fact] Paired by seed against BEATs PAFA at the last epoch: PC-MCL -0.8 ± 0.9, PAFA + PC-MCL -0.4 ± 1.1. The
  registered rule (last-epoch Score above PAFA by more than one seed SD) is not met by either cell.
- [Fact] PC-MCL's best epochs on test are 4-6 of 100 (64.3, published 65.37 ± 0.73 with the same test-selected rule); it
  loses about 3.4 Score by the last epoch. It gives the smallest conformal sets of the BEATs cells (1.95).
- [Fact] Following the user (21/10), PAFA + PC-MCL is now also trained with SC (static and dynamic bound) in stage 8.

## 11. Stage 6: Inference arms on the LODO units

880 units: four arms on each of the 220 no-SC LODO units (`src.evaluation.lodo_ttsc`), paired by seed with the source
unit; ranges over the 11 cells. Test-time SC corrects only the held-out device towards the source devices' mean spectrum,
from its unlabelled cycles. A2 maps the per-bin mean and SD of the held-out spectrograms onto the training statistics.
TTA-EQ averages the softmax over 8 random smooth per-bin gain views.

| Arm | dScore macro | dScore Litt3200 | Litt3200 coverage change | Set size change (macro) |
| --- | --- | --- | --- | --- |
| Test-time SC, static bound | -0.8 to +1.1 | AST +3.4 to +4.6, HTS-AT +2.1 to +2.8, BEATs +0.6 to +2.4 | +0.1 to +2.5 points | -0.06 to +0.12 |
| Test-time SC, dynamic bound | -1.4 to +1.1 | AST +4.1 to +5.3, HTS-AT +1.9 to +3.7, BEATs +0.7 to +2.2 | 0.0 to +2.9 | -0.05 to +0.19 |
| A2 | -6.1 to +0.8 (BEATs -4.0 to -6.1) | BEATs -4.8 to -11.6, HTS-AT +5.0 to +5.2 | -2.7 to +5.2 | -0.07 to +0.35 |
| TTA-EQ (8 views) | 0.0 to +0.5 | -0.4 to +0.6 | -1.1 to -4.6 | -0.07 to -0.32 |

### 11.1 Findings (stage 6)

- [Fact] Test-time SC raises Litt3200 Score for every cell (AST +3.4 to +5.3, HTS-AT +1.9 to +3.7, BEATs +0.6 to
  +2.4); the macro change is small (-1.4 to +1.1); coverage and set size barely move. Static and dynamic bounds are close.
- [Fact] A2 hurts every BEATs cell (macro -4.0 to -6.1; Litt3200 -4.8 to -11.6) and AST PAFA (-2.6), and helps HTS-AT on
  Litt3200 (+5). [Interpretation] Matching whole-device statistics removes the held-out device's class mix along with
  its spectrum.
- [Fact] TTA-EQ changes Score by 0.0 to +0.5 and shrinks sets by 0.07-0.32 labels (Litt3200 over-coverage down 1-5
  points).

## 12. Stage 7: Coverage under device shift (H2)

[Fact] For all 11 cells and all 4 held-out devices the baseline |Delta| stays below the 95th percentile of |Delta| under
exchangeable re-splits of the calibration patients (closest: BEATs Patch-Mix CL on Litt3200, 0.090 vs 0.091). No fold
is tested, so the H2 statistic is not computed for any arm. The registered AST LODO (models stopped at epoch 2) did
have an under-covered Litt3200 fold (0.670); the converged models of LODO v1 do not.

### 12.1 Findings (stage 7)

- [Fact] H2 cannot be tested in LODO v1: no fold of any cell has a baseline coverage error above chance, so no H2
  statistic is computed for test-time SC, trained SC, A2 or TTA-EQ.
- [Fact] Coverage on an unseen device stays near nominal through large sets on Litt3200 (2.6-3.5 of 4 labels) and class
  mix on AKGC417L.
- [Interpretation] A coverage claim for SC now needs the phantom data, earlier-epoch models or another dataset.

### 12.2 Finite-N and class-conditional conformal analyses (24/10, exploratory)

- [Fact] Registered E1-E4 predictions with per-device patient / seed conformal intervals (recalibration in every draw):
  BEATs-5 has the highest registered-group LODO macro Score (61.11 ± 1.43), but its paired interval for the gain over PAFA is
  [-0.13, +4.88]; no registered ensemble passes the seed-averaged fold-level coverage null gate; one-cycle `k rate`
  lowers specificity substantially and larger adaptation cohorts do better descriptively. SC stays full-pool there, so it
  is not a joint finite-N SC + `k rate` test (`docs/reports/2026-10-24-ensemble-finite-n.md`).
- [Fact] SC and finite-N update (880 existing LODO SC arm / group units): at N = 50 both static user ensembles have
  positive pointwise exploratory Score intervals versus PAFA, specificity preservation unproven; trained static SC gives the
  three-BEATs group +1.40 [-0.24, +3.03] versus no SC. On the selected PAFA / static model the source-device-robust threshold
  raises macro coverage to 0.915 with larger sets (2.484 vs 2.132), AKGC417L coverage stays 0.830
  (`docs/reports/2026-10-24-sc-finite-n-uncertainty.md`).
- [Fact] Class-conditional conformal on PAFA / static: AKGC417L coverage 0.700 -> 0.863 with set size 1.890 -> 2.389; crackle
  inclusion improves, wheeze and combined-class shortfalls persist (`docs/reports/2026-10-24-class-conditional-conformal.md`).
  None of these is a new H2 verdict or a serving change.

## 13. Stage 8: Trained SC, dynamic bound, and PAFA + PC-MCL with SC

BEATs PAFA, SG-SCL and Patch-Mix CL (cells `<cell>+sc`, 75 units) and BEATs PAFA + PC-MCL with trained SC static and
dynamic (cells `beats+ft+pafa_pcmcl+sc+sc_static`, `beats+ft+pafa_pcmcl+sc`, 50 units). AST and HTS-AT dynamic units were dropped (user decision 21/10); 4 BEATs Patch-Mix CL dynamic
units finished before the switch.

Done 23/10 02:20, 125 / 125 units, 0 failed (one unit cut off by a reboot of E on 22/10 was rerun from epoch 0; lists
were rebalanced twice onto B). Official split = last epoch vs the no-SC unit of the same cell and seed (PAFA + PC-MCL vs
its `pcmcl-v1` unit); LODO = macro over the 4 held-out devices, d vs the no-SC LODO cell (PAFA + PC-MCL vs BEATs PAFA).

| Cell | Official Score | d | Worst-dev Score | Set size | LODO macro | d | Litt3200 Score (Sp / Se) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BEATs PAFA, no SC | 61.7 ± 0.7 | | 58.9 | 2.47 | 58.8 ± 1.4 | | 59.7 (88.5 / 30.9) |
| BEATs PAFA, SC static | 62.3 ± 0.9 | +0.5 ± 1.5 | 58.9 | 2.33 | 60.3 ± 0.8 | +1.5 ± 1.2 | 61.5 (88.2 / 34.8) |
| BEATs PAFA, SC dynamic | 62.0 ± 0.5 | +0.3 ± 0.6 | 59.3 | 2.24 | 57.9 ± 1.6 | -0.9 ± 1.5 | 61.3 (88.5 / 34.2) |
| BEATs SG-SCL, SC static | 60.7 ± 1.0 | -0.4 ± 1.6 | 51.8 | 2.43 | 60.5 ± 1.5 | +1.5 ± 1.1 | 57.8 (68.3 / 47.3) |
| BEATs SG-SCL, SC dynamic | 62.8 ± 1.4 | +1.7 ± 1.3 | 58.4 | 2.38 | 59.5 ± 0.8 | +0.4 ± 1.2 | 60.2 (81.8 / 38.5) |
| BEATs Patch-Mix CL, SC static | 59.8 ± 0.8 | +0.1 ± 1.2 | 52.0 | 2.30 | about 60.1 | -0.0 ± 1.4 | about 57.2 |
| BEATs Patch-Mix CL, SC dynamic | 61.5 ± 1.0 | +1.8 ± 1.3 | 57.4 | 2.26 | 59.9 ± 1.6 | -0.2 ± 1.8 | 61.4 (73.6 / 49.3) |
| BEATs PAFA + PC-MCL, SC static | 61.8 ± 0.2 | +0.5 ± 1.1 | 59.0 | 2.01 | 60.7 ± 1.5 | +1.9 ± 1.2 | 59.5 (76.7 / 42.3) |
| BEATs PAFA + PC-MCL, SC dynamic | 60.9 ± 1.3 | -0.4 ± 1.7 | 56.6 | 2.03 | 57.4 ± 0.2 | -1.4 ± 1.3 | 63.0 (85.6 / 40.4) |

### 13.1 Findings (stage 8)

- [Fact] The dynamic bound helps on the official split for SG-SCL and Patch-Mix CL (+1.7, +1.8; worst device +5.5,
  +5.3): BEATs SG-SCL + SC dynamic, 62.8 ± 1.4, is the best last-epoch Score of any single cell, within one SD of BEATs
  PAFA + SC static (62.3 ± 0.9). For PAFA the two bounds are level on the official split (62.0 vs 62.3).
- [Fact] In LODO the dynamic bound raises Litt3200 for every cell (+1.7 to +5.2) but lowers the macro for PAFA (-0.9) and
  PAFA + PC-MCL (-1.4), mostly through Meditron (PAFA 57.2 to 53.5; PAFA + PC-MCL 47.9). The static bound gives the best
  LODO macro (PAFA + PC-MCL 60.7, SG-SCL 60.5, PAFA 60.3).
- [Fact] PAFA + PC-MCL with SC static: steadiest official Score (SD 0.2) and smallest sets (2.01 labels), LODO macro
  +1.9 over BEATs PAFA without SC; with SC dynamic it has the best Litt3200 Score of any arm (63.0, Sp 85.6, Se 40.4).
- [Fact] No arm changes coverage beyond noise; no LODO fold passes the null gate, so H2 stays untestable (stage 7).
- [Interpretation] For BEATs PAFA the static bound is the better SC variant: equal on the official split, +2.4 macro
  over the dynamic bound in LODO, same Litt3200.

### 13.2 Dynamic SC on AST and HTS-AT (24/10 completion run, 100 units)

[Fact] Run `sc-dynamic-completion-20261024` (4 cells x (official split + 4 folds) x 5 seeds, all 100 reports present on A;
last epoch). Scores recomputed on 27/10 from the `report.json` files; static counterparts from `lodo-v1`.

| Cell | Dynamic official | Static official | Dynamic LODO macro | Static LODO macro |
| --- | ---: | ---: | ---: | ---: |
| AST PAFA | 59.88 | 59.03 | 57.17 | 58.46 |
| AST Patch-Mix CL | 59.60 | 58.29 | 57.01 | 56.70 |
| HTS-AT Patch-Mix CL | 59.82 | 58.47 | 56.90 | 58.13 |
| HTS-AT SG-SCL | 60.09 | 59.35 | 56.60 | 58.10 |

[Interpretation] Dynamic SC is higher on the official split for all four cells (+0.9 to +1.3) and lower on the LODO macro
for three of four, as for BEATs PAFA in section 13; only AST Patch-Mix CL gains on both. The earlier partial reading
(AST Patch-Mix CL dynamic minus static: +3.35 Litt3200, -2.88 LittC2SE) is part of this table
(`outputs/train/sc-dynamic-completion-20261024`). No paired intervals were computed for this table.

## 14. Stage 9: Ensembles and distillation, from `docs/reports/2026-10-22-ensemble-v1-results.md`

Groups of the user (22/10; not the registered E1-E4): E1 = BEATs PAFA, SG-SCL, Patch-Mix CL; E2 = AST Patch-Mix CL, PAFA;
E3 = HTS-AT Patch-Mix CL, SG-SCL; and every cross-group union. Weighted mean softmax at the last epoch: `equal`, or
`fit` = simplex weights minimising the NLL of the LODO calibration patients (source devices; the official split reuses
the mean LODO weights). d = paired by seed vs BEATs PAFA alone in the same arm.

| Ensemble (equal weights) | Official Score | d | Worst-dev Score | Set size | LODO macro | d | Litt3200 | Litt3200 Sp / Se |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BEATs PAFA alone, no SC | 61.7 ± 0.7 | | 58.9 | 2.47 | 58.8 ± 1.4 | | 59.7 | 88.5 / 30.9 |
| E1 BEATs | 62.5 ± 0.5 | +0.8 ± 0.5 | 57.9 | 2.18 | 60.5 ± 1.5 | +1.7 ± 0.9 | 58.2 | 68.5 / 47.8 |
| E1+E2 | 62.3 ± 0.6 | +0.5 ± 1.2 | 56.6 | 1.92 | 60.5 ± 1.5 | +1.7 ± 0.9 | 57.8 | 64.5 / 51.0 |
| E1+E3 | 63.4 ± 0.3 | +1.7 ± 0.5 | 57.2 | 1.90 | 60.9 ± 1.5 | +2.1 ± 1.5 | 58.6 | 62.2 / 55.0 |
| E1+E2+E3 | 63.2 ± 0.3 | +1.5 ± 0.4 | 56.5 | 1.87 | 60.7 ± 1.4 | +1.9 ± 1.0 | 57.8 | 59.5 / 56.0 |
| E2 AST | 59.9 ± 0.4 | -1.8 ± 0.9 | 49.7 | 2.14 | 58.1 ± 0.7 | -0.7 ± 1.3 | 52.5 | 52.7 / 52.2 |
| E3 HTS-AT | 60.1 ± 1.0 | -1.7 ± 1.0 | 48.3 | 2.18 | 57.2 ± 1.8 | -1.6 ± 2.4 | 50.0 | 40.5 / 59.5 |
| E2+E3 | 60.8 ± 0.9 | -0.9 ± 1.0 | 48.5 | 1.98 | 58.8 ± 1.6 | -0.0 ± 1.9 | 51.9 | 46.7 / 57.1 |
| BEATs PAFA alone, trained SC static | 62.3 ± 0.9 | | 58.9 | 2.33 | 60.3 ± 0.8 | | 61.5 | 88.2 / 34.8 |
| E1 BEATs, SC static | 63.1 ± 0.6 | +0.9 ± 1.1 | 58.0 | 2.06 | 61.9 ± 1.1 | +1.6 ± 1.1 | 61.0 | 72.1 / 49.8 |
| E1+E2, SC static | 62.7 ± 0.3 | +0.5 ± 1.0 | 57.6 | 1.88 | 61.6 ± 1.1 | +1.3 ± 0.9 | 60.4 | 72.4 / 48.3 |
| E1+E3, SC static | 63.2 ± 0.7 | +0.9 ± 1.4 | 57.7 | 1.89 | 62.0 ± 0.7 | +1.7 ± 1.1 | 58.5 | 62.5 / 54.6 |
| E1+E2+E3, SC static | 63.0 ± 0.7 | +0.7 ± 1.1 | 55.9 | 1.85 | 62.1 ± 0.9 | +1.8 ± 0.9 | 59.4 | 65.1 / 53.8 |
| E2 AST, SC static | 60.2 ± 0.7 | -2.1 ± 0.4 | 52.2 | 2.11 | 58.7 ± 0.9 | -1.6 ± 1.2 | 55.8 | 65.6 / 46.1 |
| E3 HTS-AT, SC static | 59.5 ± 0.8 | -2.7 ± 1.4 | 47.9 | 2.20 | 58.7 ± 0.9 | -1.6 ± 1.6 | 50.1 | 39.6 / 60.5 |
| E2+E3, SC static | 60.9 ± 0.4 | -1.4 ± 0.7 | 49.8 | 1.96 | 59.9 ± 0.8 | -0.4 ± 1.4 | 54.2 | 53.2 / 55.1 |

E2, E3 and E2+E3 are below BEATs PAFA or level with it everywhere (official -0.7 to -2.8, LODO macro 0.0 to -2.3,
Litt3200 -5.7 to -11.6); fitted-weight rows, fitted weights and per-device Scores in the detailed report.

### 14.1 Findings (stage 9)

- [Fact] Trainable weights do not help: fitted weights are below equal weights or level with them in every ensemble
  (LODO macro -0.1 to -1.5, official -0.9 to +0.2); a class-balanced NLL objective gives the same picture. The fit gives
  BEATs PAFA the smallest weight (0.07-0.21), although PAFA is the member that holds Sp on the unseen Litt3200.
- [Fact] Equal-weight E1, E1+E3 and E1+E2+E3 beat BEATs PAFA alone by more than one paired seed SD on both the official
  split and the LODO macro without SC (best E1+E3: +1.7 and +2.1); with trained SC the LODO gains stay (+1.6 to +1.8)
  and the official-split gains fall within one SD (+0.7 to +0.9).
- [Fact] Every ensemble loses on Litt3200 (-0.6 to -11.6) and on the official-split worst-device Score (-0.9 to -11.0):
  ensembles trade PAFA's Sp for Se. HTS-AT complements BEATs a little more than AST (E1+E3 >= E1+E2).
- [Fact] Ensembles give smaller conformal sets at the same coverage (1.85-2.18 vs 2.33-2.47 labels on the official split).
- [Assumption] Exploratory: members and groups were chosen after seeing test results. The registered E1-E4
  predictions were computed on 24/10 (section 12.2); the distillation pilot and the refined KD screen are in section 17.4.

### 14.2 Raising Se while keeping Sp; best epoch on test (optimistic)

Rule `k rate` (label-free): the normal probability is scaled so that the ensemble calls the same fraction of the
unlabelled test cycles normal as BEATs PAFA alone does on the same cycles (transductive, like SC). d = paired by seed vs
PAFA alone (Sp and Se in points). Best epoch on test = every member at its own best epoch on test, plain argmax
[optimistic]. Other rules (Sp matched on source-device calibration, oracle on test labels): detailed report, section 5.

| Ensemble (equal, `k rate`) | Official Score | dSp / dSe | LODO macro | dSp / dSe | Litt3200 Sp / Se | Best epoch on test [optimistic] (Sp / Se) |
| --- | --- | --- | --- | --- | --- | --- |
| BEATs PAFA alone, no SC | 61.7 ± 0.7 | | 58.8 ± 1.4 | | 88.5 / 30.9 | 63.2 ± 0.6 (75.5 / 51.0) |
| E1 BEATs | 62.8 ± 0.7 | +0.7 / +1.5 | 60.5 ± 0.9 | +0.9 / +2.6 | 88.2 / 30.2 | 65.1 ± 0.3 (80.1 / 50.1) |
| E1+E2 | 62.4 ± 0.6 | +0.4 / +0.8 | 60.7 ± 1.1 | +1.2 / +2.6 | 88.1 / 29.0 | 65.3 ± 0.5 (83.2 / 47.5) |
| E1+E3 | 63.2 ± 0.4 | +0.6 / +2.3 | 60.9 ± 1.1 | +1.2 / +3.1 | 88.1 / 29.5 | 65.8 ± 0.7 (81.9 / 49.7) |
| E1+E2+E3 | 62.9 ± 0.3 | +0.5 / +1.8 | 61.0 ± 1.1 | +1.4 / +3.1 | 88.0 / 29.1 | 65.8 ± 0.5 (84.0 / 47.6) |
| E2+E3 | 60.7 ± 0.8 | -1.3 / -0.8 | 59.8 ± 1.5 | +0.6 / +1.5 | 86.5 / 26.4 | 64.4 ± 0.7 (83.5 / 45.2) |
| BEATs PAFA alone, trained SC static | 62.3 ± 0.9 | | 60.3 ± 0.8 | | 88.2 / 34.8 | 64.6 ± 0.4 (85.1 / 44.2) |
| E1 BEATs, SC static | 63.4 ± 0.5 | +0.9 / +1.4 | 62.0 ± 0.6 | +1.1 / +2.4 | 88.4 / 36.0 | 65.8 ± 0.5 (84.8 / 46.8) |
| E1+E3, SC static | 63.4 ± 0.4 | +0.6 / +1.6 | 62.4 ± 0.8 | +1.2 / +2.9 | 87.8 / 34.3 | 66.1 ± 0.5 (85.6 / 46.5) |
| E1+E2+E3, SC static | 63.1 ± 0.5 | +0.4 / +1.1 | 62.1 ± 0.6 | +1.0 / +2.5 | 86.9 / 31.8 | 65.8 ± 0.4 (86.1 / 45.6) |
| E2+E3, SC static | 60.8 ± 0.9 | -1.5 / -1.4 | 60.3 ± 0.6 | -0.1 / +0.1 | 84.8 / 26.8 | 64.3 ± 0.5 (82.9 / 45.7) |
| BEATs PAFA alone, trained SC dynamic | 62.0 ± 0.5 | | 57.9 ± 1.6 | | 88.5 / 34.2 | 63.9 ± 0.3 (81.6 / 46.2) |
| E1 BEATs, SC dynamic | 64.3 ± 0.5 | +1.8 / +2.9 | 59.9 ± 1.5 | +1.1 / +3.0 | 88.5 / 33.6 | 65.6 ± 1.1 (82.0 / 49.3) |

- [Fact] With `k rate`, E1, E1+E3 and E1+E2+E3 keep PAFA's Sp (dSp +0.4 to +1.4) and raise Se on both splits (official
  +1.1 to +2.3, LODO macro +2.4 to +3.1); best: E1+E3, LODO Score +2.1 (no SC: 60.9; trained SC: 62.4).
- [Fact] Plain argmax (section 14 table) trades Sp for Se under shift (LODO macro Sp -3 to -6, Litt3200 Sp 59-72); Sp
  matched on source-device calibration does not transfer (held-out Sp -7 to -18). Only the target-side anchor holds Sp.
- [Fact] Litt3200 is the exception: no ensemble raises Se there at PAFA's Sp, except E1 with trained SC (36.0 vs 34.8).
- [Fact] Best epoch on test [optimistic]: E1+E3 65.8 ± 0.7 (no SC) and 66.1 ± 0.5 (trained SC); published BEATs PC-MCL
  65.37 ± 0.73 on the same rule. The test-picked epochs raise Sp, not Se.
- [Fact] E1 with SC dynamic (23/10; d vs PAFA + SC dynamic): best official-split Score of any configuration (64.3 ± 0.5),
  but LODO macro 59.9 vs 62.0 for E1 with SC static: the dynamic bound fits the official split, the static bound unseen
  devices (detailed report, section 6).
- [Assumption] `k rate` needs a batch of unlabelled target cycles and PAFA running beside the ensemble; its behaviour with
  few cycles is not measured.

## 15. Stage 10: Candidates for the final AuscultTrust

Official split = last epoch, 5 seeds; LODO = macro Score over the 4 held-out devices (arm = source + paired change).

| Candidate | Official split Score | LODO macro Score | LODO Litt3200 Score | Note |
| --- | --- | --- | --- | --- |
| BEATs PAFA + trained SC static | 62.3 ± 0.9 | 60.3 ± 0.8 | 61.5 ± 6.0 | selected primary AuscultTrust |
| BEATs SG-SCL + trained SC static | 60.7 ± 1.0 | about 60.5 | about 57.8 | alternative single model |
| BEATs Patch-Mix CL (no SC) | 59.7 ± 1.3 | 60.1 ± 1.7 | 57.5 | trained SC adds nothing here |
| BEATs PAFA (no SC) + test-time SC static | 61.7 ± 0.7 | about 58.5 | about 62.1 | no retraining; SC only at test time |
| BEATs PAFA + trained SC dynamic | 62.0 ± 0.5 | 57.9 ± 1.6 | 61.3 ± 3.9 | stage 8; weaker LODO macro (Meditron) |
| BEATs SG-SCL + trained SC dynamic | 62.8 ± 1.4 | 59.5 ± 0.8 | 60.2 ± 5.3 | stage 8; best official-split single cell |
| BEATs PAFA + PC-MCL + trained SC static | 61.8 ± 0.2 | 60.7 ± 1.5 | 59.5 ± 4.2 | stage 8; best LODO macro, smallest sets |
| E1 BEATs ensemble + trained SC static, `k rate` | 63.4 ± 0.5 | 62.0 ± 0.6 | 62.2 | stage 9; 3 models, keeps PAFA's Sp |
| E1+E3 ensemble + trained SC static, `k rate` | 63.4 ± 0.4 | 62.4 ± 0.8 | 61.0 | selected exploratory extension; 5 models |
| E1 BEATs ensemble + trained SC dynamic, `k rate` | 64.3 ± 0.5 | 59.9 ± 1.5 | 61.1 | stage 9; best official split, weaker LODO |

[Interpretation] The differences between the BEATs candidates are within one seed SD on every axis; BEATs PAFA + trained
SC is the only one that is first or close to first on all three.

**Decision (user, 23/10): baseline = BEATs + PAFA without SC; AuscultTrust = BEATs + PAFA
trained and evaluated with static SC.** E1 and E1+E3 are exploratory extensions, not replacements
for this primary comparison. Use the fixed last epochs; best-on-test results stay optimistic.

### 15.1 Review of stages 0-4 for the chosen configuration (22/10)

- [Fact] The BEATs PAFA numbers of sections 7-9 were recomputed from the `report.json` files (last epoch, 5 seeds):
  official split 61.73 ± 0.67 (no SC) and 62.27 ± 0.86 (trained SC static); LODO macro 58.8 ± 1.4 and 60.3 ± 0.8;
  per fold without / with SC: Meditron 57.2 / 57.6, LittC2SE 62.0 / 67.5, Litt3200 59.7 / 61.5, AKGC417L 56.4 / 54.6.
  They match the tables (the earlier "about 61.6" on Litt3200 is 61.5 from the units directly).
- [Fact] Why BEATs PAFA: the encoder effect dominates (stages 2-3: BEATs first on both epoch rules and in LODO), and PAFA
  is the one consistent method gain on BEATs (+2.6 Score, +8.5 worst-device Score over BEATs CE, paired by seed); in LODO
  it is the best cell on Litt3200, the hardest device (59.7 vs 47.3-57.5).
- [Fact] Its weak side is sensitivity: PAFA raises Sp and lowers Se (official split Sp 73.2, Se 50.2; Litt3200 fold Sp
  88.5, Se 30.9 without SC, 34.8 with SC). In LODO its macro Score without SC (58.8) is the lowest of the BEATs cells
  (58.8-60.1, all within seed SD); trained SC lifts it to 60.3.
- [Fact] Trained SC static on BEATs PAFA is not uniform across devices: +5.5 on LittC2SE, +1.9 on Litt3200 (seed SD
  rises from 3.3 to 6.0), -1.8 on AKGC417L; the official-split worst-device Score does not change (-0.1 ± 1.2).
- [Interpretation] The choice rests on Score and worst-device Score, not on coverage: no LODO v1 fold passes the null
  gate (stage 7), so coverage under device shift is not yet shown for this model. The LODO numbers are selected numbers
  (section 18).

### 15.2 Benchmark protocol audit (24/10)

[Fact] `docs/reports/2026-10-24-benchmark-protocol-audit.md` checks published targets, test-selected versus fixed epochs and
the official split's two shared patients (156 and 218). The added unseen-patient test view has 2,636 cycles from 47 patients:
PAFA / static Score 62.03 and the static three-BEATs / two-HTS-AT mean 63.21 (five-seed means, raw softmax, no `k rate`).
A warm-up-duration bug in the local code was fixed with regression tests; frozen campaigns were not changed by it and no
model-quality improvement follows (preflight note removed 27/10; `docs/README.md`).

## 16. Stage 11: Frozen-model KAUH and recording screening

Primary evaluation completed 23/10 15:25, five paired seeds. Decision and full process log:
`docs/decision-logs/2026-10-23-final-model.md`, `docs/reports/2026-10-23-final-pafa-static-execution.md`.

- [Fact] KAUH-A: flip rate 0.445 ± 0.047 (baseline) → 0.209 ± 0.027 (AuscultTrust), reduction
  +0.236, patient/seed bootstrap 95% CI [+0.177, +0.305]. Test-time-only SC: 0.212 ± 0.025.
- [Fact] KAUH-B2: 2/6 ordered filter pairs pass the null gate; absolute coverage-error reduction
  +0.1130 [0.0724, 0.1532] for AuscultTrust, +0.1346 [0.0856, 0.1608] for test-time-only SC.
  Both software-filter positive controls pass; no claim about hardware H2 follows.
- [Fact] ICBHI recording screening (mean windows, 20 patient splits): baseline coverage 89.8%,
  set size 1.570, uncertain 57.0%; AuscultTrust 90.0%, 1.603, uncertain 60.3%. Argmax Se/Sp:
  70.3/62.1% vs 64.2/67.7%. These are descriptive averages, not confidence bounds.
- [Fact] ICBHI recording thresholds on KAUH (B1): AuscultTrust coverage 83.1/86.1/87.9% on
  Bell/Diaphragm/Extended. Nominal coverage does not transfer uniformly.
- [Fact] Validated PAFA/static-SC seed-0 bundle activated; local API health and checkpoint hashes verified.
- [Fact] E1 raw-mean extension: KAUH flip 0.318 → 0.188, reduction +0.130 [0.084, 0.177];
  B2 2/6 tested pairs, D +0.0615 [0.0301, 0.0887]. E1+E3: flip 0.304 → 0.190,
  reduction +0.115 [0.073, 0.159]; B2 not tested (0/6 gated pairs).
- [Fact] Recording screening with static-SC raw ensembles: E1 coverage 91.2%, set size 1.596,
  uncertain 59.6%; E1+E3 90.7%, 1.535, uncertain 53.5%. These use raw mean softmax, not `k rate`.
- [Pending] Phantom H1/H3/co-primary H2 requires physical recordings.


## 17. Stage 12: Additional improvement trials

Scope: training and inference changes tried after the 23/10 decision, in time order. Selection rule throughout:
patient-disjoint source cross-validation on the official training patients; epoch from the mean validation curve
of the arm itself (`cv_epoch`); target outcomes (official test, LODO, KAUH) never select a recipe. Common recipe:
BEATs fine-tuned, PAFA, trained static SC, 100-epoch cosine schedule, Adam 5e-5, batch 32, PCSL 50, GPAL 5e-4, unless an
arm overrides it (`experiments/train/*.yaml`). **gain6** = random per-bin gain, 6 dB SD, on top of that recipe. This
stage is exploratory: the frozen AuscultTrust of stage 10 (no gain6, official 62.3) is a different object from the
gain6 anchor below (official 63.1), and no number in this stage replaces it.

### 17.1 25/10: source CV of the PAFA pooling and gain6 (3 folds x 2 seeds)

Source: `logs/improvement-20261025/cv-selection*.json`; table in `docs/reports/2026-10-27-improvement-results.md`.

| Arm | Epoch | Source Score | Delta to control |
| --- | ---: | ---: | ---: |
| control (mean pooling) | 21 | 60.125 | 0 |
| token attention | 50 | 60.518 | +0.393 |
| attention + gain6 | 61 | 60.623 | +0.499 |
| mean + gain6 | 67 | 61.084 | +0.959 |
| attention + gain6, half LR | 33 | 60.207 | +0.082 |

[Fact] Mean pooling + gain6 was the best source arm and became the "anchor" of the later work. The released BEATs PAFA uses
learned token attention; the local variant uses mean pooling (`docs/reports/2026-10-24-pafa-pooling-audit.md`; CPU algebra and
gradient checks show a material training difference, and `ce_pafa_attn` passes the research tests). The 48-hour sprint
(48-hour sprint, 24/10) replaced the larger 20-family programme (deferred; both plans removed 27/10, `docs/README.md`).

### 17.2 26/10: nine-idea screen, LLRD confirmation, frozen comparison

Source: `docs/reports/2026-10-27-improvement-results.md` (the 26/10 completion note was folded into it). 30 screen fits (3 folds, seed 0), 6 confirmation fits (seed 1),
25 final fits (5 seeds x official + 4 LODO). Gate: mean source delta >= 0.5 and a positive majority of folds.

| Arm | Epoch | Source delta | Gate |
| --- | ---: | ---: | --- |
| LLRD | 17 | +0.752 | pass (confirmed: 21, +0.553) |
| patient cap | 13 | -0.686 | fail |
| class top-k | 21 | -0.614 | fail |
| class attention | 68 | -1.072 | fail |
| class-aware (PCSL 50) | 29 | +0.042 | fail |
| recording head | 3 | -19.460 (binary recording Score) | fail |
| recording head, PCSL 0 | 4 | -1.251 | fail |
| band noise | 11 | -0.644 | fail |
| band noise + consistency | 20 | -1.856 | fail |

Frozen five-seed comparison (Score, mean ± SD):

| Endpoint | Static control (epoch 21) | gain6 (epoch 67) | LLRD (epoch 21) |
| --- | ---: | ---: | ---: |
| official | 61.044 ± 2.001 | 62.416 ± 0.633 | 62.313 ± 1.344 |
| AKGC417L | 55.012 ± 1.112 | 55.293 ± 1.735 | 54.648 ± 1.004 |
| Litt3200 | 60.775 ± 6.898 | 61.697 ± 7.648 | 59.683 ± 6.275 |
| LittC2SE | 67.086 ± 2.600 | 67.385 ± 2.002 | 67.321 ± 1.176 |
| Meditron | 58.843 ± 4.162 | 59.999 ± 2.998 | 60.789 ± 2.635 |
| LODO macro | 60.429 | 61.093 | 60.610 |

[Fact] LLRD vs gain6: official delta -0.102 (paired 95% interval [-2.799, +2.195]), LODO macro -0.483; KAUH mean-window
screening raises LLRD referral sensitivity and lowers specificity. [Interpretation] No improvement over gain6 is shown.
The preprocessing audit kept the registered 8-second rule (binary Score: registered 71.769, 5 s 68.751, scipy 71.276,
no recording fade 71.292).

### 17.3 26/10: follow-up on B/D/E/F (anchor refit, event heads, fixed ensembles)

Sources: `docs/reports/2026-10-27-improvement-results.md`, `logs/followup-20261026/HANDOFF.md`.

- [Fact] Matched gain6 anchor refit at source-selected epoch 66 (25 fits, cell `gain6-cv66`): official **63.143 ± 1.203**;
  AKGC417L 55.406, Litt3200 63.573, LittC2SE 67.191, Meditron 60.303 (macro 61.618). Different epochs and protocols
  from the 62.416 benchmark above; no paired superiority claim.
- [Fact] Event heads (shared encoder + PAFA + auxiliary crackle / wheeze BCE, weight 0.25; 3 folds, seed 0): all 3 fits
  finished on B. The controller never recorded the gate; computed on 27/10 with `improvement_campaign.gate` against the
  earlier anchor reports: epoch 22 vs 73, mean delta **-1.637**, fold deltas -1.474 / -0.414 / -3.022: **fail**. No
  confirmation or final fits were run.
- [Fact] Fixed equal-weight additions on the common official cohort (5 seeds): E1 63.424 -> 64.307 with gain6, 63.663
  with LLRD; E1+E3 63.378 -> 64.228 with gain6, 63.806 with LLRD. LODO is mixed (E1+gain6 lower on AKGC417L 56.536 ->
  55.912 and Litt3200 62.225 -> 62.184; E1+E3+gain6 higher on Meditron 61.640 -> 62.297 and LittC2SE 70.091 -> 70.716).
  Historical LODO-v1 members and new LODO members are mixed protocols.

### 17.4 Distillation

- [Fact] KD pilot (old student, official / Litt3200): official 62.751 ± 0.757 (3 seeds: 63.607 / 62.167 / 62.480) vs matched
  PAFA 62.50 ± 1.09 and teacher 63.47 ± 0.47; Litt3200 60.565 ± 6.718 (54.144 / 60.007 / 67.545). Patient / seed intervals
  include zero (`outputs/train/distill-pilot-20261024`).
- [Fact] Refined KD (hard CE weight 0.5 + soft KL, T = 2, KL x T^2, teacher snapshot epoch 66 from the same outer-fold cohort):
  25 CPU tests pass; the 3-fold seed-0 screen finished on B (`next-improvements-20261026-s0/kd/cv0-2`). Computed on 27/10:
  epoch 13 vs anchor epoch 21, mean delta **+0.009**, fold deltas +0.994 / -1.758 / +0.791: **fail**. Earlier text that
  gave "no GPU result" is superseded by this line.

### 17.5 Recipes screened in 3 folds, then in 5 folds x 5 seeds

The five recipes below share an anchor in each screen. Config: `swa` (average from epoch 10), `lora` (rank 8),
`multiscale`, `class_aware_light` (class-aware PAFA pairing, PCSL 5), `lora_class_aware`.

**3-fold, seed 0 (`next-improvements-20261026`; gate recomputed on 27/10 from the B reports).** Anchor epoch 21.

| Arm | Epoch | Mean delta | Fold deltas | Gate |
| --- | ---: | ---: | --- | --- |
| SWA | 94 | -0.444 | +0.96 / -2.58 / +0.28 | fail |
| LoRA | 92 | -3.246 | -1.72 / -6.22 / -1.80 | fail |
| multi-scale | 33 | -0.100 | +1.03 / -1.25 / -0.09 | fail |
| class-aware light | 32 | +0.044 | +3.10 / -2.26 / -0.71 | fail |
| LoRA + class-aware | 77 | -1.097 | -0.24 / -0.21 / -2.84 | fail |
| refined KD | 13 | +0.009 | +0.99 / -1.76 / +0.79 | fail |

This programme was stopped; its service on E fails the snapshot integrity check (section 4.2).

**5 folds x 5 seeds (150 fits, `fivefold-20261026`, certified `summary.json`, mirror manifest 150 / 150 receipts).**
Anchor = gain6 + static SC at its source-selected epoch 66. Epoch of each arm from its own mean curve.

| Arm | Epoch | Mean source Score | Delta to anchor | Seed deltas (0-4) | Fold deltas (0-4) |
| --- | ---: | ---: | ---: | --- | --- |
| anchor | 66 | 60.587 | 0 | | |
| SWA | 43 | 60.663 | +0.076 | -0.37 / +1.28 / +0.53 / -0.21 / -0.86 | -0.37 / +0.60 / +0.57 / +0.21 / -0.64 |
| LoRA | 83 | 61.194 | +0.607 | +0.23 / +0.92 / +0.84 / -0.27 / +1.32 | +1.31 / +4.17 / -1.54 / +0.72 / -1.62 |
| multi-scale | 68 | 60.266 | -0.321 | -0.76 / +0.24 / +0.52 / -0.89 / -0.72 | -0.20 / -1.53 / +0.37 / -0.54 / +0.30 |
| class-aware light | 4 | 62.045 | +1.458 | +0.88 / +2.53 / +2.33 / +1.93 / -0.38 | +0.92 / +4.19 / +1.63 / +0.31 / +0.24 |
| LoRA + class-aware | 43 | 61.282 | +0.695 | +0.14 / +1.01 / +1.29 / +0.06 / +0.97 | +1.27 / +5.06 / -2.93 / +1.03 / -0.95 |

At epoch 4, class-aware light scores 62.045 and the anchor 55.802 on the same 25 units (paired +6.243).

[Interpretation] Read these as source-validation estimates only:

0. The selected epoch is noisy. The same anchor recipe (identical `train_args.json` apart from the fold count and unused
   arm switches), same folds and seed 0, was selected at epoch 21 in `next-improvements-20261026` and at epoch 73 in
   `improvements-20261026`; the 3-fold table above and the 5-fold table below use different anchors and epochs.
1. Epochs and deltas come from the same validation curves (selection on the evaluation data). Fold 1 carries most of the
   LoRA and class-aware gains (+4.2, +5.1); the sign of the LoRA effect flipped between the 3-fold screen (-3.25) and
   the 5-fold run (+0.61).
2. Class-aware light is selected at epoch 4 of a 100-epoch cosine schedule, i.e. an early-stopped model; the gain may
   reflect shallow fine-tuning. It also changes PCSL from 50 to 5, so the pairing change is not isolated.
   A non-class-aware PAFA control with PCSL 5 is required before any mechanism claim.
3. By a mean >= +0.5 and positive-majority-of-folds reading, LoRA (3 of 5 folds, one negative seed), class-aware light
   (5 of 5 folds) and LoRA + class-aware (3 of 5 folds) pass here; SWA and multi-scale do not. The earlier 3-fold
   screen passed none.
4. [Pending] None of the five has an official, LODO or KAUH result. The serving model and the main comparison do not
   change. The plan to measure them is `docs/updates/2026-10-27-target-evaluation-plan.md`.

### 17.6 Adaptation, fusion, audits

- [Fact] Target-spectrum adaptation (gain6 anchor, unlabelled per-filter SC; 33 adaptation, 78 evaluation patients,
  5 seeds): Score change +3.38 on Bell (95% paired interval [-4.58, +11.38]), -0.74 on Diaphragm, -0.56 on Extended; every
  interval crosses zero. Software filters; separate protocol; not deployed
  (`outputs/evaluation/next-addons-20261026/summary.json`).
- [Fact] Learned fusion was not fitted: complementary patient-disjoint out-of-fold predictions exist for the anchor and
  LLRD only (4,142 cycles, 79 patients, 3 folds) and no independent fusion-validation set was available
  (`outputs/evaluation/next-addons-20261026/fusion-feasibility.json`).
- [Fact] Source audio audit (official training patients, 4,025 cycles, 77 patients; patients 156 and 218 excluded): 0 invalid
  boundaries, 0 confirmed label errors, 50 recordings or cycles waiting for human listening
  (`outputs/evaluation/source-audio-audit-20261026/review.csv`). Source audio and annotations were not modified.

## 18. Caveats

- One split and 5 seeds: a 1-point difference is within seed noise for most cells. The last epoch is the reference
  schedule's, not a validated epoch; best-epoch numbers are test-selected.
- Grid cells differ in batch size, cycle length and SpecAugment as well as in the method; only PAFA has a settings
  control (ce_pafa_cfg).
- [Fact] The LODO v1 and SC cells were chosen after seeing official-split test results; the held-out folds contain those
  test cycles (mild leakage, stated in Amendment 4). The final AuscultTrust is selected on the LODO folds, so its LODO
  numbers are selected numbers. The registered AST results stay the registered outcomes.
- No LODO v1 fold passes the null gate, so no coverage claim (H2) is made from LODO v1. The AKGC417L fold is class-mix
  dominated (training keeps at most 321 of 1,864 crackle cycles).
- ICBHI device is confounded with class mix and site. GPU numerics are not bit-reproducible (`cudnn.benchmark`, AMP).
  Units cut off by a crash or kill are rerun from epoch 0, never resumed, so every reported unit is an uninterrupted run.
- SC units ran on three code states (C: commit before the PC-MCL port; D, A, E: current; B: older), identical on every
  path the SC cells use (checked by file hashes and diff).
- [Fact] Stage 12 results are source-CV numbers with epochs chosen on the same curves. The 3-fold and 5-fold screens
  disagree in sign for several recipes. No stage-12 recipe has a target-device result. The 150 fits ran from the frozen
  snapshot `build/campaigns/fivefold-20261026/src`, which is identical to the refactored tree except for the 27/10
  class-level defaults in `src/training/encoders/beats.py` (no effect on trained instances).
- KAUH filters are vendor software transformations of the same recordings, not physical stethoscopes.

## 19. Next steps

State on 27/10: all registered and exploratory training of the main study is finished; stage 12 source CV is finished.

1. **Target evaluation of the stage-12 recipes** on A, B, D, E: official, four-device LODO and KAUH (with recording
   screening) for 5 seeds, from the frozen `fivefold-20261026` snapshot, with a PCSL-5 non-class-aware control and the
   anchor as paired references. Tiers, GPU-hour estimate, preflight checks and stop rules:
   `docs/updates/2026-10-27-target-evaluation-plan.md`. [Running since 27/10 18:12 as campaign `target-20261027`; see `docs/handoff/HANDOFF.md` section 0.]
2. Human listening of the 50 audio-audit candidates; keep the registered 8-second preprocessing.
3. Optional before the cut-off (31/01/2027): `k rate` with few target cycles and SC jointly at finite N; distillation of an
   ensemble into one BEATs model. The registered E1-E4 and the AST / HTS-AT dynamic-SC cells are already computed (sections 12.2, 13.2).
4. Phantom: hardware, capture protocol, remove-and-replace ceiling, then H1, H3 and co-primary H2 on the frozen
   weights and s_ref (`outputs/train/final-pafa-static/*/seed*/freeze.json`). No H2 criterion is relaxed after seeing results.
5. Commit the working tree in separate commits (refactor, campaign tooling, app, documents): it currently holds 51 modified,
   42 deleted (moved) and 75 new files that are not in git.
