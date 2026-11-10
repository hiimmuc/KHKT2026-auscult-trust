# AuscultTrust: results

| | |
| --- | --- |
| Evidence cut-off | 29/10/2026 |
| Study | Registered v9 (`docs/registered/`); current proposal v10 |
| Model | **AuscultTrust = distilled student `kd_g4_k1`** (BEATs + PAFA, trained static SC, gain6 recipe, epoch 66, soft targets from the equal-weight G4 teacher ensemble) + split conformal layer. Decision: `docs/decision-logs/2026-10-29-final-model-distilled.md` |
| Project baseline | BEATs + PAFA, fine-tuned, no SC, epoch 100 |
| Per-unit index | `outputs/index.csv` (`python scripts/index_results.py`) |

Tags: [Fact] measured, [Interpretation] reasoned, [Assumption], [Pending]. Score = (Sp + Se) / 2, ICBHI 4-class, official split,
last epoch, mean ± SD over seeds 0-4, unless stated. Best-on-test numbers are optimistic and labelled.

## 1. Findings

1. [Fact] The encoder matters more than the method. BEATs leads on the official split and under device shift; most methods are within one seed SD of CE on the same encoder.
2. [Fact] PAFA on BEATs is the one consistent method gain: +2.6 Score and +8.5 worst-device Score over BEATs CE.
3. [Fact] Trained static SC costs nothing on the official split and raises the LODO macro Score for 6 of 7 cells. Test-time SC raises Litt3200 Score for every cell.
4. [Fact] H2 (conformal coverage under device shift) is not testable in LODO: no held-out device of any converged model passes the null gate. A coverage claim needs the phantom, early-epoch models or other data.
5. [Fact] Equal-weight ensembles of BEATs models raise Score and shrink conformal sets; distilling the G4 ensemble into one student keeps about half of the LODO gain.
6. [Fact] KAUH positive controls pass (SC lowers the cross-filter flip rate). KAUH filters are vendor software, not stethoscopes: no hardware claim.
7. [Fact] Phantom replay moves over half of the cycle predictions; label-free processing plus device SC recovers part of it (section 9).
8. [Pending] Phantom H1, H3 and co-primary H2 need transducer recordings.

## 2. Official split: encoder x method grid (`grid-v1`)

70 cells (5 encoders x 14 methods) x 5 seeds. Fine-tuned: CE, Patch-Mix, Patch-Mix CL, PAFA, SG-SCL, QLung, ADD-RSC, CE at PAFA's settings. Frozen: the same without Patch-Mix CL and SG-SCL.

Best fine-tuned cell per encoder (last epoch):

| Encoder | Cell | Sp | Se | Score | Worst-device Score | Set size |
| --- | --- | --- | --- | --- | --- | --- |
| BEATs | pafa | 73.2 ± 2.7 | 50.2 ± 2.9 | **61.7 ± 0.7** | 58.9 | 2.47 |
| DASS | patchmix | 70.6 ± 2.8 | 48.8 ± 1.2 | 59.7 ± 0.9 | 49.8 | 2.15 |
| HTS-AT | sg_scl | 72.4 ± 2.2 | 46.9 ± 1.2 | 59.6 ± 0.5 | 49.9 | 2.38 |
| AST | ce_pafa_cfg | 71.8 ± 2.4 | 46.2 ± 1.3 | 59.0 ± 1.0 | 48.5 | 2.29 |
| OPERA-CT | patchmix_cl | 65.4 ± 0.9 | 38.0 ± 1.4 | 51.7 ± 0.7 | 35.3 | 2.83 |

- [Fact] Fine-tuned Score range: BEATs 58.9-61.7, HTS-AT 56.3-59.6, DASS 57.5-59.7, AST 56.9-59.0, OPERA-CT 49.3-51.7. Fine-tuning beats freezing (CE) by 2.5 (AST) to 7.7 (BEATs).
- [Fact] Method effect vs CE on BEATs, paired by seed: PAFA +2.6 ± 1.4, SG-SCL +1.9 ± 2.7, Patch-Mix CL +0.5 ± 2.0, QLung +0.1 ± 1.3. PAFA and SG-SCL raise Sp, Patch-Mix CL raises Se.
- [Fact] Frozen PAFA equals frozen CE at PAFA's settings (the PAFA losses train a projector the classifier does not use). Frozen OPERA-CT nearly ignores abnormal sounds (Se 19-25).
- [Fact] Best epoch on test adds 0.6-4.4 Score over the last epoch. Top: BEATs Patch-Mix CL 63.6 ± 0.8.
- [Fact] Pooled conformal coverage is nominal (88.2-90.5%); the worst device (Litt3200) falls to 85-88% (fine-tuned) and 67-82% (many frozen cells).
- [Fact] PC-MCL on BEATs: 60.9 ± 0.6 (PC-MCL) and 61.3 ± 1.0 (PAFA + PC-MCL), below PAFA; best epochs on test are 4-6 of 100.
- [Assumption] Cells differ in batch size, cycle length and SpecAugment as well as in method; only PAFA has a settings control.

## 3. LODO (`lodo-v1`)

11 cells x 4 held-out devices x 5 seeds. Test = every cycle of the held-out device; conformal LAC at 90%, calibrated on source-device calibration patients.

| Cell | Meditron | LittC2SE | Litt3200 | AKGC417L | Macro |
| --- | --- | --- | --- | --- | --- |
| beats+ft+patchmix_cl | 57.7 | 66.4 | 57.5 | 58.8 | 60.1 ± 1.7 |
| beats+ft+qlung | 58.5 | 65.1 | 57.2 | 57.8 | 59.6 ± 0.8 |
| beats+ft+ce | 57.6 | 64.4 | 56.3 | 57.9 | 59.1 ± 1.3 |
| beats+ft+sg_scl | 58.4 | 64.8 | 54.9 | 57.9 | 59.0 ± 1.7 |
| beats+ft+pafa | 57.2 | 62.0 | 59.7 | 56.4 | 58.8 ± 1.4 |
| ast+ft+pafa | 57.0 | 64.5 | 50.5 | 56.1 | 57.0 ± 1.0 |
| htsat+ft+patchmix_cl | 58.3 | 63.4 | 49.6 | 56.1 | 56.9 ± 2.0 |

- [Fact] The encoder order holds under shift. Litt3200 is the hardest fold for every cell (Sp 33-63%), except BEATs PAFA (Sp 88.5%).
- [Fact] Methods vs CE stay within seed SD (macro -0.3 to +1.0). BEATs PAFA gains +3.3 ± 2.9 on Litt3200.
- [Fact] AKGC417L raw coverage is 74-85% but the prior-matched |Δ| is 0.03-0.06: the shortfall is class mix.
- [Fact] H2: for all 11 cells and 4 devices the baseline |Δ| stays below the 95th percentile of exchangeable re-splits (closest 0.090 vs 0.091). No fold is tested. The registered AST LODO (models stopped at epoch 2) had an under-covered Litt3200 fold (0.670); converged models do not.

## 4. SC when training (static bound, 175 units; dynamic bound, 225 units)

Official split and LODO macro, last epoch, paired by seed with the no-SC cell.

| Cell | Official (Δ) | LODO macro (Δ) |
| --- | --- | --- |
| beats+ft+pafa, static | 62.3 ± 0.9 (+0.5 ± 1.5) | 60.3 ± 0.8 (+1.5 ± 1.2) |
| beats+ft+pafa, dynamic | 62.0 ± 0.5 (+0.3 ± 0.6) | 57.9 ± 1.6 (-0.9 ± 1.5) |
| beats+ft+sg_scl, static | 60.7 ± 1.0 (-0.4 ± 1.6) | 60.5 ± 1.5 (+1.5 ± 1.1) |
| beats+ft+sg_scl, dynamic | 62.8 ± 1.4 (+1.7 ± 1.3) | 59.5 ± 0.8 (+0.4 ± 1.2) |
| beats+ft+pafa_pcmcl, static | 61.8 ± 0.2 (+0.5 ± 1.1) | 60.7 ± 1.5 (+1.9 ± 1.2) |

- [Fact] Trained SC static: official -0.6 to +0.7; LODO macro +0.3 to +1.5 for 6 of 7 cells. Gains are uneven by device for BEATs PAFA: +5.5 LittC2SE, +1.9 Litt3200, -1.8 AKGC417L.
- [Fact] Dynamic bound: higher on the official split for all four AST/HTS-AT cells (+0.9 to +1.3), lower on the LODO macro for three of four. Static is the better bound for unseen devices.
- [Fact] Coverage and set size barely move with trained SC.

## 5. SC when inferencing (880 inference units on the no-SC LODO cells)

| Arm | Δ LODO macro | Δ Litt3200 | Set size |
| --- | --- | --- | --- |
| Test-time SC, static | -0.8 to +1.1 | AST +3.4 to +4.6, HTS-AT +2.1 to +2.8, BEATs +0.6 to +2.4 | -0.06 to +0.12 |
| Test-time SC, dynamic | -1.4 to +1.1 | AST +4.1 to +5.3, HTS-AT +1.9 to +3.7, BEATs +0.7 to +2.2 | -0.05 to +0.19 |
| A2 (target statistics) | -6.1 to +0.8 | BEATs -4.8 to -11.6 | -0.07 to +0.35 |
| TTA-EQ (8 views) | 0.0 to +0.5 | -0.4 to +0.6 | -0.07 to -0.32 |

[Fact] A2 hurts every BEATs cell: matching whole-device statistics removes the held-out device's class mix with its spectrum. TTA-EQ shrinks sets slightly at the same Score.

## 6. Ensembles

Equal-weight mean softmax, last epoch. E1 = BEATs PAFA + SG-SCL + Patch-Mix CL; E3 = HTS-AT Patch-Mix CL + SG-SCL. Groups chosen after seeing test results [exploratory].

| Ensemble | Official | LODO macro | Set size |
| --- | --- | --- | --- |
| BEATs PAFA alone | 61.7 ± 0.7 | 58.8 ± 1.4 | 2.47 |
| E1 | 62.5 ± 0.5 | 60.5 ± 1.5 | 2.18 |
| E1+E3 | 63.4 ± 0.3 | 60.9 ± 1.5 | 1.90 |
| E1+E3, trained SC static | 63.2 ± 0.7 | 62.0 ± 0.7 | 1.89 |
| 8 BEATs methods (no test pick) | 62.0 ± 0.2 | | 1.95 |

- [Fact] Fitted member weights never beat equal weights. Every ensemble loses on Litt3200 (-0.6 to -11.6): it trades PAFA's Sp for Se.
- [Fact] With the label-free `k rate` rule (call as many unlabelled target cycles normal as PAFA does), E1 and E1+E3 keep PAFA's Sp and gain 1.4-3.1 Se; best E1+E3 LODO 60.9 (no SC), 62.4 (trained SC static).

## 7. Distillation and Stage-12 recipes

Source CV selected the recipes; target endpoints are exploratory (teacher subsets chosen on the same test scores).

| Arm | Official | LODO macro |
| --- | --- | --- |
| Anchor `gain6-cv66` | 63.13 | 61.62 |
| SWA | 63.32 | 61.77 |
| Multi-scale | 62.45 | 61.41 |
| PCSL-5 control | 62.16 | 60.72 |
| Class-aware light | 58.66 | 60.07 |
| LoRA | 57.05 | 57.81 |
| LoRA + class-aware | 54.84 | 55.11 |
| G4 ensemble (anchor, control, class-aware light, multi-scale) | 64.11 | 62.22 |
| **Student `kd_g4_k1`** | **64.15** (+1.02 vs anchor, 4 of 5 seeds) | 62.2 ± 2.0 |
| Student K2 / G7-K2 | 63.47 / 62.75 | 61.77 (K2, seeds 0-2) |

- [Fact] Source-CV gains of class-aware light and LoRA did not transfer to the official split or LODO.
- [Fact] Student vs its no-SC twin: official +0.2 ± 0.8; LODO macro +1.8 ± 1.2 (5 of 5 seeds); KAUH-A flip 0.289 → 0.228 (+0.061 [+0.029, +0.093]). SC is supported on device-shift endpoints, not on the official split.
- [Interpretation] Teacher targets are in-sample on training cycles, so they are close to one-hot; this limits the distillation gain.

## 8. KAUH and recording screening (frozen models, 5 paired seeds)

| Condition | KAUH-A flip | Reduction vs baseline [95% CI] | Screening uncertain |
| --- | --- | --- | --- |
| Baseline (BEATs + PAFA, no SC) | 0.445 | | 0.570 |
| BEATs + PAFA + static SC | 0.209 | +0.236 [+0.177, +0.305] | 0.603 |
| Anchor | 0.204 | +0.240 [+0.196, +0.285] | 0.559 |
| **Student** | 0.228 | +0.217 [+0.166, +0.266] | 0.525 |

- [Fact] B2: 2 of 6 filter pairs pass the null gate; coverage-error reduction +0.100 to +0.113. Both positive controls pass.
- [Fact] ICBHI recording thresholds on KAUH give coverage 83.1/86.1/87.9% (Bell/Diaphragm/Extended): nominal coverage does not transfer uniformly.
- [Fact] Student recording screening (Bell): uncertain rate 0.31 vs 0.47 for the anchor, coverage 0.79 vs 0.81.

## 9. Phantom replay (ICBHI audio → speaker in phantom → electret / phone)

20 takes per capture device; 5-seed student ensemble; flip = share of cycles whose 4-class prediction differs from the original audio.

| Variant | Flip electret | Flip phone | Score electret | Score phone |
| --- | --- | --- | --- | --- |
| raw | 0.547 | 0.581 | 45.2 | 43.6 |
| raw + SC | 0.503 | 0.430 | 49.2 | 53.6 |
| band-pass, tone removal, denoise, SC (`tdn_sc`) | 0.408 | 0.380 | 55.8 | 58.6 |

- [Fact] Alignment is reliable (drift ≤ 4 ppm); coherence with the original is 0.02-0.15, so the playback path destroys most waveform-level agreement.
- [Deviation] The processing chain was tuned on these takes; freeze it and retest on a new capture before any claim. 20 takes per device: flip differences of 0.03 are noise.
- [Pending] Transducer-pair H1, H3 and co-primary H2.

## 10. Caveats

- One split, 5 seeds: a 1-point difference is within seed noise for most cells.
- LODO cells and the final model were chosen after seeing official-split test results; the LODO folds contain those test cycles (mild leakage). LODO numbers of the chosen model are selected numbers.
- ICBHI device is confounded with class mix and site. GPU numerics are not bit-reproducible. Cut-off units are rerun from epoch 0.
- KAUH filters are software transforms of the same recordings, not stethoscopes.
- The registered v9 criteria are unchanged. Not claimed: a new algorithm, SOTA Score, disease diagnosis, patient-level guarantees.
