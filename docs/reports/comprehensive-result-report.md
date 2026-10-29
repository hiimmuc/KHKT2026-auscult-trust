# AuscultTrust: báo cáo kết quả tổng hợp toàn diện (comprehensive result report)

| Type | Date | Status |
| --- | --- | --- |
| report | 2026-10-29 | current (exploratory; không tiêu chí đăng ký nào bị đổi) |

| | |
| --- | --- |
| Phạm vi | Mọi kết quả đo được của dự án, theo từng stage: (a) các mô hình train **trước** overall plan (Phase 0, R-plan, Stage 1 đăng ký v9 trên AST); (b) overall plan (Stage 2-11: grid, LODO, SC, PC-MCL, ensemble, KAUH); (c) additional ideas (Stage 12: CV nguồn, target evaluation, distillation, campaign `combo-20261028`) |
| Thời điểm dữ liệu | 29/10/2026 (evidence cut-off của `report.md` v7.0 là 28/10 23:18; campaign `combo-20261028` hoàn tất 29/10 02:27) |
| Nguồn gốc | Các phần mang số 2-4 và 7-19 được ghép nguyên văn từ `report.md` v7.0; mục 5 và 6 ghép từ `docs/reports/2026-10-07-phase0-report.md` và `docs/reports/2026-10-09-registered-core-report.md`; mục 1, 17.8 và Phụ lục A là phần viết mới. Số liệu không được tính lại, trừ khi ghi rõ |
| Ngôn ngữ | Phần viết mới bằng tiếng Việt; các bảng và đoạn ghép giữ tiếng Anh như file gốc để số liệu và thuật ngữ không đổi |
| Quy ước | [Fact] đo được, [Interpretation] suy luận, [Assumption] giả định, [Pending] chưa chạy. Score = (Sp + Se)/2 trên ICBHI 4 lớp, split chính thức, trừ khi ghi khác. Số là trung bình ± SD qua seed 0-4 trừ khi nêu số seed |
| Chi tiết gốc | `docs/reports/` (grid-v1, lodo-v1, ensemble-v1, pcmcl-v1, final-pafa-static, audits 24/10, improvement-results, findings, target-distillation), `docs/updates/` (prereg, amendment 1-5, plans), `docs/decision-logs/`, `docs/handoff/HANDOFF.md` |

## 1. Tóm tắt điều hành và bảng xếp hạng mọi mô hình

### 1.1 Trạng thái

[Fact] Mọi lần train của nghiên cứu chính và của stage 12 đã xong (campaign cuối `combo-20261028` xong 29/10).
**Quyết định 29/10 (người dùng): AuscultTrust = student distill `kd_g4_k1`** = BEATs + PAFA + SC tĩnh đã train, recipe gain6, epoch 66, train bằng soft target (K1) từ ensemble G4 (anchor, control PCSL-5, class-aware light, multi-scale), cộng split conformal; một mô hình khi phục vụ. Official 64.2 ± 0.4 (+1.0 so với anchor 63.1), LODO macro 62.2 ± 2.0 trên 5 seed (anchor 61.6 ± 1.4; mô hình cũ 60.3 ± 0.8). Chi tiết và giới hạn: `docs/decision-logs/2026-10-29-final-model-distilled.md`.
Định nghĩa trước (quyết định 23/10) là BEATs + PAFA + SC tĩnh đã train, epoch 100: official 62.3 ± 0.9, LODO macro 60.3 ± 0.8; giữ làm hàng tham chiếu. Baseline của dự án (quyết định 23/10, không đổi) = BEATs + PAFA không SC, epoch 100. Tương phản SC của student dùng bản sinh đôi không SC của nó (đang huấn luyện và đánh giá); chưa có thì chưa tuyên bố "SC giúp student" (xem Phụ lục B, mục 3).
Đánh giá đóng băng cuối cho mô hình mới đã chạy phần split chính thức, KAUH A/B2 và sàng lọc (mục 17.9); bản sinh đôi không SC, LODO 5 seed và H2 từng fold đang chạy; bundle phục vụ chưa đổi. Chưa có dữ liệu phổi giả nên H1, H3 và H2 đồng chính vẫn [Pending].

### 1.2 Kết luận theo thứ tự thời gian

1. [Fact] Trước v9: encoder yếu là nguyên nhân mô hình nền yếu (AST đóng băng 54.7 so với OPERA-CT 41.7); Patch-Mix CL tái hiện 61.49 ± 0.91 (best epoch trên test) và 59.59 ± 0.47 (epoch cuối) so với công bố 62.37 ± 0.61 (mục 5).
2. [Fact] Stage 1 (AST, đã đăng ký): cổng G đạt (+1.91, KTC95% [+0.31, +3.50], 10 seed); chỉ Litt3200 vượt null gate với độ phủ 0.670 → 0.904 (D = +0.177) nhưng tập dự đoán lớn hơn (mục 6).
3. [Fact] Grid (stage 2): encoder quyết định hơn phương pháp. BEATs dẫn đầu; PAFA trên BEATs là mức tăng nhất quán duy nhất (+2.6 Score, +8.5 worst-device so với CE; mục 7).
4. [Fact] LODO v1 (stage 3, 7): không fold nào của mô hình hội tụ vượt null gate nên **H2 không kiểm định được**; độ phủ gần danh nghĩa nhờ tập dự đoán lớn trên Litt3200 (mục 8, 12).
5. [Fact] SC đã train (stage 4, 8): không tốn gì trên split chính thức (-0.6 đến +0.7); tăng LODO macro cho 6/7 cell; BEATs PAFA 61.7 → 62.3 official, 58.8 → 60.3 LODO macro; lợi ích không đều (+5.5 LittC2SE, +1.9 Litt3200, -1.8 AKGC417L; mục 9, 13, 15.1).
6. [Fact] Bound động chỉ giúp split chính thức; bound tĩnh tốt hơn cho thiết bị chưa thấy (mục 13).
7. [Fact] Ensemble (stage 9): trung bình đều E1+E3 +1.7 official và +2.1 LODO macro so với PAFA đơn, ghi nhận có điều kiện; giữ Sp của PAFA chỉ khi dùng neo phía đích `k rate` (mục 14).
8. [Fact] KAUH (stage 11): hai kiểm soát dương đạt trên mô hình đã đóng băng (flip 0.445 → 0.209), nhưng chỉ là bộ lọc phần mềm; sàng lọc theo bản ghi còn 60.3% uncertain (mục 16).
9. [Fact] Stage 12, CV nguồn: class-aware light +1.458, LoRA +0.607 trên 5 fold x 5 seed (mục 17.5). **Đánh giá đích (mục 17.7) không xác nhận**: class-aware light 58.66 official (anchor 63.13), LoRA 57.05, LoRA + class-aware 54.84. Lợi ích CV nguồn không chuyển sang split chính thức hay LODO.
10. [Fact] Stage 12, ensemble và distillation: ensemble G4 64.11 official / 62.22 LODO macro; student distill K1 đạt 64.15 official (+1.02 so với anchor) và giữ khoảng một nửa lợi ích LODO (+0.50 trên 3 seed) (mục 17.7).
11. [Fact] Mô hình ghép `class_aware_multiscale` (29/10): CV nguồn +0.378 so với anchor, thấp hơn class-aware light riêng (+1.458); hai cải tiến không cộng dồn; chưa đánh giá đích (mục 17.8).
12. [Fact] Đánh giá đóng băng cuối cho student (29/10, mục 17.9): flip KAUH-A 0.228 (giảm +0.217 so với baseline) không hơn AuscultTrust cũ (0.209) hay anchor (0.204); tỷ lệ uncertain giảm (ICBHI 0.525 so với 0.603) với độ phủ KAUH thấp hơn một chút. Bản sinh đôi không SC, KAUH ghép cặp (mục 17.11: SC lúc huấn luyện giảm flip +0.061 [+0.029, +0.093]; D độ phủ +0.018 [-0.013, +0.041]) và LODO 5 seed (mục 17.10): SC không đóng góp ở split chính thức cho student (+0.2 ± 0.8; twin 64.0) nhưng nâng LODO macro (+1.8 ± 1.2, 5/5 seed); lợi ích so với baseline đến từ công thức huấn luyện. H2 vẫn không kiểm định được (không fold nào qua null gate).
13. [Pending] Phổi giả: H1, H3, H2 đồng chính. Tiêu chí đăng ký v9 không đổi.

### 1.3 Bảng xếp hạng: mọi mô hình chính, hai điểm cuối và giao thức

Official = split chính thức 60/40, LODO macro = trung bình 4 thiết bị bị giữ kín. "n/a" = không đo. Mức độ so sánh được: cột *Epoch* cho biết cách chọn epoch; best-on-test là lạc quan.
Các hàng thuộc các giao thức khác nhau (AST epoch 50 / BEATs epoch 100 / source-CV epoch) nên **không so sánh trực tiếp giữa nhóm**.

| Nhóm | Mô hình | Official Score | LODO macro | Epoch | Ghi chú / nguồn |
| --- | --- | ---: | ---: | --- | --- |
| Trước plan | AST L0/L1 đóng băng (head tốt nhất) | 54.7 ± 1.3 | n/a | CV, test một lần | mục 5 (4.1) |
| Trước plan | AST ft-k8-attn+mix (cache layer 4) | 57.3 ± 1.9 | n/a | CV | tốt nhất của ladder, mục 5 (4.2) |
| Trước plan | HeAR ft-k4-attn | 57.0 ± 0.2 | n/a | CV | mục 5 (4.2) |
| Trước plan | AST Patch-Mix CL, tái hiện | 61.49 ± 0.91 (best on test); 59.59 ± 0.47 (cuối) | n/a | best on test / cuối | công bố 62.37 ± 0.61 |
| Stage 1 (v9) | AST baseline, refit epoch CV | 56.80 ± 1.49 | LODO riêng (mục 6) | CV | mục 6 |
| Stage 1 (v9) | AST AuscultTrust (SC) | 58.31 ± 1.08 | LODO riêng (mục 6) | CV | G: +1.91 [+0.31, +3.50] |
| Grid | BEATs ft CE | 59.2 ± 1.5 | 59.1 ± 1.3 | cuối (100) | mục 7, 8 |
| Grid | BEATs ft PAFA | 61.7 ± 0.7 | 58.8 ± 1.4 | cuối | baseline của AuscultTrust |
| Grid | BEATs ft Patch-Mix CL | 59.7 ± 1.3 | 60.1 ± 1.7 | cuối | |
| Grid | BEATs ft SG-SCL | 61.1 ± 1.3 | 59.0 ± 1.7 | cuối | |
| Grid | BEATs ft QLung | 59.3 ± 0.8 | 59.6 ± 0.8 | cuối | |
| PC-MCL | BEATs PC-MCL / PAFA + PC-MCL | 60.9 ± 0.6 / 61.3 ± 1.0 | n/a | cuối | mục 10 |
| SC tĩnh | **BEATs PAFA + SC tĩnh (AuscultTrust)** | **62.3 ± 0.9** | **60.3 ± 0.8** | cuối | mô hình chính thức, mục 9, 15 |
| SC động | BEATs PAFA + SC động | 62.0 ± 0.5 | 57.9 ± 1.6 | cuối | mục 13 |
| SC động | BEATs SG-SCL + SC động | 62.8 ± 1.4 | 59.5 ± 0.8 | cuối | điểm đơn lẻ cao nhất của stage 8 |
| SC tĩnh | BEATs PAFA + PC-MCL + SC tĩnh | 61.8 ± 0.2 | 60.7 ± 1.5 | cuối | LODO macro tốt nhất, tập nhỏ nhất |
| Ensemble | E1 BEATs + SC tĩnh, `k rate` | 63.4 ± 0.5 | 62.0 ± 0.6 | cuối | 3 mô hình |
| Ensemble | E1+E3 + SC tĩnh, `k rate` | 63.4 ± 0.4 | 62.4 ± 0.8 | cuối | 5 mô hình, thăm dò |
| Stage 12 (cũ) | control tĩnh (epoch 21) | 61.044 ± 2.001 | 60.429 | CV nguồn | mục 17.2 |
| Stage 12 (cũ) | gain6 (epoch 67) | 62.416 ± 0.633 | 61.093 | CV nguồn | mục 17.2 |
| Stage 12 (cũ) | LLRD (epoch 21) | 62.313 ± 1.344 | 60.610 | CV nguồn | mục 17.2 |
| Stage 12 (target) | **anchor gain6-cv66 (epoch 66)** | 63.13 | 61.62 | CV nguồn | tham chiếu của đánh giá đích |
| Stage 12 (target) | SWA (epoch 43) | 63.32 | 61.77 | CV nguồn | ổn định nhất trong các thêm vào đơn |
| Stage 12 (target) | multi-scale (epoch 68) | 62.45 | 61.41 | CV nguồn | |
| Stage 12 (target) | control PCSL-5 (epoch 78) | 62.16 | 60.72 | CV nguồn | control không class-aware |
| Stage 12 (target) | class-aware light (epoch 4) | 58.66 | 60.07 | CV nguồn | lợi ích CV nguồn không chuyển |
| Stage 12 (target) | LoRA (epoch 83) | 57.05 | 57.81 | CV nguồn | |
| Stage 12 (target) | LoRA + class-aware (epoch 43) | 54.84 | 55.11 | CV nguồn | kém nhất |
| Stage 12 (ensemble) | G7 (bảy arm) | 64.21 ± 0.74 | 62.09 ± 1.36 | cuối | chọn trên test, lạc quan |
| Stage 12 (ensemble) | G4 (anchor, control, class-aware light, multi-scale) | 64.11 ± 0.65 | 62.22 ± 1.21 | cuối | chọn trên test, lạc quan |
| Stage 12 (distill) | **student kd_g4_k1 (AuscultTrust, 29/10)** | 64.2 ± 0.4 | 62.2 ± 2.0 (5 seed) | 66 | +1.0 official so với anchor; qua cổng; mục 17.10 |
| Stage 12 (twin) | kd_g4_k1_nosc (student tắt SC) | 64.0 ± 0.9 | 60.4 ± 2.7 (5 seed) | 66 | SC cho student: +0.2 official, +1.8 LODO macro; mục 17.10 |
| Stage 12 (distill) | student kd_g4_k2 | 63.47 | 61.77 (3 seed) | 66 | +0.33; qua cổng |
| Stage 12 (distill) | student kd_g7_k2 | 62.75 | n/a | 66 | -0.38; không qua cổng |
| Stage 12 (combo) | class_aware_multiscale | chưa đo | chưa đo | 4 | CV nguồn 60.965, +0.378 so với anchor (mục 17.8) |

### 1.4 Điều cần đọc kèm

- [Fact] Hầu hết chênh lệch giữa các mô hình BEATs trong cùng giao thức nằm trong một SD của seed; không có xếp hạng nào ngoài nhiễu trừ những mức được ghi rõ khoảng tin cậy.
- [Fact] Lựa chọn recipe, SC và nhóm ensemble đều xảy ra sau khi đã thấy kết quả split chính thức; mọi số LODO của mô hình chọn là "số đã chọn" (mục 18).
- [Fact] KAUH là bộ lọc phần mềm trên cùng bản ghi, không phải ống nghe vật lý; không có tuyên bố phần cứng nào.

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
| 12 | Additional improvement trials | Source-CV search for training improvements (25/10-27/10): attention, gain6, LLRD, event heads, KD, SWA, LoRA, multi-scale, class-aware PAFA; adaptation, fusion, audio audit | Source CV done; target evaluation (175 fits) and distillation (39 fits) done 28/10 (section 17.7); combo campaign done 29/10 (section 17.8) |
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
| 27-28/10 | Target evaluation of the stage-12 recipes executed as `target-20261027` (175 fits) and ensemble distillation as `distill-20261028` (39 fits); results in section 17.7. |
| 28/10 | Train a single model that stacks class-aware light and multi-scale (`combo-20261028`, 25 fits); also on machine 10.148.182.137 (shared, 30 GB cap). Result in section 17.8. |

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

## 5. Stage 0: các mô hình train trước overall plan (Phase 0, v4-v8, R-plan)

[Fact] Giai đoạn 02-07/10/2026, trước khi đăng ký v9. Gồm bốn nhóm thí nghiệm: (1) thang encoder đóng băng, (2) thang fine-tune (L2 last-k, L4 cache), (3) tái hiện Patch-Mix CL trên AST,
(4) Part II dịch chuyển thiết bị (độ phủ conformal khi giữ kín thiết bị / bộ lọc KAUH) và Stage 1 sàng lọc biến thể thiết bị bằng CV 3 fold. Mốc kết quả:

| Mô hình / thí nghiệm | Kết quả chính | Chi tiết |
| --- | --- | --- |
| AST đóng băng (head tốt nhất theo CV) | Test Score 0.547 ± 0.013 | 4.1 |
| OPERA-CT / CLAP / HeAR đóng băng | 0.417 ± 0.023 / 0.519 ± 0.018 / 0.527 ± 0.010 | 4.1 |
| HeAR ft-k4-attn | 0.570 ± 0.002 | 4.2 |
| AST ft-k8-attn+mix (cache layer 4) | 0.573 ± 0.019 (tốt nhất ladder; tiêu chí Part I **không đạt**) | 4.2 |
| AST Patch-Mix CL, tái hiện | 61.49 ± 0.91 best on test; 59.59 ± 0.47 epoch cuối; công bố 62.37 ± 0.61 | 4.3 |
| Part II (độ phủ, giữ kín thiết bị) | thiếu phủ AKGC417L (0.80-0.83 head AST) và Litt3200 (0.47 probe AST) | 4.4 |
| Stage 1 sàng lọc, CV 3 fold | P1 A1 input 62.4 so với baseline 61.3; SD fold ~4; không biến thể nào tách khỏi baseline | 4.5 |

Toàn văn các bảng (nguyên văn từ `docs/reports/2026-10-07-phase0-report.md`; các tiêu đề được gắn tiền tố `P0-`):

### P0-3. Models and configs used

| Item | Setting |
|---|---|
| Data | ICBHI official split (6898 cycles, 126 patients; 4142 train / 2756 test cycles), split file `../data/splits/icbhi_official_v1.json`; KAUH (112 patients x 3 filters, 8 s windows); HF_Lung_V1 |
| Frozen encoders | AST AudioSet (primary), HeAR ViT-L, CLAP HTS-AT, OPERA-CT, OPERA-CE, HTS-AT AudioSet (`htsat_audioset`, non-leaking). Configs `configs/extract_*.yaml`, `configs/ladder_*.yaml` |
| Ladder rungs | L0 logistic probe; L1 best frozen head by grouped CV (`<last\|concat\|scalar>-<mean\|max\|meanmax\|attn>`); L2 last-k blocks (`ft-k*`); L3 LoRA; L4 Patch-Mix CL. 5 seeds, epoch by grouped CV, test once |
| Patch-Mix CL | `baselines/patchmix_cl/run_repro.sh`: AST (ImageNet + AudioSet), bs 8, Adam lr 5e-5, wd 1e-6, cosine, 50 epochs, temperature 0.06, alpha 1.0, mix_beta 1.0, EMA 0.5, repeat padding, 8 s cycles |
| Stage 1 screening | `configs/stage1_variants/*.json` = Patch-Mix recipe + one flag: P1 `a1_input`, P2 `bin_norm`, P3 `rand_bin_gain` 6 dB, P4 `freq_mixstyle` (p 0.25/0.5/0.75), P6 `bin_affine`, P7 `hc_fuse` (20 MFCC frame SDs), P8 `select_metric: worst_device`, P1P3 `sc_gain`. P5 dropped (no licensed IR set). 3-fold patient-grouped CV on the 60% training patients, validation-selected epoch |
| Conformal | LAC score, alpha = 0.1; V1 split, V2 Mondrian (`src/conformal/conformal.py`), V3-V5 and V4-oracle implemented |
| Hardware | RTX 5000 Ada 32 GB (Patch-Mix seed 1 on an 8 GB laptop) |

### P0-4. Results

#### P0-4.1 Frozen encoders, ICBHI official test (5 seeds; selection by grouped CV or val)

| Encoder | CV Score | MLP val | Test Score | V1 coverage | Set size |
|---|---|---|---|---|---|
| OPERA-CT | 0.497 ± 0.056 | 0.501 | 0.417 ± 0.023 | 0.855 | 2.56 |
| AST | **0.601 ± 0.035** | **0.653** | **0.547 ± 0.013** | 0.830 | 1.85 |
| CLAP | 0.578 ± 0.028 | 0.573 | 0.519 ± 0.018 | 0.847 | 2.16 |
| HeAR | 0.580 ± 0.026 | 0.628 | 0.527 ± 0.010 | 0.857 | 2.09 |
| HTS-AT (AudioSet) | 0.601 (last-mean) | | 0.534 ± 0.005 (`last-mean`); 0.532 ± 0.006 (`last-meanmax`); L0 probe 0.491 (CI 0.430-0.546) | L0 V1 0.786 | |

[Fact] AST is the primary encoder. OPERA-CT, although pretrained on ICBHI, is lowest, so pretraining overlap did not inflate it. HTS-AT with AudioSet weights is below AST and about level with CLAP/HeAR. [Interpretation] The weak base model of the early v4 experiments was the encoder.

#### P0-4.2 Fine-tuning ladder (test Score, 5 seeds, test once)

| Rung | CV Score | Test Score |
|---|---|---|
| HeAR `last-mean` / `concat-attn` (frozen) | 0.603 | 0.513 ± 0.010 / 0.511 ± 0.005 |
| HeAR `ft-k4-attn` | 0.587 ± 0.025 | **0.570 ± 0.002** |
| AST `ft-k4-attn+mix` (cache after block 8) | 0.585 ± 0.037 | 0.557 ± 0.009 |
| AST layer-4 cache `ft-k8-attn+mix` | 0.605 ± 0.020 | **0.573 ± 0.019** (best of the ladder) |
| HTS-AT `ft-k2-meanmax` (stage 4 only) | 0.585 ± 0.014 | 0.527 ± 0.007 |

[Fact] Fine-tuning helps HeAR (+0.057) and AST (+0.02 over its frozen head 0.547); it does not help HTS-AT (0.527 vs 0.534 frozen). No paired CIs for cell-to-cell differences of 0.01-0.02.

#### P0-4.3 Patch-Mix CL reproduction (5 seeds)

| Protocol | Score | Published |
|---|---|---|
| Best epoch on test (paper protocol, optimistic) | 61.49 ± 0.91 (seeds 61.18, 62.41, 60.60, 62.51, 60.76) | 62.37 ± 0.61 |
| Last epoch (no test selection) | 59.59 ± 0.47 | plain fine-tuning 59.55 ± 0.88 |

[Fact] Reproduction accepted (0.88 below the published mean, inside the +-1 point target). Part I criterion (best rung within 1 SD of or above 62.37 under the val-selected protocol) is **not met**: 57.3 (AST) is 4.2 below the test-selected and 2.3 below the last-epoch reproduction. [Interpretation] Plausible cause: the cached-token design leaves the lower blocks frozen; not tested by a full fine-tune in our protocol.

#### P0-4.4 Part II: device shift (V1 coverage, nominal 0.90)

E1, ICBHI one device held out (AST L0 / L1 / L2; HeAR L2):

| Held-out device (cycles) | AST L0 | AST L1 | AST L2 | HeAR L2 |
|---|---|---|---|---|
| AKGC417L (1836) | 0.97 | 0.80 | 0.83 | 0.87 |
| Litt3200 (461) | 0.47 | 0.97 | 0.95 | 0.91 |
| Meditron (459) | 0.99 | 0.96 | 0.93 | 0.92 |

Binomial half-width +-0.014 (1836), +-0.027 (about 460). [Fact] Coverage is below nominal beyond the half-width on AKGC417L and Litt3200 for most rungs, not on Meditron. Whether this is device or label shift is **not separated** (V4 not run on these pairs). E2 (KAUH filter pairs, 30 units per rung): AST L1 0.892 to 0.914, L2 0.904 to 0.927; OPERA probes drop to 0.49 (CT) and 0.45 (CE). Decodability AUC 0.9-1.0 saturates and cannot rank rungs. [Fact] Fine-tuning did not raise decodability or the coverage deficit (AST 0.904 vs 0.910; HeAR 0.900 vs 0.869), reported as negative finding F4.

#### P0-4.5 Stage 1 screening of device variants (3-fold patient CV, 12 of 24 runs)

Validation-selected epoch on held-out training patients, mean over 3 folds:

| Variant | Folds 0 / 1 / 2 | Mean CV Score | Sp / Se | Worst-device | Fold SD |
|---|---|---|---|---|---|
| P0 baseline | 64.1 / 63.5 / 56.2 | 61.3 | 84.1 / 38.4 | 54.4 | 4.3 |
| P1 A1 input | 63.3 / 66.1 / 57.9 | **62.4** | 82.0 / 42.9 | **58.2** | 4.2 |
| P2 bin norm | 62.0 / 65.1 / 55.7 | 60.9 | 82.0 / 39.8 | 55.5 | 4.8 |
| P3 random bin gain | 60.3 / 64.1 / 55.5 | 60.0 | 87.7 / 32.2 | 55.1 | 4.3 |
| P4, P6, P7, P8, P1P3 | not run | | | | |

[Fact] P1 leads P0 by 1.1 (mean) and 3.8 (worst device) but fold SD is about 4, so no variant is distinguishable yet; fold 2 is the hardest for all. P4/P6/P7 launched with a runner that passed `--cv_fold` before `main.py` accepted it (exit at argument parsing, 0 epochs; fixed since in `aeedd4e`); P8 never started. Smoke run (1 epoch, all flags on) passed end to end, so the variant code paths work. Scores in `baselines/patchmix_cl/save/results.json`, logs in `baselines/patchmix_cl/logs/screen_*.log`.

#### P0-4.6 Earlier v4-era findings (full tables in the archive)

[Fact] Short-window branch: no gain (H1 negative). Hard phase pooling: improved AST crackle F1 on HF_Lung only. Tent: no reliable calibration gain. Conformal sets average 2.5-3.5 of 4 labels with OPERA-CT, 1.85 with AST.

### P0-5. Caveats and open issues

- **Prereg timing.** `prereg-v9-amendment-1.md` (registered 2026-10-07) states that no Stage-1 run had started. File timestamps of the screening runs (local +07: P0 fold 0 01:23, P1 03:24, P2 06:29, P3 08:31 on 2026-10-07) are earlier than or close to the commit of the amendment (`d9c9419`, 04:24 UTC = 11:24 +07), so P0-P3 started before the amendment was committed. The primary hypotheses are unaffected by P0-P3 (no test access), but the amendment sentence should be corrected or the runs declared exploratory. [Assumption] The timestamps are in one timezone; check `logs/screening_driver.log`.
- **HTS-AT** (committed `9bccec8`) is not in the v8/v9 variant pool; its numbers above come from `outputs/ladder_htsat/ladder-htsat-v1`.
- **A1/A2 in extraction** are tested on stubs only; no corrected feature cache exists, so H3 (correction closes the deficit) is untested.
- **ICBHI device is confounded** with class prior and site; E1 deficits are not decomposed.
- **KAUH** test sets hold about 35 windows per partition; differences below 0.1 in coverage are inside binomial noise, and no CIs were computed for E2.
- Patch-Mix runs are not deterministic (`cudnn.benchmark`, AMP): seed SD is the right comparison, not run-to-run equality.



## 6. Stage 1: Registered core (v9, AST), from `docs/reports/2026-10-09-registered-core-report.md`

[Fact] Nghiên cứu đã đăng ký trên AST, 09/10/2026. Gồm so sánh sáu encoder, hai kiểm tra lựa chọn (CV 3 fold, rồi refit và kiểm thử một lần), cổng G (10 seed),
LODO bốn thiết bị (baseline, Freq-MixStyle, SC đã train, SC lúc kiểm thử, A2, TTA-EQ) và kiểm soát dương KAUH. Tóm tắt:

| Item | Result |
| --- | --- |
| Patch-Mix CL reproduction (AST, best epoch on test) | 61.49 ± 0.91 vs 62.37 ± 0.61 published; last epoch 59.59 ± 0.47 (Phase 0 runs; the registered `repro-wave1` runs give 61.4 ± 1.3, section R3.1) |
| Refit at the CV epoch (seeds 0-4) | AuscultTrust (SC) 58.31 ± 1.08; baseline 56.80 ± 1.49; SC + random gain 59.72 (not selected by the registered rule) |
| Gate G (10 seeds) | Paired difference +1.91, 95% CI [+0.31, +3.50]: passes (lower bound > -1.5). Superiority not shown (patient bootstrap [-0.83, +4.84]) |
| LODO on AST (H2) | Only Litt3200 passes the null gate. Prior-matched coverage 0.670 (baseline) vs 0.904 (SC); D = +0.177 [+0.055, +0.326]; set size 1.92 vs 2.67 labels. Freq-MixStyle D = +0.126 [-0.006, +0.269] |
| KAUH control A | Test-time SC on baseline weights lowers the cross-filter flip rate 0.188 to 0.104 (+0.085 [+0.041, +0.132]); AuscultTrust weights do not (0.175). Control B2 not interpretable (no filter pair beyond the chance range) |
| H1, H3 | [Pending]: phantom recordings |

[Interpretation] H2 failed only on set size, so the LODO v1 analysis reads set size next to abs(Delta), and ensembles (sharper softmax) may shrink sets.

Toàn văn các bảng (nguyên văn từ `docs/reports/2026-10-09-registered-core-report.md`; các tiêu đề được gắn tiền tố `R`):

### R3. Part 1: encoder comparison

#### R3.1 Results (Sp, Se, Score, HS; best epoch on test, five seeds)

| Encoder | Mode | Method | Seeds | Sp | Se | **Score** | HS | 2-class Score | Score at epoch 50 |
|---|---|---|---|---|---|---|---|---|---|
| AST | fine-tune | Patch-Mix CL | 5 | 78.1±5.8 | 44.8±5.1 | **61.4±1.3** | 56.6±3.0 | 67.9±1.3 | 58.5±1.3 |
| AST | fine-tune | Patch-Mix | 5 | 79.2±5.2 | 39.8±5.1 | **59.5±0.6** | 52.7±3.5 | 66.0±1.9 | 56.6±0.8 |
| HTS-AT | fine-tune | Patch-Mix CL | 5 | 76.1±3.6 | 46.3±2.8 | **61.2±0.5** | 57.5±1.1 | 67.5±0.3 | 58.5±0.8 |
| HTS-AT | fine-tune | Patch-Mix | 5 | 77.4±3.8 | 44.9±2.8 | **61.2±0.9** | 56.7±1.3 | 68.2±0.8 | 59.1±0.6 |
| DASS | fine-tune | Patch-Mix CL | 5 | 71.8±3.8 | 49.2±2.0 | **60.5±1.2** | 58.3±0.9 | 67.1±1.3 | 57.9±0.9 |
| DASS | fine-tune | Patch-Mix | 5 | 75.6±3.6 | 47.5±2.5 | **61.6±1.3** | 58.3±1.4 | 67.6±1.0 | 59.7±1.2 |
| AST | frozen | Patch-Mix CL | 5 | 68.8±1.8 | 44.3±1.8 | **56.6±0.3** | 53.9±0.8 | 63.9±0.7 | 55.1±0.4 |
| AST | frozen | Patch-Mix | 5 | 76.7±3.4 | 37.9±2.7 | **57.3±0.7** | 50.7±1.9 | 62.9±0.9 | 56.5±0.5 |
| HTS-AT | frozen | Patch-Mix CL | 5 | 71.6±3.3 | 34.6±3.2 | **53.1±0.3** | 46.5±2.2 | 60.0±0.8 | 51.5±0.4 |
| HTS-AT | frozen | Patch-Mix | 5 | 79.8±4.7 | 28.5±6.0 | **54.2±1.1** | 41.5±6.0 | 59.9±3.1 | 50.1±0.4 |
| DASS | frozen | Patch-Mix CL | 5 | 69.6±2.5 | 40.8±2.4 | **55.2±0.2** | 51.3±1.2 | 61.5±0.6 | 54.1±0.2 |
| DASS | frozen | Patch-Mix | 5 | 75.9±3.1 | 36.6±2.9 | **56.2±0.4** | 49.2±2.1 | 61.0±0.2 | 54.6±0.4 |
| CLAP | frozen | Patch-Mix CL | 5 | 67.8±4.4 | 39.8±3.6 | **53.8±0.5** | 49.9±1.6 | 61.3±0.5 | 52.6±0.3 |
| CLAP | frozen | Patch-Mix | 5 | 71.2±3.5 | 38.7±3.2 | **54.9±0.4** | 50.0±1.9 | 61.8±0.6 | 54.1±0.4 |
| HeAR | frozen | Patch-Mix CL | 5 | 87.5±8.2 | 20.0±8.2 | **53.7±1.0** | 31.6±10.5 | 56.0±1.9 | 51.1±0.2 |
| HeAR | frozen | Patch-Mix | 5 | 82.0±8.2 | 24.5±7.6 | **53.3±0.8** | 36.8±8.8 | 56.3±1.5 | 51.3±0.2 |
| OPERA-CT | frozen | Patch-Mix CL | 5 | 86.9±8.6 | 12.3±9.1 | **49.6±0.3** | 20.2±13.7 | 52.1±1.7 | 46.6±0.3 |
| OPERA-CT | frozen | Patch-Mix | 5 | 84.4±5.6 | 18.6±4.5 | **51.5±0.7** | 30.1±5.6 | 54.7±1.4 | 48.6±0.7 |

Per-class recall at the best-on-test epoch (mean ± SD, %):

| Encoder | Mode | Method | Normal | Crackle | Wheeze | Both |
|---|---|---|---|---|---|---|
| AST | fine-tune | Patch-Mix CL | 78.1±5.8 | 57.3±6.6 | 34.1±7.0 | 16.9±10.5 |
| AST | fine-tune | Patch-Mix | 79.2±5.2 | 50.7±3.8 | 30.6±6.1 | 15.4±14.7 |
| HTS-AT | fine-tune | Patch-Mix CL | 76.1±3.6 | 55.2±5.8 | 38.7±3.4 | 26.7±6.5 |
| HTS-AT | fine-tune | Patch-Mix | 77.4±3.8 | 54.7±3.0 | 35.6±4.8 | 25.3±10.6 |
| DASS | fine-tune | Patch-Mix CL | 71.8±3.8 | 52.6±2.1 | 53.7±4.2 | 21.4±3.5 |
| DASS | fine-tune | Patch-Mix | 75.6±3.6 | 51.3±3.4 | 48.7±2.5 | 26.9±4.4 |
| AST | frozen | Patch-Mix CL | 68.8±1.8 | 54.5±1.5 | 37.0±5.8 | 18.0±4.5 |
| AST | frozen | Patch-Mix | 76.7±3.4 | 53.8±3.2 | 19.7±4.5 | 15.0±4.7 |
| HTS-AT | frozen | Patch-Mix CL | 71.6±3.3 | 41.7±7.1 | 33.4±4.3 | 6.0±6.3 |
| HTS-AT | frozen | Patch-Mix | 79.8±4.7 | 38.0±11.1 | 22.6±4.2 | 1.0±1.1 |
| DASS | frozen | Patch-Mix CL | 69.6±2.5 | 45.3±3.3 | 42.0±3.2 | 16.9±3.6 |
| DASS | frozen | Patch-Mix | 75.9±3.1 | 43.1±2.2 | 35.3±7.2 | 10.3±4.4 |
| CLAP | frozen | Patch-Mix CL | 67.8±4.4 | 48.3±4.4 | 35.9±7.5 | 11.3±5.1 |
| CLAP | frozen | Patch-Mix | 71.2±3.5 | 48.1±4.5 | 35.0±2.6 | 5.7±2.1 |
| HeAR | frozen | Patch-Mix CL | 87.5±8.2 | 35.9±14.6 | 0.8±1.6 | 0.0±0.0 |
| HeAR | frozen | Patch-Mix | 82.0±8.2 | 39.5±10.2 | 7.5±6.8 | 2.2±2.3 |
| OPERA-CT | frozen | Patch-Mix CL | 86.9±8.6 | 20.5±16.3 | 2.3±1.4 | 1.5±1.3 |
| OPERA-CT | frozen | Patch-Mix | 84.4±5.6 | 33.1±7.8 | 0.9±0.8 | 0.3±0.6 |

#### R3.2 Findings

- **F1.1 Fine-tuning [Fact].** AST, HTS-AT and DASS lie within about 1 point (60.5-61.6; seed SD 0.5-1.3): the data do not rank them. Patch-Mix CL helps AST by about 2 points (61.4 vs 59.5); it does not help HTS-AT or DASS (DASS is 1.1 points higher with Patch-Mix alone).
- **F1.2 Frozen encoders [Fact].** AST 56.6-57.3 > DASS 55.2-56.2 > CLAP 53.8-54.9, HTS-AT 53.1-54.2, HeAR 53.3-53.7 > OPERA-CT 49.6-51.5. For the same encoder, fine-tuning gains 4-8 points over freezing.
- **F1.3 Class balance [Fact].** HeAR and OPERA-CT reach their best-on-test Score through high Sp and very low Se (12-25) and almost never predict wheeze or both. DASS has the best wheeze recall (49-54) and the best Se (47-49) among fine-tuned encoders.
- **F1.4 Epoch selection [Fact].** Best-on-test epochs are often 1-10 and the Score at epoch 50 is about 3 points lower than the best-on-test value for fine-tuned runs. Published best-on-test numbers overstate what a fixed training length gives. This is why Part 2 fixes the epoch by cross-validation.
- **F1.5 Why "high Sp, low Se" appears [Fact + Interpretation].** Three effects combine.
  - *Class structure.* Normal is 49.8% of the training cycles and 57.3% of the test cycles, whereas Se needs the correct class among three abnormal classes (crackle 29%, wheeze 12%, both 9% of training). Sp is structurally easier than Se: among fine-tuned encoders the best Sp is 79.2 and the best Se 49.2.
  - *Epoch choice.* The best-on-test epoch is often 1-10, where the head still favours the majority class. Score = (Sp + Se) / 2 does not penalise this imbalance; HS does. HeAR's best-on-test row reads Sp 87.5 / Se 20.0; at epoch 50 the same runs give Sp 59.6 / Se 42.5. The high Sp of HeAR and OPERA-CT in Section 3.1 is therefore a property of the selected epoch, not of the model.
  - *Weak embeddings (frozen encoders).* The abnormal-vs-normal AUROC of the saved test softmax at epoch 50 (threshold-free; seeds 0-4; computed with `src.evaluation.fm_diagnose`, diagnosis only) is 0.55 for OPERA-CT, 0.62 HeAR, 0.64 HTS-AT, 0.65 CLAP, 0.66 DASS and 0.70 AST. These are well below what a pure threshold problem would show, so moving the threshold would trade Sp for Se with little net gain. OPERA-CT is the extreme case: 66% of cycles are predicted normal and Se is 25.
  - HeAR's input length is not a cause: an 8 s cycle becomes four 2 s clips that are all encoded and pooled.
  - *Consequence for reading Section 3.1:* compare encoders by Score and HS together and trust the epoch-50 column for what a fixed training length gives; the frozen foundation models are not "good normal detectors".

#### R3.3 Comparison with the reference papers

| Setting | Paper | Ours | Gap |
|---|---|---|---|
| AST, Patch-Mix only | 59.46 ± 0.78 (ablation of Patch-Mix CL) | 59.5 ± 0.6 | 0.0 |
| AST, Patch-Mix CL | 62.37 ± 0.61 (Sp 81.66, Se 43.07) | 61.4 ± 1.3 (Sp 78.1, Se 44.8) | -1.0 |
| AST, Patch-Mix CL, 2-class | 68.71 ± 0.59 | 67.9 ± 1.3 | -0.8 |
| DASS, CE fine-tune | 61.06 ± 1.27 (Lung-SRAD) | not run (we use Patch-Mix) | n/a |
| DASS, regularisation + AST-style Patch-Mix CL | 62.42 ± 0.33 | 60.5 ± 1.2 (without regularisation) | not like for like |
| DASS, Lung-SRAD | 64.48 ± 0.25 | not run | n/a |

[Fact] The AST reproduction is credible (Patch-Mix only matches; Patch-Mix CL is about 1.4 standard errors below, inside the 1.0-point acceptance window set beforehand). Our DASS run is plain DASS and cannot be compared with the regularised variants. SG-SCL (61.71), LungAdapter (62.40), Audio-CLAP (62.56), BTS (63.54) and PAFA (64.84) have no matching run here. No paper reports frozen HTS-AT, HeAR, OPERA-CT, CLAP or DASS on ICBHI 4-class: those rows are our own measurements.

### R4. Part 2: AuscultTrust (spectrum correction in the training pipeline)

#### R4.1 Method and arms

- **SC.** STFT (n_fft 1024, hop 512, 16 kHz); each device gets a coefficient curve that maps its mean spectrum to a common reference (arithmetic mean of the training devices' mean spectra); the device is read from the file name; the target-device spectrum comes from unlabelled audio only. Two coefficient bounds: `dynamic` (every bin clipped to +-20 dB) and `static` (50-2000 Hz unclipped, no-op outside), chosen per arm by cross-validation (tie: dynamic).
- **Arms** (code names): baseline (Patch-Mix CL, no SC); `sc` (SC); `sc_gain` (SC plus random per-bin gain, 6 dB SD); `freq_mixstyle` (Freq-MixStyle, p = 0.5; competitor). AuscultTrust = the better of `sc` and `sc_gain` by mean CV Score (tie: `sc`).
- **Gate G** (registered): AuscultTrust must not lose more than 1.5 Score points to the baseline: lower bound of the 95% t-interval of the paired Score difference over seeds > -1.5 (registered: seeds 0-9; seeds 0-4 are the benchmark report).

#### R4.2 Cross-validation screening (3 folds, registered rule)

Three-fold patient-grouped cross-validation on the training patients (test set unused); every SC arm was run with both bounds and the better bound kept.

| Arm | Cell kept | Bound | CV epoch | Mean CV Score | Fold Scores | Gap to baseline |
|---|---|---|---|---|---|---|
| **SC (selected: AuscultTrust)** | sc + static | static | 14 | **60.51** | 52.2, 66.9, 62.3 | +1.54 |
| SC + random gain | sc_gain | dynamic | 2 | 60.01 | 54.5, 65.2, 60.4 | +1.04 |
| Baseline | baseline | - | 2 | 58.97 | 52.9, 64.0, 60.0 | 0 |
| Freq-MixStyle | freq_mixstyle | - | 3 | 58.71 | 52.4, 61.3, 62.5 | -0.26 |

Other cells: SC with the dynamic bound 58.44 (epoch 26); SC + random gain with the static bound 59.36 (epoch 1).

[Interpretation] The registered rule selects SC with the static bound. Caveats: only three folds with fold Scores spanning 52-67, so gaps of 1-1.5 points are inside the spread; the best CV epoch is 1-3 for several arms, which points to noise peaks; averaged over epochs 10-50 the paired gaps to the baseline are +1.3 (SC static), +1.7 (SC + random gain, static), -0.1 (SC dynamic), -0.2 (SC + random gain, dynamic) and +0.5 (Freq-MixStyle). The static bound looks consistently useful; the dynamic bound and the random gain do not. The rule was not changed after seeing this. The screening report files did not store Sp, Se and HS per epoch (only Score); every run from now on logs all four metrics each epoch (Section 4.4).

#### R4.3 Final refit on the test set (seeds 0-4, epoch fixed by Section 4.2)

Protocol. All four arms are refit on all training patients and scored once on the official test set at the epoch fixed in Section 4.2 (`--selection fixed`): baseline 2, Freq-MixStyle 3, SC (static bound) 14, SC + random gain 2. Benchmark report: seeds 0-4. Gate G uses seeds 0-9 for the baseline and for SC, the two arms it compares (the other two arms have seeds 0-4 only). The last column repeats the best-on-test epoch of the same runs for comparison with the papers (**optimistic**, not a result).

**Pooled test set (mean ± SD over seeds 0-4)**

| arm | cell | CV epoch | seeds | Sp | Se | Score | HS | macro-F1 | 2-cls Score | best-on-test Score (optimistic) |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | baseline | 2 | 5 | 78.15 ± 9.06 | 35.45 ± 8.19 | 56.80 ± 1.49 | 47.78 ± 6.76 | 0.41 ± 0.03 | 62.76 ± 2.29 | 61.11 ± 0.45 |
| freq_mixstyle | freq_mixstyle | 3 | 5 | 66.69 ± 9.26 | 45.01 ± 10.67 | 55.85 ± 1.16 | 52.29 ± 4.78 | 0.44 ± 0.04 | 64.10 ± 2.77 | 61.18 ± 0.55 |
| sc (AuscultTrust) | sc+sc_static | 14 | 5 | 71.84 ± 4.79 | 44.77 ± 2.91 | 58.31 ± 1.08 | 54.99 ± 0.95 | 0.46 ± 0.01 | 65.94 ± 0.93 | 60.64 ± 0.60 |
| sc_gain | sc_gain | 2 | 5 | 84.69 ± 5.50 | 34.75 ± 5.36 | 59.72 ± 1.15 | 48.90 ± 4.16 | 0.43 ± 0.03 | 65.08 ± 1.97 | 61.80 ± 0.49 |

**Paired difference against the baseline (same seeds)**

| arm | seeds | ΔScore mean | 95% t-CI (paired seeds) | 95% patient bootstrap CI |
|---|---|---|---|---|
| freq_mixstyle | 5 | -0.95 | [-2.19, 0.29] | [-2.65, 0.77] |
| sc | 10 | +1.91 | [0.31, 3.50] | [-0.83, 4.84] |
| sc_gain | 5 | +2.92 | [0.95, 4.89] | [0.70, 5.08] |

**Gate G.** SC (AuscultTrust) against the baseline over ten seeds: Delta Score = +1.91, 95% t-interval [+0.31, +3.50]; lower bound 0.31 > -1.5, so the **gate passes**. The patient bootstrap interval [-0.83, +4.84] (test patients resampled, same seeds) also stays above -1.5.

**Per device (official test set: AKGC417L 1,836, Litt3200 461, Meditron 459 cycles; LittC2SE has no test cycles), mean ± SD over seeds 0-4**

**Sp**

| arm | Meditron | Litt3200 | AKGC417L |
|---|---|---|---|
| baseline | 95.80 ± 2.54 | 43.88 ± 30.17 | 84.11 ± 9.40 |
| freq_mixstyle | 92.61 ± 3.52 | 18.95 ± 7.42 | 74.53 ± 15.15 |
| sc | 92.48 ± 2.73 | 67.38 ± 20.58 | 66.49 ± 6.49 |
| sc_gain | 95.41 ± 2.65 | 70.71 ± 18.30 | 85.94 ± 5.94 |

**Se**

| arm | Meditron | Litt3200 | AKGC417L |
|---|---|---|---|
| baseline | 15.86 ± 5.62 | 45.59 ± 22.85 | 37.08 ± 7.66 |
| freq_mixstyle | 27.03 ± 15.13 | 63.82 ± 5.31 | 45.07 ± 11.33 |
| sc | 25.93 ± 8.41 | 36.03 ± 13.95 | 49.15 ± 2.23 |
| sc_gain | 14.21 ± 5.49 | 32.21 ± 18.66 | 38.46 ± 5.95 |

**Score**

| arm | Meditron | Litt3200 | AKGC417L |
|---|---|---|---|
| baseline | 55.83 ± 1.92 | 44.73 ± 4.75 | 60.59 ± 2.47 |
| freq_mixstyle | 59.82 ± 5.84 | 41.39 ± 2.42 | 59.80 ± 2.39 |
| sc | 59.21 ± 3.06 | 51.71 ± 3.86 | 57.82 ± 2.39 |
| sc_gain | 54.81 ± 2.29 | 51.46 ± 1.52 | 62.20 ± 1.17 |

**HS**

| arm | Meditron | Litt3200 | AKGC417L |
|---|---|---|---|
| baseline | 26.84 ± 8.21 | 31.93 ± 12.19 | 50.61 ± 6.11 |
| freq_mixstyle | 39.81 ± 17.16 | 28.34 ± 8.51 | 53.99 ± 4.76 |
| sc | 39.84 ± 9.74 | 42.67 ± 5.64 | 56.31 ± 1.41 |
| sc_gain | 24.37 ± 8.07 | 38.94 ± 13.16 | 52.70 ± 4.65 |

Reading.

- **[Fact] Epoch cost.** At the fixed CV epoch the baseline scores 56.80 ± 1.49, 4.3 points below its best-on-test value (61.11 ± 0.45), because the CV epoch is 2 and the model is still strongly biased towards normal (Sp 78.2 ± 9.1, Se 35.5 ± 8.2). The best-on-test column is what the papers report; the fixed-epoch column is what a deployable choice gives.
- **[Fact] Ranking at the fixed epoch.** SC + random gain 59.72 > SC 58.31 > baseline 56.80 > Freq-MixStyle 55.85. Seed SD is 1.1-1.5, so only the paired differences are informative: SC +1.91 (ten seeds), SC + random gain +2.92 (five seeds, patient bootstrap [+0.70, +5.08]), Freq-MixStyle -0.95 (interval includes zero).
- **[Fact] Selection and test ranking disagree.** The registered rule picked SC (CV gap +1.54 against +1.04 for SC + random gain). On the test set SC + random gain is higher. The rule is not changed after seeing this; both are reported. SC + random gain reaches its Score through Sp (84.7) with Se unchanged (34.8 against 35.5), whereas SC reaches it through Se (44.8) at lower Sp (71.8).
- **[Fact] Balance and stability.** SC has the highest HS (55.0 ± 0.95 against 47.8 ± 6.8 for the baseline) and the smallest seed SD of Se (±2.9) and HS. [Interpretation] HS and Score tell different stories here: Score rewards Sp of the larger normal class, HS rewards balance.
- **[Fact + Interpretation] High Sp, low Se in the refit.** At the fixed epoch the baseline has Sp 78.2, Se 35.5 and predicts normal for 67.3% of the test cycles (57.3% are normal). The arms differ mainly in where they sit on the Sp-Se trade-off: SC + random gain moves further to Sp (84.7 / 34.8; 71.8% predicted normal), while SC (44.8 Se; 58.2%) and Freq-MixStyle (45.0 Se; 54.6%) move towards Se. The threshold-free abnormal-vs-normal AUROC of the same softmax (seeds 0-4) differs little: baseline 0.71 ± 0.02, Freq-MixStyle 0.70 ± 0.03, SC 0.72 ± 0.01, SC + random gain 0.73 ± 0.01. So most of the Score gap between arms (up to 3 points) is a shift along the trade-off, not a better separation of abnormal sounds. What SC adds beyond the shift is a smaller seed spread and a balanced operating point (HS +7.2). This is why the gate result is stated as non-inferiority and not as improvement.
- **[Fact] Devices.** SC raises Score on Litt3200 (+7.0: 51.71 vs 44.73) and Meditron (+3.4), and lowers it on AKGC417L (-2.8: 57.82 vs 60.59), where Sp falls from 84.1 to 66.5 while Se rises from 37.1 to 49.2. Litt3200 Sp has a seed SD of 20-30 points for the baseline and SC, so single-device numbers on 461 cycles are unstable.
- **[Interpretation] Gate G passes, non-inferiority is shown; superiority is not.** The paired seed interval for SC excludes zero, the patient bootstrap interval does not. Combined with the early selection epochs (2-3 of 50), the 3-point gaps between rows should not be read as a ranking of methods.

#### R4.4 Supplementary: five-fold screening with Sp, Se and HS

Purpose. The 3-fold screening of Section 4.2 did not record Sp, Se and HS per epoch and its fold Scores are noisy (52-67). This supplementary screen repeats it with five patient-grouped, device-stratified folds and logs Sp, Se, Score and HS every epoch.

Protocol. Cells: `baseline`, `freq_mixstyle`, `sc` with the static bound, `sc_gain` (the bound each arm kept in Section 4.2; the dynamic variants are not repeated). Five folds each = 20 runs. Each run keeps the 50-epoch learning-rate schedule and stops after epoch 20 (`--stop_epoch 20`; the weights equal those of the full schedule at that epoch). The CV epoch is the epoch of the highest mean validation Score over folds, searched within epochs 1-20. The test set is not used. Role: robustness check only; the refit epochs of Section 4.3 were fixed from the 3-fold screen before any test scoring and are not changed by this result.

**Registered rule (best mean-CV epoch per arm, mean over five folds)**

| Rank | Arm | Cell | CV epoch | Sp | Se | Mean CV Score | HS | Fold Scores |
|---|---|---|---|---|---|---|---|---|
| 1 | baseline | baseline | 9 | 78.6 | 42.0 | 60.29 | 53.9 | 59.2, 61.3, 59.8, 56.5, 64.6 |
| 2 | sc (static) | sc+sc_static | 5 | 79.1 | 40.0 | 59.55 | 50.4 | 62.2, 51.3, 58.9, 59.4, 66.0 |
| 3 | sc_gain | sc_gain | 13 | 75.2 | 43.7 | 59.46 | 54.8 | 58.9, 60.3, 59.9, 55.9, 62.3 |
| 4 | freq_mixstyle | freq_mixstyle | 10 | 76.7 | 42.0 | 59.37 | 53.9 | 58.8, 54.0, 62.1, 60.3, 61.7 |

**Less noise-prone summary: mean over epochs 10-20 (all folds), and paired gap to the baseline**

| Arm | Sp | Se | Score | HS | Gap to baseline (Score) | Gap per fold | SD of the gap over folds |
|---|---|---|---|---|---|---|---|
| baseline | 77.3 | 36.6 | 56.97 | 47.9 | 0 | | |
| sc (static) | 76.7 | 38.6 | 57.67 | 49.8 | +0.70 | +2.8, +2.4, -1.7, +1.7, -1.7 | 2.2 |
| sc_gain | 76.5 | 39.5 | 58.01 | 51.0 | +1.04 | +0.2, +4.9, +0.8, +0.8, -1.5 | 2.3 |
| freq_mixstyle | 77.3 | 37.8 | 57.53 | 49.5 | +0.56 | +0.6, +3.0, +0.2, +0.6, -1.6 | 1.6 |

Reading.

- **[Fact] The 3-fold ranking is not reproduced.** At the best epoch the baseline is highest (60.29) and SC is 0.74 lower (59.55), against a gap of +1.54 in favour of SC in the 3-fold screen. All four arms lie within 0.9 points while the fold Scores span 51-66, so the screen does not separate the arms.
- **[Fact] The best-epoch value is noise-prone.** Mean CV curves move by 2-3 points between nearby epochs (SC: 59.55 at epoch 5, 56.28 at epoch 15), and taking the maximum over 20 epochs per arm favours whichever curve has the largest noise peak. The mean over epochs 10-20 is steadier: the three non-baseline arms are 0.6-1.0 points above the baseline, each gap is about one third to one half of the fold SD of the gap (1.6-2.3), and one or two of the five folds go the other way. [Interpretation] The direction agrees with the epoch-averaged gaps of the 3-fold screen (+1.3 SC static, +1.7 SC + gain), but neither screen shows an advantage that exceeds the fold spread.
- **[Fact] Sp and Se.** In epochs 10-20 the baseline has the lowest Se (36.6) and every other arm has higher Se (37.8-39.5) at equal or slightly lower Sp (76.5-77.3), so HS is 1.6-3.1 points higher. This is the same pattern as in Section 4.3: the arms differ mainly in how they trade Sp against Se.
- **[Fact] Epoch sensitivity.** The five-fold CV epochs are 9 (baseline), 5 (SC), 13 (SC + gain) and 10 (Freq-MixStyle), against the registered 2, 14, 2 and 3. The refit epochs 2 and 3 are early; this is consistent with the 4.3 observation that the baseline at epoch 2 is 4.3 points below its best-on-test value. A later fixed epoch would probably raise the baseline and the other arms alike; this was not tested on the test set and the registered epochs are not changed.
- **[Interpretation] What this does and does not change.** Gate G (non-inferiority of AuscultTrust, Section 4.3) is a test-set result and stands. The cross-validation evidence for an advantage of SC over the baseline is weak in both screens, so the report does not claim one.

### R5. Leave-one-device-out (LODO), AST

#### R5.1 Protocol

- **Split.** All cycles of one held-out device (Meditron, LittC2SE, Litt3200 or AKGC417L) are the target. A patient with any recording on the held-out device leaves training and calibration (their held-out-device recordings stay in the target); the two patients that the official split puts on both sides (156, 218) are excluded. 20% of the patients of each source device form the calibration set (it changes with the seed); the rest train.
- **Training.** Recipe and epoch frozen from Section 4.2 (baseline 2, Freq-MixStyle 3, SC with static bound 14); the run stops at that epoch (identical weights, a fraction of the compute). Five seeds. Arms: baseline, Freq-MixStyle, AuscultTrust (SC, static).
- **Conformal.** Split conformal with the LAC score, alpha = 0.1, 4-class sets on cycles. Delta = 0.90 minus prior-matched coverage, so positive means under-coverage (test cycles are importance-weighted to the calibration class mix, which removes the class-mix part of the error); plain coverage is listed next to it.
- **Null gate.** A fold is tested only if the baseline |Delta| exceeds the 95th percentile of |Delta| over 200 exchangeable re-splits of the source-device patients with the same sizes; otherwise the coverage error is within chance.
- **H2 statistic.** D = mean over tested folds of |Delta_baseline| - |Delta_arm|; 95% two-level bootstrap interval (seeds, patients), with recalibration inside every draw.

#### R5.2 Results (mean over five seeds)

| Held-out device | Arm | Seeds | Test cycles | Sp | Se | Score | HS | Coverage | Delta | Set size | Singleton | Empty |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Meditron | baseline | 5 | 1370 | 89.8 | 28.6 | 59.2 | 43.1 | 0.932 | +0.002 | 1.88 | 0.21 | 0.000 |
| Meditron | sc+sc_static | 5 | 1370 | 72.3 | 38.6 | 55.5 | 48.0 | 0.924 | +0.015 | 2.52 | 0.17 | 0.000 |
| Meditron | freq_mixstyle | 5 | 1370 | 88.3 | 32.3 | 60.3 | 46.5 | 0.934 | -0.007 | 1.90 | 0.22 | 0.000 |
| LittC2SE | baseline | 5 | 594 | 92.9 | 33.9 | 63.4 | 48.9 | 0.902 | +0.024 | 1.93 | 0.27 | 0.000 |
| LittC2SE | sc+sc_static | 5 | 594 | 95.4 | 35.5 | 65.4 | 51.5 | 0.927 | -0.003 | 2.07 | 0.33 | 0.000 |
| LittC2SE | freq_mixstyle | 5 | 594 | 95.6 | 35.8 | 65.7 | 51.6 | 0.902 | +0.021 | 1.89 | 0.30 | 0.000 |
| Litt3200 | baseline | 5 | 481 | 35.9 | 55.6 | 45.8 | 36.3 | 0.785 | +0.230 | 1.92 | 0.23 | 0.000 |
| Litt3200 | sc+sc_static | 5 | 481 | 54.2 | 56.7 | 55.4 | 54.0 | 0.958 | -0.004 | 2.67 | 0.05 | 0.000 |
| Litt3200 | freq_mixstyle | 5 | 481 | 59.8 | 48.6 | 54.2 | 52.4 | 0.900 | +0.104 | 2.08 | 0.13 | 0.000 |
| AKGC417L | baseline | 5 | 4216 | 90.8 | 16.4 | 53.6 | 27.6 | 0.687 | +0.073 | 1.75 | 0.45 | 0.000 |
| AKGC417L | sc+sc_static | 5 | 4216 | 93.8 | 15.4 | 54.6 | 26.3 | 0.707 | +0.063 | 1.87 | 0.42 | 0.000 |
| AKGC417L | freq_mixstyle | 5 | 4216 | 83.4 | 23.2 | 53.3 | 35.4 | 0.704 | +0.074 | 1.79 | 0.42 | 0.000 |

Null gate:

| Device | Baseline abs Delta | Null p95 | Tested |
|---|---|---|---|
| Meditron | 0.029 | 0.143 | no |
| LittC2SE | 0.056 | 0.111 | no |
| Litt3200 | 0.230 | 0.139 | yes |
| AKGC417L | 0.073 | 0.120 | no |

H2 statistic D (tested fold: Litt3200 only; without AKGC417L the value is the same because AKGC417L is not tested):

| Arm | D | 95% interval |
|---|---|---|
| sc+sc_static (AuscultTrust) | +0.177 | [+0.055, +0.326] |
| freq_mixstyle | +0.126 | [-0.006, +0.269] |

#### R5.3 Reading

- **[Fact]** Only Litt3200 has a coverage error beyond chance. There the baseline under-covers (plain coverage 0.785, prior-matched coverage 0.670; Sp 35.9: a device-specific shift). SC restores prior-matched coverage to 0.904 (|Delta| 0.004) and raises Score by 9.6 points and Sp by 18.3; Freq-MixStyle raises Score by 8.4 and leaves Delta = +0.104 (prior-matched coverage 0.796).
- **[Fact]** The interval of D excludes zero for SC and only just includes it for Freq-MixStyle. The two arms are not separated from each other (intervals overlap).
- **[Fact, cost]** SC buys coverage with larger sets on Litt3200 (2.67 labels against 1.92; singleton share 0.05 against 0.23). H2 requires "set size not larger"; this condition is not met on the tested fold. Plain coverage under SC is 0.958, above the nominal 0.90, because the larger sets also cover the class mix that the prior matching removes.
- **[Fact]** On Meditron SC loses 3.7 Score points (55.5 vs 59.2) and 17.5 Sp points. The cause was not investigated (candidates: the Meditron spectrum estimate and the static bound). No coverage claim is made there because the fold is inside chance.
- **[Fact]** AKGC417L (4,216 test cycles, class mix dominated: training keeps few crackle cycles) shows very low Se for every arm (15-23) and no arm difference worth reporting; it is flagged as class-mix dominated.
- **[Fact + Interpretation] "High Sp, low Se" in LODO, and its reverse.** The baseline predicts normal for 83% of Meditron cycles (true normal share 75%), 73% of LittC2SE (58%) and 79% of AKGC417L (45%), giving Sp 90-93 and Se 16-34. On AKGC417L the source devices contain few of its crackle cycles (321 of 1,864 are kept), so the model learns "normal" for that device (Se 16.4): the cause is the class mix, which is why this fold is flagged as class-mix dominated. Litt3200 is the opposite: only 33% are predicted normal against 69% true (Sp 35.9, Se 55.6): the device shift makes normal cycles look abnormal, and this is the fold where coverage fails. The threshold-free AUROC (abnormal vs normal, five seeds) separates ranking from operating point. Baseline: Meditron 0.733, LittC2SE 0.816, Litt3200 0.613, AKGC417L 0.714. SC: 0.696, 0.854, 0.705, 0.665. Freq-MixStyle: 0.730, 0.838, 0.688, 0.697. SC therefore improves ranking on Litt3200 (+0.09) and LittC2SE (+0.04) and worsens it on Meditron (-0.04) and AKGC417L (-0.05). The Litt3200 gain is a ranking gain plus a move of the operating point (predicted normal 33% to 46%); the Meditron loss (Sp 89.8 to 72.3) is SC pushing normal cycles to abnormal while ranking gets worse. Low Se also explains the set sizes: weak ranking needs sets of about 1.9 labels, with a singleton share of 0.2, to reach 90% coverage.
- **[Interpretation]** One tested fold of four is thin evidence for H2: the statistic is an average over a single fold, so the result is a case study of the Litt3200 shift. A second tested fold would need a different device split.
- **[Interpretation]** Score and coverage move together on Litt3200, so the coverage gain may partly reflect better ranking (a higher-Score model gives tighter sets for the same threshold) and not only a calibration effect of SC. The larger sets under SC show that part of the gain is bought with less informative predictions.

### R6. KAUH positive controls (frozen ICBHI models, no KAUH training)

#### R6.1 Data and protocol

KAUH provides, for each of 111 patients, the same recording rendered with three vendor software filters (Bell, Diaphragm, Extended; 333 recordings), so the filters are near-linear versions of one signal. Frozen refit models (seeds 0-4) are applied to 8 s windows; SC (when used) maps each filter's mean spectrum, estimated from its own unlabelled windows, to the saved ICBHI reference. Conditions: **p0** (baseline weights, no SC), **p0_sc** (baseline weights + test-time SC: isolation contrast), **auscult** (AuscultTrust weights + SC).

- **Control A.** The same window under two filters should give the same prediction. Flip rate (4-class argmax differs) and TV distance, averaged over the three filter pairs; reduction against p0 with a 95% bootstrap interval over patients and seeds.
- **Control B2.** Recording-level screening view (normal vs abnormal; p(abnormal) = mean over windows of 1 - p(normal)); 5 patient-rotation groups; calibrate on source-filter recordings of four groups, test on the target-filter recordings of the held-out group; six ordered filter pairs; same null gate and D statistic as Section 5.

#### R6.2 Results

| Condition | Flip rate | TV distance |
|---|---|---|
| p0 | 0.188 ± 0.097 | 0.150 ± 0.049 |
| p0_sc | 0.104 ± 0.052 | 0.081 ± 0.022 |
| auscult | 0.175 ± 0.038 | 0.173 ± 0.026 |

Flip-rate reduction against p0: p0_sc **+0.085** [+0.041, +0.132]; auscult +0.013 [-0.089, +0.103].

B2: for each of the six ordered pairs the baseline mean |Delta| is 0.052-0.099 against a null p95 of 0.120-0.123, so **0 of 6 pairs are tested** and D is undefined (see `outputs/kauh/final/kauh_summary.md` for every pair and condition).

#### R6.3 Reading

- **[Fact]** Control A passes for the isolation contrast: test-time SC on fixed weights nearly halves the flip rate and the TV distance, with an interval away from zero. This is what a near-linear filter difference predicts, so the SC code works.
- **[Fact]** Control A does **not** show the same effect for the AuscultTrust weights: flip rate 0.175 (reduction not distinguishable from zero) and a TV distance above p0 (0.173 vs 0.150). The model trained with SC is no more invariant across the filters than the baseline, although the same test-time SC is applied.
- **[Hypothesis]** A model trained on SC-corrected ICBHI audio may have adapted to the residual spectral shape of the corrected training devices and respond differently to the KAUH correction. This was not investigated (no ablation on SC at train versus test time).
- **[Fact]** B2 has no verdict: the filter pairs do not produce a coverage error larger than chance with these models, so the control cannot discriminate SC from the baseline. It neither passes nor fails.
- **[Fact + Interpretation] "High Sp, low Se" on KAUH comes from the prior, not the filters.** Using p(abnormal) > 0.5 at the recording level, the baseline weights give Sp 78-95 and Se 53-57 per filter and call 54-62% of recordings normal, while only 31.5% of the 111 recordings per filter are normal (68.5% abnormal; ICBHI test cycles are 43% abnormal). The ICBHI-trained head is applied to a population with a different class mix, so Se is low whatever the filter. Ranking is better than the operating point: recording-level AUROC 0.80 (Bell), 0.88 (Diaphragm), 0.89 (Extended). Test-time SC raises the worst filter (Bell AUROC 0.80 to 0.86, Sp 77.7 to 88.0) and slightly lowers the others (Diaphragm 0.88 to 0.86, Extended 0.89 to 0.88): SC makes the filters agree by pulling Bell up, which matches the flip-rate fall in Control A. The AuscultTrust weights give 0.81, 0.80, 0.82, lower than p0 on Diaphragm and Extended, which is consistent with their missing flip-rate gain. The filter-invariance control does not depend on the threshold, so the low Se does not affect its verdict.
- **[Interpretation]** KAUH is a positive control, not evidence about hardware transducers: the filters are software renderings, and the claim of the study about real stethoscopes rests on the pending phantom experiment.

### R7. Hypothesis scoreboard

| Item | Statement | Status |
|---|---|---|
| G | AuscultTrust is not worse than baseline by more than 1.5 Score points | Passes: Delta Score +1.91, 95% interval [+0.31, +3.50] over ten seeds (margin -1.5) |
| H1 | SC lowers the flip rate between devices on the same sound | [Pending] primary (phantom). Control A: isolation contrast passes, AuscultTrust weights do not |
| H2 | SC lowers |Delta| on an unseen device, set size not larger | Partly: lowers |Delta| on the single tested fold (D = +0.177 [+0.055, +0.326]); set size is larger |
| H3 | SC explains at least 80% of the paired log-mel shift between transducers | [Pending] (phantom) |
| Control B2 | Coverage error drops with SC on tested filter pairs | Not interpretable (0 of 6 pairs tested) |



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
4. [Done 28/10, section 17.7] The five recipes have official, LODO and KAUH results; the source-CV gains of class-aware light and LoRA did not transfer. The serving
   model and the main comparison did not change at that time. Plan: `docs/updates/2026-10-27-target-evaluation-plan.md`.

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

### 17.7 Target evaluation (target-20261027) and distillation (distill-20261028), 27-28/10

Detailed numbers: `docs/reports/2026-10-28-target-distillation-results.md`. Exploratory: teacher and ensemble subsets were
chosen on the same test scores.

- [Fact] Official (5 seeds): anchor 63.13; SWA 63.32; multi-scale 62.45; PCSL-5 control 62.16; class-aware light 58.66;
  LoRA 57.05; LoRA + class-aware 54.84. LODO macro (4 devices): 61.62, 61.77, 61.41, 60.72, 60.07, 57.81, 55.11.
  Source-CV gains of class-aware light and LoRA did not transfer.
- [Fact] Equal-weight ensembles of the target arms: all seven 64.21 official / 62.09 LODO macro; G4 (anchor, control,
  class-aware light, multi-scale) 64.11 / 62.22. Subset selection on seeds 0-1 holds on seeds 2-4 (G4: 64.15 / 62.60 vs
  anchor 63.56 / 62.45), a small gain.
- [Fact] KAUH: LoRA and LoRA + class-aware flip rates 0.040 and 0.072 vs 0.209 for the 23/10 AuscultTrust (BEATs + PAFA + static SC; labelled "frozen" in
  the 28/10 report, not a no-SC base), with intervals above zero; no SC or distilled arm survives the flip-reduction bootstrap. B2: no pair is tested when
  that SC model is the base (2 of 6 with the no-SC baseline).
- [Fact] Distillation (student at epoch 66, soft targets from the G4 or G7 teacher ensemble; gate fixed before results):
  kd_g4_k1 official 64.15 (+1.02 vs anchor, 4 of 5 seeds above), kd_g4_k2 63.47 (+0.33), kd_g7_k2 62.75 (-0.38, stopped).
- [Fact] Distilled G4 LODO macro (seeds 0-2): kd_g4_k1 61.72, kd_g4_k2 61.77 vs anchor 61.22 and teacher ensemble 62.26.
  The student keeps about half the ensemble LODO gain; K2 does not beat K1.
- [Interpretation] Teacher targets are in-sample on training cycles, so close to one-hot; this limits the gain.
- [Fact] Recording screening (mean aggregation, Bell): kd_g4_k1 uncertain rate 0.31 vs anchor 0.47; coverage 0.79 vs 0.81.
- [Pending] Not run: KAUH and screening on LODO units, patient-level bootstrap, out-of-fold or augmented-view teacher targets.
- Incident (28/10 20:50): LODO workers on A, C, D and E failed at start (teacher member files only on B). Files were synced
  from B and the four services restarted; the controller reran the jobs.

### 17.8 Campaign `combo-20261028`: class-aware light + multi-scale trong một mô hình (28-29/10)

[Fact] Câu hỏi: hai cải tiến CV nguồn của mục 17.5 (class-aware light, multi-scale) có cộng dồn khi train chung trong một mô hình không?
Cấu hình: recipe anchor (BEATs, PAFA, SC tĩnh, gain6) cộng `class_aware_light.yaml` (`method: ce_pafa_class_aware`, `lambda_pcsl: 5.0`,
`lambda_gpal: 0.0005`) và `multiscale.yaml` (`multiscale: true`: đặc trưng = 0.5 x toàn đoạn + 0.25 x (nửa đầu + nửa cuối)).
Giao thức giống mục 17.5: 5 fold bệnh nhân x 5 seed = 25 fit, 100 epoch, epoch chọn từ đường validation trung bình của chính arm, so với anchor cùng fold và seed.
Chạy trên B, D, E và F (F là máy dùng chung, giới hạn 30 GB). 25/25 fit hoàn tất 29/10 02:27.

| Arm | Epoch | Mean source Score | Delta to anchor | Seed deltas (0-4) |
| --- | ---: | ---: | ---: | --- |
| anchor | 66 | 60.587 | 0 | 0 |
| multi-scale | 68 | 60.266 | -0.321 | -0.76 / +0.24 / +0.52 / -0.89 / -0.72 |
| class-aware light | 4 | 62.045 | +1.458 | +0.88 / +2.53 / +2.33 / +1.93 / -0.38 |
| **class-aware + multi-scale (mô hình ghép)** | 4 | 60.965 | **+0.378** | -0.24 / +0.83 / -0.55 / +0.30 / +1.54 |

- [Fact] Mô hình ghép (+0.378) thấp hơn class-aware light riêng (+1.458) và cao hơn multi-scale riêng (-0.321). 3 trong 5 seed dương, không có seed nào vượt mức class-aware light riêng trừ seed 4. Hai cải tiến **không cộng dồn**.
- [Fact] Epoch được chọn là 4 trên 100, giống class-aware light riêng: mô hình được chọn là mô hình dừng sớm.
- [Interpretation] Lợi ích CV nguồn của class-aware light là một hiệu ứng ở giai đoạn đầu huấn luyện, và nó không chuyển sang đánh giá đích (mục 17.7: official 58.66 so với anchor 63.13). Không có cơ sở để đánh giá đích mô hình ghép; nó chưa được chạy trên split chính thức, LODO hay KAUH.
- [Fact] Kết quả này khớp với arm ghép đã có trước đó: LoRA + class-aware (+0.695) thấp hơn class-aware light riêng (+1.458) ở CV nguồn.
- Điểm yếu của so sánh: lợi ích của class-aware light trộn lẫn hai thay đổi (căn chỉnh theo (bệnh nhân, lớp) và PCSL giảm từ 50 xuống 5); control PCSL-5 (official 62.16, LODO macro 60.72, mục 17.7) là phép tách duy nhất; chỉ hạ PCSL không nâng điểm ở split chính thức (-0.97 so với anchor 63.13).
- Sự cố khi chạy: fit cuối chuyển sang F khiến 3 job trùng chạy trên B, D, E (sweep chỉ đọc danh sách job một lần), controller crash-loop ở bước thu kết quả; xử lý bằng gán lại job đúng máy chứa checkpoint, và thêm tên campaign vào danh sách cho phép giải phóng đĩa của F. Mã huấn luyện không đổi (`programme.json` ghi `amendments`).
- Nguồn: `build/campaigns/combo-20261028/outputs/evaluation/combo-20261028/{summary.json,report.md}` trên B; script `scripts/combo_campaign.py` (worktree `worktree-combo-campaign`).

### 17.9 Đánh giá đóng băng cuối cho student `kd_g4_k1` (run `final-kd-g4-k1`, 29/10)

[Fact] Thực hiện bằng `scripts/final_evaluation_student.py` (bản tham số hóa của `final_evaluation.py`; script 23/10 không bị sửa), 5 seed ghép cặp (0-4), mã và môi trường hiện tại.
Bốn điều kiện: **baseline** (BEATs + PAFA không SC, epoch 100), **previous** (AuscultTrust cũ: BEATs + PAFA + SC tĩnh, epoch 100), **anchor** (`gain6-cv66`, epoch 66), **student** (`kd_g4_k1`, epoch 66).
Đóng băng và chứng nhận: hash trọng số đối chiếu với biên nhận của campaign (`collected-checkpoint.json`, `completion.json`), hash `s_ref` của SC tĩnh, serving seed = 0 ghi trước khi đọc kết quả từng seed
(`outputs/train/final-kd-g4-k1/<điều kiện>/seed<s>/freeze.json`). Không có bundle phục vụ nào được tạo hay kích hoạt. Bootstrap hai tầng (theo bệnh nhân và seed), 1.000 lần, 200 lần chia null.
Export 23/10 của baseline và previous không còn chứng nhận provenance khớp với mã hiện tại (mã `src/` đã đổi sau refactor 27/10), nên được tính lại; các export 23/10 không bị ghi đè.

**Kiểm tra chéo (đạt).** Baseline tái tạo flip 0.445 ± 0.047 và previous 0.209 ± 0.027, bằng số của lần 23/10; sàng lọc bản ghi của previous (độ phủ 0.900, cỡ tập 1.603, uncertain 0.603) cũng khớp lần 23/10.

KAUH-A, tỷ lệ đổi kết luận giữa ba bộ lọc (cửa sổ, 5 seed):

| Điều kiện | Flip rate | TV distance | Giảm flip so với baseline [KTC95%, bootstrap bệnh nhân và seed] |
| --- | ---: | ---: | --- |
| baseline | 0.445 ± 0.047 | 0.437 ± 0.043 | |
| previous | 0.209 ± 0.027 | 0.206 ± 0.025 | +0.236 [+0.177, +0.305] |
| anchor | 0.204 ± 0.037 | 0.205 ± 0.036 | +0.240 [+0.196, +0.285] |
| **student** | 0.228 ± 0.056 | 0.191 ± 0.040 | **+0.217** [+0.166, +0.266] |

KAUH-B2 (hiệu chuẩn trên một bộ lọc, kiểm thử trên bộ lọc khác; |Δ| trung bình, null p95): 2/6 cặp vượt null gate (cả hai đích Bell).

| Cặp | baseline | previous | anchor | student | null p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Diaphragm → Bell | 0.219 | 0.095 | 0.098 | 0.109 | 0.124 |
| Extended → Bell | 0.205 | 0.103 | 0.100 | 0.115 | 0.126 |

D (giảm sai số độ phủ so với baseline, 2 cặp được kiểm): previous +0.1130 [+0.0724, +0.1532]; anchor +0.1129 [+0.0698, +0.1455]; **student +0.1000 [+0.0585, +0.1299]**.

Sàng lọc bản ghi (hiệu chuẩn ICBHI, nửa đánh giá ICBHI và KAUH, gộp theo trung bình cửa sổ; 20 lần chia hiệu chuẩn):

| Điều kiện | ICBHI: độ phủ / cỡ tập / uncertain | argmax Se / Sp | KAUH Bell: độ phủ / uncertain | Diaphragm | Extended |
| --- | --- | --- | --- | --- | --- |
| baseline | 0.898 / 1.570 / 0.570 | 0.703 / 0.621 | 0.803 / 0.524 | 0.960 / 0.625 | 0.950 / 0.560 |
| previous | 0.900 / 1.603 / 0.603 | 0.642 / 0.677 | 0.831 / 0.464 | 0.861 / 0.512 | 0.879 / 0.522 |
| anchor | 0.902 / 1.559 / 0.559 | 0.624 / 0.720 | 0.805 / 0.471 | 0.847 / 0.512 | 0.868 / 0.529 |
| **student** | 0.902 / 1.525 / 0.525 | 0.692 / 0.676 | 0.790 / 0.311 | 0.846 / 0.357 | 0.875 / 0.391 |

- [Fact] Student **không cải thiện** tính bất biến theo bộ lọc so với mô hình cũ và anchor: flip 0.228 so với 0.209 và 0.204; giảm so với baseline +0.217, trong khoảng tin cậy của hai mô hình kia. D thấp hơn một chút (+0.100 so với +0.113).
- [Fact] Student giảm tỷ lệ "uncertain" (ICBHI 0.525 so với 0.603; KAUH Bell 0.311 so với 0.464) với độ phủ KAUH thấp hơn một chút (Bell 0.790 so với 0.831). Độ phủ trên ICBHI giữ ở 0.902 (mức danh nghĩa 0.90).
- [Interpretation] Lợi ích của student nằm ở điểm Score trên split chính thức (mục 17.7), không ở độ bất biến theo bộ lọc hay độ phủ; không có tuyên bố phần cứng hay H2 nào đi kèm (KAUH là bộ lọc phần mềm).

**Các phần còn lại của đánh giá đóng băng đã xong** (campaign `student-final-20261029`): bản sinh đôi không SC, LODO 5 seed, null gate và H2 từng fold, unseen-patient, best-on-test (mục 17.10) và KAUH ghép cặp student so với twin (mục 17.11).

<!-- begin 17.10 -->
### 17.10 Student `kd_g4_k1` so với bản sinh đôi không SC, anchor, AuscultTrust cũ và baseline: split chính thức, unseen-patient, best-on-test và LODO 5 seed (29/10)

[Fact] Năm điều kiện, seed 0-4 ghép cặp, epoch cuối đã ghi: **baseline** (BEATs + PAFA không SC, epoch 100), **previous** (AuscultTrust cũ, BEATs + PAFA + SC tĩnh, epoch 100), **anchor** (`gain6-cv66`, epoch 66),
**student** (`kd_g4_k1`, epoch 66, SC tĩnh), **twin** (`kd_g4_k1_nosc`: cùng công thức và soft target G4, tắt SC; epoch 66). Tạo bằng `scripts/student_final_report.py --h2` từ `report.json` và `test_preds.npz` của từng unit;
twin và student seed 3-4 từ campaign `student-final-20261029`, student seed 0-2 từ `distill-20261028`. Đủ 5 seed cho mọi điều kiện (LODO twin LittC2SE seed 3 chạy lại sau lỗi hết bộ nhớ GPU ở lần đầu).

Split chính thức (Sp, Se, Score, HS; worst-dev = thiết bị kém nhất; unseen-patient = bỏ bệnh nhân 156 và 218; best-on-test là **lạc quan**, chọn epoch trên test):

| Condition | Sp | Se | Score | HS | Worst-dev Score | Unseen-patient Score | Best on test [optimistic] |
|---|---|---|---|---|---|---|---|
| baseline | 73.2 ± 2.7 | 50.2 ± 2.9 | 61.7 ± 0.7 | 59.5 ± 1.3 | 58.9 ± 0.5 | 61.5 ± 0.6 | 63.2 ± 0.6 |
| previous | 75.6 ± 3.1 | 48.9 ± 2.9 | 62.3 ± 0.9 | 59.3 ± 1.3 | 58.9 ± 1.4 | 62.0 ± 0.9 | 64.6 ± 0.4 |
| anchor | 75.3 ± 3.1 | 50.9 ± 1.5 | 63.1 ± 1.2 | 60.7 ± 0.8 | 60.0 ± 2.0 | 63.0 ± 1.2 | 65.0 ± 0.4 |
| student | 74.5 ± 1.3 | 53.8 ± 0.8 | 64.2 ± 0.4 | 62.5 ± 0.2 | 60.6 ± 0.6 | 64.0 ± 0.4 | 65.2 ± 0.3 |
| twin | 75.6 ± 3.4 | 52.4 ± 1.7 | 64.0 ± 0.9 | 61.8 ± 0.4 | 60.7 ± 1.4 | 63.9 ± 0.8 | 64.9 ± 0.5 |

Hiệu ghép cặp theo seed (trung bình ± SD; dương = điều kiện đầu cao hơn):

| Contrast | Δ Score | Δ Unseen-patient | Δ Sp | Δ Se | seeds above 0 |
|---|---|---|---|---|---|
| student - twin | 0.2 ± 0.8 | 0.1 ± 0.6 | -1.1 ± 3.3 | 1.4 ± 1.8 | 3/5 |
| student - anchor | 1.0 ± 0.9 | 1.0 ± 0.9 | -0.8 ± 2.2 | 2.8 ± 1.0 | 4/5 |
| student - previous | 1.9 ± 0.9 | 2.0 ± 0.9 | -1.1 ± 4.2 | 4.9 ± 3.6 | 5/5 |
| student - baseline | 2.4 ± 0.7 | 2.5 ± 0.8 | 1.3 ± 3.9 | 3.6 ± 3.6 | 5/5 |
| twin - baseline | 2.2 ± 1.2 | 2.4 ± 1.1 | 2.3 ± 4.3 | 2.1 ± 3.5 | 5/5 |
| previous - baseline | 0.5 ± 1.5 | 0.5 ± 1.4 | 2.4 ± 1.9 | -1.3 ± 1.3 | 4/5 |

LODO (Score trên thiết bị bị giữ kín; số trong ngoặc vuông = số seed có sẵn; macro chỉ tính các seed đủ cả bốn thiết bị):

| Condition | Meditron | LittC2SE | Litt3200 | AKGC417L | Macro (seeds with all 4 devices) |
|---|---|---|---|---|---|
| baseline | 57.2 ± 2.4 [5] | 62.0 ± 1.4 [5] | 59.7 ± 3.3 [5] | 56.4 ± 1.6 [5] | 58.8 ± 1.4 [5] |
| previous | 57.6 ± 2.8 [5] | 67.5 ± 1.1 [5] | 61.5 ± 6.0 [5] | 54.6 ± 0.5 [5] | 60.3 ± 0.8 [5] |
| anchor | 60.3 ± 1.6 [5] | 67.2 ± 2.3 [5] | 63.6 ± 7.5 [5] | 55.4 ± 1.5 [5] | 61.6 ± 1.4 [5] |
| student | 60.1 ± 1.0 [5] | 69.9 ± 1.4 [5] | 63.7 ± 8.6 [5] | 55.2 ± 1.1 [5] | 62.2 ± 2.0 [5] |
| twin | 57.9 ± 1.8 [5] | 66.2 ± 1.8 [5] | 60.8 ± 8.1 [5] | 56.8 ± 1.4 [5] | 60.4 ± 2.7 [5] |

Hiệu macro ghép cặp theo seed chung:

| Contrast | Δ macro Score | per-seed Δ |
|---|---|---|
| student - twin | 1.8 ± 1.2 [5 seeds] | +2.99 / +0.72 / +0.51 / +1.92 / +2.88 |
| student - anchor | 0.6 ± 1.0 [5 seeds] | -0.41 / +2.12 / -0.21 / +0.51 / +1.04 |
| student - previous | 1.9 ± 1.6 [5 seeds] | -0.72 / +3.58 / +2.05 / +2.58 / +2.14 |
| student - baseline | 3.4 ± 0.7 [5 seeds] | +2.43 / +3.57 / +3.83 / +3.23 / +4.12 |
| twin - baseline | 1.6 ± 1.5 [5 seeds] | -0.56 / +2.85 / +3.32 / +1.31 / +1.23 |
| previous - baseline | 1.5 ± 1.2 [5 seeds] | +3.15 / -0.00 / +1.78 / +0.64 / +1.97 |

LODO theo thiết bị: Sp, Se, Score, HS, độ phủ conformal 90% (LAC), sai lệch độ phủ đã khớp tỷ lệ lớp (Delta, dương = thiếu phủ), cỡ tập, ngưỡng bền (q_robust):

| Held-out device | Arm | Seeds | Test cycles | Sp | Se | Score | HS | Coverage | Delta (prior-matched) | Set size | Singleton | Empty | q_robust coverage | q_robust Delta | q_robust set size |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Meditron | baseline | 5 | 1370 | 85.6 | 28.7 | 57.2 | 43.0 | 0.918 | +0.033 | 1.94 | 0.35 | 0.000 | 0.939 | -0.007 | 2.24 |
| Meditron | previous | 5 | 1370 | 78.7 | 36.5 | 57.6 | 49.7 | 0.902 | +0.062 | 1.77 | 0.42 | 0.000 | 0.940 | +0.003 | 2.18 |
| Meditron | anchor | 5 | 1370 | 80.7 | 39.9 | 60.3 | 53.3 | 0.916 | +0.030 | 1.78 | 0.39 | 0.000 | 0.949 | -0.020 | 2.15 |
| Meditron | student | 5 | 1370 | 82.1 | 38.0 | 60.1 | 51.7 | 0.913 | +0.040 | 1.69 | 0.39 | 0.000 | 0.938 | +0.005 | 2.01 |
| Meditron | twin | 5 | 1370 | 83.0 | 32.9 | 57.9 | 47.0 | 0.902 | +0.062 | 1.72 | 0.38 | 0.000 | 0.944 | -0.008 | 2.10 |
| LittC2SE | baseline | 5 | 594 | 90.8 | 33.2 | 62.0 | 48.5 | 0.857 | +0.118 | 1.95 | 0.49 | 0.000 | 0.912 | +0.032 | 2.41 |
| LittC2SE | previous | 5 | 594 | 95.3 | 39.7 | 67.5 | 55.9 | 0.880 | +0.109 | 1.86 | 0.54 | 0.000 | 0.906 | +0.072 | 2.15 |
| LittC2SE | anchor | 5 | 594 | 95.7 | 38.7 | 67.2 | 54.9 | 0.869 | +0.095 | 1.68 | 0.60 | 0.000 | 0.908 | +0.039 | 2.00 |
| LittC2SE | student | 5 | 594 | 96.8 | 43.0 | 69.9 | 59.5 | 0.908 | +0.050 | 1.60 | 0.54 | 0.000 | 0.957 | -0.034 | 2.04 |
| LittC2SE | twin | 5 | 594 | 94.1 | 38.3 | 66.2 | 54.2 | 0.898 | +0.060 | 1.80 | 0.38 | 0.000 | 0.934 | +0.004 | 2.11 |
| Litt3200 | baseline | 5 | 481 | 88.5 | 30.9 | 59.7 | 44.5 | 0.966 | -0.002 | 2.76 | 0.10 | 0.000 | 0.983 | -0.048 | 3.22 |
| Litt3200 | previous | 5 | 481 | 88.2 | 34.8 | 61.5 | 48.3 | 0.975 | -0.045 | 3.01 | 0.06 | 0.000 | 0.983 | -0.072 | 3.22 |
| Litt3200 | anchor | 5 | 481 | 80.8 | 46.4 | 63.6 | 58.0 | 0.988 | -0.065 | 3.32 | 0.00 | 0.000 | 0.995 | -0.088 | 3.63 |
| Litt3200 | student | 5 | 481 | 75.9 | 51.5 | 63.7 | 58.6 | 0.947 | +0.025 | 2.33 | 0.06 | 0.000 | 0.963 | -0.016 | 2.70 |
| Litt3200 | twin | 5 | 481 | 74.1 | 47.4 | 60.8 | 54.2 | 0.928 | +0.076 | 2.10 | 0.10 | 0.000 | 0.956 | +0.006 | 2.60 |
| AKGC417L | baseline | 5 | 4216 | 84.7 | 28.0 | 56.4 | 41.8 | 0.809 | +0.002 | 2.26 | 0.35 | 0.000 | 0.888 | -0.039 | 2.68 |
| AKGC417L | previous | 5 | 4216 | 93.2 | 15.9 | 54.6 | 27.2 | 0.700 | +0.061 | 1.89 | 0.40 | 0.000 | 0.830 | -0.002 | 2.39 |
| AKGC417L | anchor | 5 | 4216 | 90.6 | 20.2 | 55.4 | 32.8 | 0.660 | +0.082 | 1.67 | 0.56 | 0.000 | 0.755 | +0.029 | 2.20 |
| AKGC417L | student | 5 | 4216 | 91.6 | 18.9 | 55.2 | 31.0 | 0.648 | +0.098 | 1.36 | 0.70 | 0.000 | 0.706 | +0.066 | 1.54 |
| AKGC417L | twin | 5 | 4216 | 86.3 | 27.2 | 56.8 | 41.1 | 0.734 | +0.059 | 1.57 | 0.55 | 0.000 | 0.787 | +0.026 | 1.77 |

Null gate (|Delta| của baseline so với phân vị 95 của |Delta| dưới các lần chia lại hoán đổi được của bệnh nhân hiệu chuẩn):

| Device | baseline abs Delta | null p95 | tested |
|---|---|---|---|
| Meditron | 0.035 | 0.113 | no |
| LittC2SE | 0.118 | 0.150 | no |
| Litt3200 | 0.044 | 0.115 | no |
| AKGC417L | 0.032 | 0.115 | no |

**Đọc kết quả:**

- [Fact] **SC không đóng góp được ở split chính thức cho student**: student 64.2 ± 0.4 so với twin 64.0 ± 0.9, hiệu +0.2 ± 0.8 (3/5 seed dương); unseen-patient +0.1 ± 0.6. Phần lớn lợi ích của student so với baseline (+2.4 ± 0.7, 5/5 seed) có ở cả twin không SC (+2.2 ± 1.2, 5/5 seed): nguồn của lợi ích là công thức huấn luyện (gain6 cộng distillation từ G4), không phải SC.
- [Fact] **SC có lợi ở LODO macro**: student 62.2 ± 2.0 so với twin 60.4 ± 2.7, hiệu +1.8 ± 1.2 (dương ở cả 5 seed). Lợi ích đến từ LittC2SE, Litt3200 và Meditron; AKGC417L giảm (55.2 so với 56.8), giống mô hình cũ (54.6 so với baseline 56.4).
- [Fact] So với anchor và mô hình cũ: student - anchor +1.0 ± 0.9 official (4/5 seed), LODO macro +0.6 ± 1.0 (3/5 seed dương); student - previous +1.9 ± 0.9 official (5/5), LODO macro +1.9 ± 1.6.
- [Fact] **H2 vẫn không kiểm định được**: baseline không có thiết bị nào vượt null gate (gần nhất LittC2SE 0.118 so với p95 0.150), nên không tính được thống kê D cho student, twin, anchor hay previous. Với twin làm nền cũng không thiết bị nào được kiểm.
- [Fact] **Student cho tập dự đoán gọn hơn nhưng thiếu phủ nặng hơn trên AKGC417L**: cỡ tập 1.36 (previous 1.89, twin 1.57), độ phủ thô 0.648 (previous 0.700, twin 0.734, baseline 0.809), Delta đã khớp tỷ lệ lớp +0.098 (lớn nhất trong các điều kiện). Trên Litt3200 student phủ 0.947 với cỡ tập 2.33, thấp hơn previous (0.975, 3.01). Không có kiểm định H2 đi kèm; đây là quan sát mô tả.
- [Fact] Best-on-test (lạc quan): student 65.2 ± 0.3, twin 64.9 ± 0.5, anchor 65.0 ± 0.4; PAFA công bố 64.84 ± 0.60 và PC-MCL 65.37 ± 0.73 theo cùng quy tắc chọn epoch trên test (mục 7.7).
- [Interpretation] Tên "hiệu chỉnh phổ" trong AuscultTrust vẫn có lý do ở LODO (+1.8 macro) nhưng không ở split chính thức; kết quả này không thay đổi quyết định chọn student, vì student cao hơn cả mô hình cũ và baseline ở cả hai điểm cuối.
<!-- end 17.10 -->

<!-- begin 17.11 -->
### 17.11 KAUH ghép cặp: student so với bản sinh đôi không SC (run `final-kd-g4-k1-twin`, 29/10)

[Fact] Cùng giao thức mục 17.9 (5 seed, hash trọng số và `s_ref` đóng băng, bootstrap theo bệnh nhân và seed 1.000 lần, 200 lần chia null), nền = **twin** (student tắt SC).
`twin_ttsc` = trọng số twin cộng SC tĩnh **chỉ lúc kiểm thử** với tham chiếu của student (tương phản tách biệt). Đây là kiểm soát dương H1 cho mô hình mới, trên bộ lọc phần mềm.

| Điều kiện | Flip rate | TV distance | Giảm flip so với twin [KTC95%] |
| --- | ---: | ---: | --- |
| twin (không SC) | 0.289 ± 0.058 | 0.238 ± 0.035 | |
| **student** (SC lúc huấn luyện và kiểm thử) | 0.228 ± 0.056 | 0.191 ± 0.040 | +0.061 [+0.029, +0.093] |
| twin_ttsc (SC chỉ lúc kiểm thử) | 0.206 ± 0.049 | 0.170 ± 0.030 | +0.083 [+0.043, +0.118] |

KAUH-B2 (2/6 cặp qua null gate khi nền là twin; cả hai đích Bell): |Delta| trung bình tại cặp được kiểm (null p95 khoảng 0.119):

| Cặp | twin | student | twin_ttsc |
| --- | ---: | ---: | ---: |
| Diaphragm → Bell | 0.130 | 0.109 | 0.099 |
| Extended → Bell | 0.131 | 0.115 | 0.094 |

D (giảm sai số độ phủ so với twin, 2 cặp được kiểm): **student +0.0184 [-0.0125, +0.0410]** (khoảng chứa 0); twin_ttsc +0.0343 [+0.0102, +0.0643].

Sàng lọc bản ghi (hiệu chuẩn ICBHI, trung bình cửa sổ): uncertain ICBHI twin 0.481, student 0.525; KAUH Bell twin 0.349, student 0.311 (độ phủ 0.778 so với 0.790).

- [Fact] SC lúc huấn luyện làm student bất biến hơn twin theo bộ lọc phần mềm: giảm flip +0.061 [+0.029, +0.093]. Tuy vậy SC lúc kiểm thử chỉ trên trọng số của twin (+0.083) còn tốt hơn, nên lợi ích bất biến đến từ bản thân việc hiệu chỉnh phổ lúc suy luận hơn là từ việc huấn luyện cùng SC.
- [Fact] Về độ phủ conformal (B2), student không khác twin có ý nghĩa (D +0.018, khoảng chứa 0); chỉ SC chỉ lúc kiểm thử có D dương có ý nghĩa (+0.034). Không có tuyên bố phần cứng: KAUH là bộ lọc phần mềm trên cùng bản ghi.
- [Interpretation] Kết hợp với mục 17.10: SC của student cho lợi ích đo được ở bất biến bộ lọc (+0.061) và LODO macro (+1.8), không phải ở điểm Score trên split chính thức (+0.2). Bằng chứng vật lý (phổi giả đăng ký) vẫn [Pending].
<!-- end 17.11 -->

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

1. [Done 28/10, section 17.7] **Target evaluation of the stage-12 recipes** on A, B, D, E: official, four-device LODO and KAUH (with recording
   screening) for 5 seeds, from the frozen `fivefold-20261026` snapshot, with a PCSL-5 non-class-aware control and the
   anchor as paired references. Tiers, GPU-hour estimate, preflight checks and stop rules:
   `docs/updates/2026-10-27-target-evaluation-plan.md`. [Finished 28/10 as campaign `target-20261027`.]
2. Human listening of the 50 audio-audit candidates; keep the registered 8-second preprocessing.
3. Optional before the cut-off (31/01/2027): `k rate` with few target cycles and SC jointly at finite N; distillation of an
   ensemble into one BEATs model. The registered E1-E4 and the AST / HTS-AT dynamic-SC cells are already computed (sections 12.2, 13.2).
4. Phantom: hardware, capture protocol, remove-and-replace ceiling, then H1, H3 and co-primary H2 on the frozen
   weights and s_ref (`outputs/train/final-pafa-static/*/seed*/freeze.json`). No H2 criterion is relaxed after seeing results.
5. Commit the working tree in separate commits (refactor, campaign tooling, app, documents): it currently holds 51 modified,
   42 deleted (moved) and 75 new files that are not in git.

## Phụ lục A. Đính chính và nguồn gốc của báo cáo này

1. **Đính chính trạng thái stage 12.** Tài liệu `docs/reports/2026-10-27-findings.md` (27/10) ghi năm recipe stage 12 "chưa có kết quả official, LODO hay KAUH". Điều đó đã lỗi thời: campaign `target-20261027` (175 fit, xong 28/10) đã đánh giá cả năm recipe và control PCSL-5 (mục 17.7). Kết quả: lợi ích CV nguồn của class-aware light và LoRA **không** chuyển sang split chính thức hay LODO. Báo cáo này lấy số của `report.md` v7.0 và `docs/reports/2026-10-28-target-distillation-results.md`.
2. **Cách ghép.** Mục 2-4, 7-19 là nguyên văn `report.md` v7.0 (giữ nguyên số mục để các tham chiếu chéo còn đúng); mục 5 và 6 chứa tóm tắt viết mới cộng nguyên văn hai báo cáo giai đoạn đầu; mục 1, mục 17.8 và phụ lục này viết mới. Không có số liệu nào được tính lại.
3. **Mục 19 (Next steps) và mục 1 của `report.md` gốc** lỗi thời một phần (viết trước 28/10 và 29/10): bước "target evaluation" đã làm (mục 17.7); campaign `combo-20261028` đã xong (mục 17.8). Phần còn mở: nghe thủ công 50 mẫu audit, phổi giả, distillation bằng teacher out-of-fold, commit working tree.
4. **Hạn chế bổ sung của bản tổng hợp.** Giao thức khác nhau giữa các nhóm ở bảng 1.3 (AST 50 epoch, BEATs 100 epoch, epoch chọn bằng CV nguồn); số của các nhóm ensemble và distillation là thăm dò, được chọn trên cùng dữ liệu test; student distill có 3 seed ở LODO; mô hình `class_aware_multiscale` chỉ có CV nguồn.
5. **Nơi lấy lại số liệu.** `outputs/train/<run>/<cell>/seed<s>/` (A, HDD), `build/campaigns/<programme>/` (B chuẩn), `outputs/kauh/`, `outputs/screening/`, `outputs/evaluation/`; lệnh tái tạo bảng trong `docs/QUICKSTART.md` mục 7 và `docs/handoff/HANDOFF.md`.

## Phụ lục B. Kiểm tra nhất quán giữa các báo cáo theo ngày (29/10)

Đối chiếu `docs/reports/`, `docs/decision-logs/`, `docs/updates/`, `docs/handoff/HANDOFF.md`, `report.md` và bản này. Các mâu thuẫn tìm được và cách xử lý:

| # | Chủ đề | Mâu thuẫn | Xử lý |
| --- | --- | --- | --- |
| 1 | Nhãn "frozen (no SC)" = 0.209 | Báo cáo 28/10 (mục 3, 4) gọi dòng `frozen` là "no SC". Kiểm tra: cờ `sc` của `outputs/kauh/target-20261027/frozen` là true và hash checkpoint bằng `final-pafa-static/auscult` (seed 0: 02bdb2df, seed 1: 9c27c14a). Mô hình không SC của quyết định 23/10 có flip **0.445**, mô hình có SC **0.209** (đánh giá 23/10 và 29/10 khớp nhau) | Sửa nhãn trong báo cáo 28/10, `report.md` (17.7), `HANDOFF.md`, bản này (17.7): "frozen" = AuscultTrust cũ có SC. Mọi so sánh "vs frozen" trong báo cáo 28/10 là so với mô hình SC, không phải so với baseline không SC |
| 2 | B2 "0/6 cặp" so với "2/6 cặp" | Báo cáo 28/10: không cặp nào qua null gate; báo cáo 23/10 và 29/10: 2/6 cặp (đích Bell) với baseline không SC | Cùng nguyên nhân với mục 1: null gate dựa vào |Δ| của mô hình nền; nền là mô hình SC (đã nhỏ) nên 0/6. Thêm chú thích đính chính; kết quả đúng với baseline không SC là 2/6 |
| 3 | "Baseline" có nhiều nghĩa theo thời kỳ | (i) v9 đăng ký: AST Patch-Mix CL không xử lý thiết bị, epoch 2 (mục 6, R2.1); (ii) quyết định 23/10: BEATs + PAFA không SC, epoch 100; (iii) stage 12: anchor `gain6-cv66` (có SC tĩnh + gain ngẫu nhiên) là chuẩn so sánh; (iv) student 29/10: bản sinh đôi không SC | Giữ nguyên từng nghĩa trong ngữ cảnh của nó. Baseline của dự án (quyết định 23/10) = BEATs + PAFA không SC, epoch 100, không còn là "giả định". Tương phản SC cho student dùng bản sinh đôi không SC (đang huấn luyện); chưa có thì không tuyên bố "SC giúp student" |
| 4 | "[Pending] official/LODO/KAUH cho 5 recipe" | `2026-10-27-findings`, `2026-10-27-improvement-results`, `HANDOFF` mục 1, `report.md` mục 1 (điểm 9) và 17.5 (điểm 4) vẫn ghi chưa chạy; đã chạy xong 28/10 (mục 17.7) | Thêm ghi chú cập nhật ở hai báo cáo 27/10, sửa `HANDOFF`, `report.md`, bản này. Kết luận: lợi ích CV nguồn của class-aware light và LoRA không chuyển sang đánh giá đích |
| 5 | Trạng thái kế hoạch | `docs/README.md` ghi kế hoạch đánh giá đích "(proposed)"; kế hoạch distillation ghi "proposed, not launched" | Đổi thành "executed" |
| 6 | Điểm anchor trên split chính thức | 63.143 ± 1.203 (báo cáo 27/10, `report.md` 17.3) so với 63.13 (báo cáo 28/10) so với 63.135 ± 1.190 (tính lại ngày 29/10 từ `report.json` của 5 unit `gain6-cv66`) | Chênh 0.008; chưa tìm ra nguồn của 63.143. Dùng 63.13 (làm tròn 63.135) cho hiệu student - anchor = +1.02 (khớp tính lại: +1.019). [Pending] đối chiếu nguồn của 63.143 |
| 7 | Phạm vi bootstrap | Báo cáo 28/10 (mục 7): "bootstrap chỉ theo seed"; tóm tắt KAUH: "theo bệnh nhân và seed" | Hai phạm vi khác nhau: hiệu Score (official, LODO) chỉ bootstrap theo seed; flip, B2 và D bootstrap theo bệnh nhân và seed. Đã ghi rõ ở báo cáo 28/10 và log 29/10 |
| 8 | Tái hiện Patch-Mix CL | `report.md` mục 6 ghi 61.49 ± 0.91 (best on test); báo cáo registered-core ghi 61.4 ± 1.3 | Hai bộ chạy khác nhau: Phase 0 (script `patchmix_cl`, 10/05) và `repro-wave1` của nghiên cứu đăng ký. Cả hai nằm trong ngưỡng chấp nhận so với 62.37; mục 6 nêu cả hai |
| 9 | Mô hình trong báo cáo replay phổi giả 29/10 | `2026-10-29-replay-phone-electret.md` dùng "distilled AuscultTrust, 5-seed softmax ensemble" (`kd_g4_k1_ens5`); định nghĩa 29/10 là một mô hình (phục vụ seed 0) | Không sửa tài liệu của phiên khác. [Pending] quyết định chính thức: ensemble 5 seed của student chỉ là biến thể thăm dò, không phải định nghĩa AuscultTrust |
| 10 | Trạng thái phổi giả | Các báo cáo khác ghi H1, H3, H2 đồng chính "chờ phần cứng"; báo cáo replay 29/10 đã có 20 take mỗi thiết bị thu (điện cực, điện thoại) | Đó là dữ liệu phát lại tier 1, chỉ để đánh giá (chuỗi phát lại làm mất nội dung: coherence 0.02-0.15) và có sai lệch quy trình ghi trong chính báo cáo đó. Không thay thế phép thử phổi giả đăng ký; tiêu chí H1, H3, H2 không đổi và vẫn [Pending] |
| 11 | Số seed LODO của student | Bảng 28/10: LODO 3 seed (0-2); kế hoạch cuối: 5 seed | Đã thay bằng 5 seed trong mục 1.3 và 17.10 (twin LittC2SE seed 3 chạy lại sau lỗi hết bộ nhớ GPU của lần đầu) |
