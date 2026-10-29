# Tổng hợp các cải tiến AuscultTrust — 27/10/2026

| Type | Date | Status |
| --- | --- | --- |
| report | 2026-10-27 | current as of 27/10; target results superseded by 2026-10-28 |

> Update 28/10: the five recipes were evaluated on the official split, LODO and KAUH (`target-20261027`): class-aware light 58.66 official, LoRA 57.05 (anchor 63.13); the source-CV gains did not transfer (`2026-10-28-target-distillation-results.md`). Statements below that call these endpoints unrun describe the 27/10 state.

## Phạm vi và trạng thái

| Hạng mục | Trạng thái | Quy mô | Artifact |
| --- | --- | --- | --- |
| Source CV control, attention, gain6 | Hoàn tất | 5 arm × 3 fold × 2 seed | `logs/improvement-20261025/cv-selection*.json` |
| Chín cải tiến tiếp theo | Hoàn tất | 30 fit screen + 6 fit confirm + 25 fit final | `outputs/evaluation/improvements-20261026/` |
| SWA, LoRA, multi-scale, class-aware light, LoRA + class-aware | Hoàn tất source CV | 6 arm × 5 fold × 5 seed = 150 fit | `build/campaigns/fivefold-20261026/outputs/evaluation/fivefold-20261026/` |
| KD cũ | Hoàn tất pilot | 3 seed official + 3 seed Litt3200 | `outputs/train/distill-pilot-20261024/` |
| KD hard + soft cải tiến | Screen 3 fold seed 0 xong trên B; không qua gate (epoch 13, Δ +0,009; fold +0,99/−1,76/+0,79). Chưa có official/LODO | 25 test CPU đạt, 3 fit GPU | `next-improvements-20261026-s0/kd/cv0-2` trên B; gate tính lại 27/10 bằng `improvement_campaign.gate` |
| Event heads (BCE crackle/wheeze phụ) | Screen 3 fold seed 0 xong trên B; không qua gate (epoch 22, Δ −1,637; fold −1,47/−0,41/−3,02). Không confirm/final | 3 fit | `followup-20261026-s0/event_heads/cv0-2` trên B |
| Equal-weight ensemble | Hoàn tất | 5 seed × official/4 LODO | tables in this file (section on equal-weight ensembles) |
| Learned fusion | Chưa fit | OOF anchor/LLRD: 4.142 cycle, 79 patient, 3 fold | `outputs/evaluation/next-addons-20261026/fusion-feasibility.json` |
| Target adaptation | Hoàn tất đánh giá tách riêng | 5 seed; 33 adaptation + 78 evaluation patient | `outputs/evaluation/next-addons-20261026/summary.json` |
| Source label/boundary audio audit | Hoàn tất tự động; nghe người chưa hoàn tất | 4.025 cycle, 77 patient | `outputs/evaluation/source-audio-audit-20261026/` |
| Trained SC static/dynamic và PAFA + PC-MCL | Hoàn tất | 125/125 fit, 5 seed × official/4 LODO | `report.md` mục 13 |
| External pretraining | Bỏ qua theo yêu cầu | 0 fit | user decision 26/10 (`report.md` section 3) |

## Cấu hình và cách đánh giá

| Thành phần | Thiết lập |
| --- | --- |
| Backbone / objective | BEATs fine-tune; PAFA; 4 lớp Normal, Crackle, Wheeze, Both |
| Input / SC | Cửa sổ huấn luyện 5 s; 128 mel; SC static; gain ngẫu nhiên 6 dB SD trong các arm gain6 |
| Optimizer | Adam, learning rate 5×10⁻⁵, weight decay 10⁻⁶, cosine schedule 100 epoch; batch size 32; PCSL 50, GPAL 0,0005 trừ khi override |
| Source CV 25/10 | 3 fold patient-disjoint × seed 0–1; epoch lấy từ đường Score validation trung bình |
| Chương trình 26/10 | Screen 3 fold seed 0; confirm 3 fold seed 1; final 5 seed official + 4 LODO; cùng split/seed trong paired contrast |
| Chương trình 27/10 | 5 fold patient-disjoint × 5 seed (0–4), 100 epoch, epoch chọn trên mean source-validation curve riêng từng arm; anchor epoch 66 |
| Thước đo | ICBHI Score = (Sp + Se)/2, đơn vị điểm phần trăm; Δ là hiệu Score paired với anchor cùng seed/fold tại epoch source-CV đã chọn |
| Phạm vi 150-fit | Source patient CV; chưa có official-test, LODO hoặc KAUH cho SWA/LoRA/multi-scale/class-aware light/combo |

| Năm arm mới | Official test | LODO bốn thiết bị | KAUH |
| --- | --- | --- | --- |
| SWA | Chưa chạy | Chưa chạy | Chưa chạy |
| LoRA | Chưa chạy | Chưa chạy | Chưa chạy |
| Multi-scale | Chưa chạy | Chưa chạy | Chưa chạy |
| Class-aware light | Chưa chạy | Chưa chạy | Chưa chạy |
| LoRA + class-aware | Chưa chạy | Chưa chạy | Chưa chạy |

| Phép đo tiếp theo | Quy trình |
| --- | --- |
| Official | Retrain với epoch và recipe đã chọn bằng source CV; đánh giá official test bằng seed ghép cặp. |
| LODO | Retrain riêng khi giữ lại từng thiết bị; đánh giá trên thiết bị đó, bốn cohort. |
| KAUH | Đóng băng checkpoint rồi chạy inference mức bản ghi và bộ lọc KAUH, không chọn theo nhãn KAUH. |

## Trained SC static/dynamic và PAFA + PC-MCL (5 seed)

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

## Kết quả source CV: attention và gain6 (3 fold × 2 seed)

| Arm | Epoch | Score CV | Δ với control | Score epoch cuối |
| --- | ---: | ---: | ---: | ---: |
| control | 21 | 60.125 | +0.000 | 59.259 |
| attention | 50 | 60.518 | +0.393 | 59.164 |
| attention_gain | 61 | 60.623 | +0.499 | 60.346 |
| mean_gain | 67 | 61.084 | +0.959 | — |
| attention_gain_lrhalf | 33 | 60.207 | +0.082 | — |

## Kết quả source CV: chín cải tiến (3 fold)

| Arm | Epoch chọn | Δ Score screen | Fold Δ (0/1/2) | Cycle Δ | Kết quả gate |
| --- | ---: | ---: | --- | ---: | --- |
| llrd | 17 | +0.752 | +0.96/+2.30/-1.00 | +0.752 | Đạt |
| patient_cap | 13 | -0.686 | +0.20/-0.93/-1.33 | -0.686 | Không đạt |
| class_topk | 21 | -0.614 | -1.39/+0.29/-0.74 | -0.614 | Không đạt |
| class_attention | 68 | -1.072 | -0.52/-0.75/-1.95 | -1.072 | Không đạt |
| class_aware | 29 | +0.042 | +0.50/+0.28/-0.66 | +0.042 | Không đạt |
| recording_head | 3 | -19.460 | -12.26/-17.31/-28.81 | -12.117 | Không đạt |
| band_noise | 11 | -0.644 | +4.65/-5.92/-0.66 | -0.644 | Không đạt |
| band_noise_consistency | 20 | -1.856 | +0.74/-4.02/-2.29 | -1.856 | Không đạt |
| recording_head_pcsl0 | 4 | -1.251 | -2.19/+1.37/-2.94 | -2.551 | Không đạt |

| LLRD confirm (2 seed × 3 fold) | 21 | +0.553 | +0.39/+0.52/+0.75 | +0.553 | Đạt |

| Metric ở bảng screen | Định nghĩa |
| --- | --- |
| `recording_head`, `recording_head_pcsl0`: Δ screen | Binary recording Score |
| Các arm khác: Δ screen | Cycle Score |
| Cycle Δ | Cycle Score |

## Kết quả source CV: 5 fold × 5 seed (150/150 fit)

| Arm | Epoch | Mean Score | Δ với anchor | Δ theo seed 0–4 | Δ theo fold 0–4 |
| --- | ---: | ---: | ---: | --- | --- |
| anchor | 66 | 60.587 | +0.000 | +0.00 / +0.00 / +0.00 / +0.00 / +0.00 | +0.00 / +0.00 / +0.00 / +0.00 / +0.00 |
| swa | 43 | 60.663 | +0.076 | -0.37 / +1.28 / +0.53 / -0.21 / -0.86 | -0.37 / +0.60 / +0.57 / +0.21 / -0.64 |
| lora | 83 | 61.194 | +0.607 | +0.23 / +0.92 / +0.84 / -0.27 / +1.32 | +1.31 / +4.17 / -1.54 / +0.72 / -1.62 |
| multiscale | 68 | 60.266 | -0.321 | -0.76 / +0.24 / +0.52 / -0.89 / -0.72 | -0.20 / -1.53 / +0.37 / -0.54 / +0.30 |
| class_aware_light | 4 | 62.045 | +1.458 | +0.88 / +2.53 / +2.33 / +1.93 / -0.38 | +0.92 / +4.19 / +1.63 / +0.31 / +0.24 |
| lora_class_aware | 43 | 61.282 | +0.695 | +0.14 / +1.01 / +1.29 / +0.06 / +0.97 | +1.27 / +5.06 / -2.93 / +1.03 / -0.95 |

| Arm | Score seed 0 | Score seed 1 | Score seed 2 | Score seed 3 | Score seed 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| anchor | 61.107 | 59.584 | 59.840 | 61.024 | 61.380 |
| swa | 60.739 | 60.868 | 60.375 | 60.813 | 60.522 |
| lora | 61.340 | 60.503 | 60.682 | 60.751 | 62.696 |
| multiscale | 60.344 | 59.821 | 60.362 | 60.138 | 60.664 |
| class_aware_light | 61.988 | 62.119 | 62.167 | 62.956 | 60.996 |
| lora_class_aware | 61.247 | 60.595 | 61.133 | 61.082 | 62.353 |

| Source CV contrast at epoch 4 | Anchor Score | Class-aware light Score | Paired Δ | Seed-mean Δ (0–4) |
| --- | ---: | ---: | ---: | --- |
| 25 matched fold/seed units | 55.802 | 62.045 | +6.243 | +9.90 / +8.27 / +2.49 / +5.75 / +4.80 |

## Official và LODO đã chạy (Score mean ± seed SD, 5 seed)

| Candidate | Cohort | Arm | Score mean ± seed SD |
| --- | --- | --- | --- |
| anchor | official | control | 61.044 ± 2.001 |
| anchor | official | gain | 62.416 ± 0.633 |
| anchor | official | llrd | 62.313 ± 1.344 |
| anchor | official | anchor | 63.143 ± 1.203 |
| anchor | AKGC417L | control | 55.012 ± 1.112 |
| anchor | AKGC417L | gain | 55.293 ± 1.735 |
| anchor | AKGC417L | llrd | 54.648 ± 1.004 |
| anchor | AKGC417L | anchor | 55.406 ± 1.462 |
| anchor | Litt3200 | control | 60.775 ± 6.898 |
| anchor | Litt3200 | gain | 61.697 ± 7.648 |
| anchor | Litt3200 | llrd | 59.683 ± 6.275 |
| anchor | Litt3200 | anchor | 63.573 ± 7.500 |
| anchor | LittC2SE | control | 67.086 ± 2.600 |
| anchor | LittC2SE | gain | 67.385 ± 2.002 |
| anchor | LittC2SE | llrd | 67.321 ± 1.176 |
| anchor | LittC2SE | anchor | 67.191 ± 2.260 |
| anchor | Meditron | control | 58.843 ± 4.162 |
| anchor | Meditron | gain | 59.999 ± 2.998 |
| anchor | Meditron | llrd | 60.789 ± 2.635 |
| anchor | Meditron | anchor | 60.303 ± 1.650 |

## KAUH và ICBHI recording screening: gain6 so với LLRD

| Arm | Cohort | Coverage % | Argmax Se % | Argmax Sp % | Screening Se % | Screening Sp % |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| gain | ICBHI evaluation half | 90.44 | 65.34 | 66.48 | 91.48 | 30.97 |
| gain | KAUH Bell (gain) | 81.90 | 51.05 | 86.86 | 76.17 | 49.74 |
| gain | KAUH Diaphragm (gain) | 85.63 | 58.16 | 89.14 | 81.47 | 52.54 |
| gain | KAUH Extended (gain) | 87.02 | 62.63 | 84.00 | 84.00 | 47.51 |
| llrd | ICBHI evaluation half | 91.57 | 55.96 | 78.59 | 89.69 | 33.01 |
| llrd | KAUH Bell (llrd) | 91.90 | 59.47 | 82.29 | 90.55 | 40.46 |
| llrd | KAUH Diaphragm (llrd) | 90.61 | 57.37 | 90.29 | 87.67 | 44.80 |
| llrd | KAUH Extended (llrd) | 93.59 | 67.37 | 84.57 | 92.22 | 35.20 |

| KAUH metric | gain6 | LLRD |
| --- | ---: | ---: |
| A: mean argmax flip rate | 0.223 | 0.239 |
| A: mean total-variation distance | 0.218 | 0.182 |
| B2: tested filter pairs | 0/6 | 0/6 |

## Preprocessing audit (ICBHI source recording, 3 fold)

| Variant | Binary Score trung bình |
| --- | ---: |
| registered8 | 71.769 |
| duration5 | 68.751 |
| scipy8 | 71.276 |
| no_recording_fade8 | 71.292 |

## KD pilot và KD cải tiến

| Arm | Phạm vi | Seed | Score mean ± SD | Score từng seed |
| --- | --- | ---: | ---: | --- |
| KD student cũ | Official | 3 | 62.751 ± 0.757 | 63.607 / 62.167 / 62.480 |
| KD student cũ | LODO Litt3200 | 3 | 60.565 ± 6.718 | 54.144 / 60.007 / 67.545 |
| KD hard + soft cải tiến | Screen 3 fold (không phải official/LODO) | 3 fold | Δ source +0,009 (không qua gate) | — |

| KD cải tiến: tham số | Giá trị |
| --- | --- |
| Hard-label CE weight | 0,5 |
| Teacher KL temperature | 2; KL nhân T² |
| Teacher | Snapshot epoch 66; cùng outer-fold optimizer cohort; target in-sample; student khởi tạo pretrained encoder |
| Kiểm thử | 25 CPU tests đạt |

## Equal-weight ensemble: fixed member additions

| Group | Split | Seeds complete | Sp | Se | Score | HS |
|---|---|---|---|---|---|---|
| E1 | official | 5 | 76.517 | 50.331 | 63.424 | 60.628 |
| E1+gain6_cv66 | official | 5 | 77.264 | 51.351 | 64.307 | 61.586 |
| E1+LLRD_cv21 | official | 5 | 76.808 | 50.518 | 63.663 | 60.851 |
| E1_E3 | official | 5 | 76.238 | 50.518 | 63.378 | 60.663 |
| E1_E3+gain6_cv66 | official | 5 | 76.935 | 51.521 | 64.228 | 61.609 |
| E1_E3+LLRD_cv21 | official | 5 | 76.719 | 50.892 | 63.806 | 61.070 |
| E1 | Meditron | 5 | 79.863 | 41.040 | 60.452 | 54.001 |
| E1+gain6_cv66 | Meditron | 5 | 80.000 | 41.214 | 60.607 | 54.160 |
| E1+LLRD_cv21 | Meditron | 5 | 79.961 | 41.734 | 60.848 | 54.534 |
| E1_E3 | Meditron | 5 | 80.273 | 43.006 | 61.640 | 55.702 |
| E1_E3+gain6_cv66 | Meditron | 5 | 80.547 | 44.046 | 62.297 | 56.692 |
| E1_E3+LLRD_cv21 | Meditron | 5 | 80.449 | 43.988 | 62.219 | 56.589 |
| E1 | LittC2SE | 5 | 96.657 | 41.296 | 68.976 | 57.760 |
| E1+gain6_cv66 | LittC2SE | 5 | 96.542 | 41.862 | 69.202 | 58.274 |
| E1+LLRD_cv21 | LittC2SE | 5 | 96.196 | 41.619 | 68.908 | 57.994 |
| E1_E3 | LittC2SE | 5 | 97.349 | 42.834 | 70.091 | 59.368 |
| E1_E3+gain6_cv66 | LittC2SE | 5 | 97.464 | 43.968 | 70.716 | 60.445 |
| E1_E3+LLRD_cv21 | LittC2SE | 5 | 97.406 | 43.725 | 70.566 | 60.236 |
| E1 | Litt3200 | 5 | 88.424 | 36.026 | 62.225 | 49.708 |
| E1+gain6_cv66 | Litt3200 | 5 | 88.606 | 35.762 | 62.184 | 49.568 |
| E1+LLRD_cv21 | Litt3200 | 5 | 88.424 | 35.232 | 61.828 | 49.079 |
| E1_E3 | Litt3200 | 5 | 87.818 | 34.305 | 61.061 | 47.845 |
| E1_E3+gain6_cv66 | Litt3200 | 5 | 88.182 | 35.497 | 61.839 | 49.205 |
| E1_E3+LLRD_cv21 | Litt3200 | 5 | 87.939 | 34.305 | 61.122 | 47.972 |
| E1 | AKGC417L | 5 | 94.948 | 18.125 | 56.536 | 30.387 |
| E1+gain6_cv66 | AKGC417L | 5 | 94.498 | 17.326 | 55.912 | 29.228 |
| E1+LLRD_cv21 | AKGC417L | 5 | 94.571 | 17.422 | 55.997 | 29.368 |
| E1_E3 | AKGC417L | 5 | 94.906 | 18.359 | 56.633 | 30.725 |
| E1_E3+gain6_cv66 | AKGC417L | 5 | 94.665 | 17.899 | 56.282 | 30.062 |
| E1_E3+LLRD_cv21 | AKGC417L | 5 | 94.665 | 17.830 | 56.248 | 29.962 |

## Target adaptation: unlabelled spectrum, KAUH

| Filter | Raw Score mean ± SD | Adapted Score mean ± SD | Δ | Paired 95% CI |
| --- | ---: | ---: | ---: | --- |
| Bell | 61.94 ± 2.89 | 65.32 ± 2.29 | +3.38 | [-4.58, 11.38] |
| Diaphragm | 69.77 ± 1.50 | 69.03 ± 3.20 | −0.74 | [-6.99, 4.59] |
| Extended | 71.76 ± 2.10 | 71.20 ± 2.43 | −0.56 | [-6.32, 4.15] |

| Protocol | Thiết lập |
| --- | --- |
| Model | gain6cv66, seed 0–4, checkpoint cố định |
| Tách patient | 33 adaptation / 78 evaluation; cùng split cho ba filter |
| Adaptation | Ước lượng spectrum theo filter chỉ từ audio không nhãn của adaptation patients |
| Evaluation | Recording binary normal/abnormal, threshold 0,5, mean abnormal probability qua các window |

## Source label/boundary audio audit

| Hạng mục | Kết quả |
| --- | ---: |
| Source cycles / patients | 4.025 / 77 |
| Shared patients loại khỏi audit | 156, 218 |
| Candidate cần nghe | 50 |
| Invalid boundary | 0 |
| Confirmed label error | 0 |
| Unusual-duration flags | 38 |
| Clipping flags | 1.281 |
| Human listening | Chưa hoàn tất |

## Cấu hình riêng của từng idea

| Idea/arm | Thiết lập cụ thể |
| --- | --- |
| Alignment token attention | `method: ce_pafa_attn` |
| Gain6 | `random_gain_db: 6.0` |
| LLRD | `layer_decay: 0.8` |
| Classification top-k | `class_pool: topk`, `class_topk_fraction: 0.2` |
| Classification attention | `class_pool: attention` |
| Class-aware PAFA (screen) | `method: ce_pafa_class_aware`, PCSL 50 |
| Class-aware light (5×5 CV) | `method: ce_pafa_class_aware`, PCSL 5, GPAL 0,0005 |
| Patient cap | `patient_cap: true` |
| Recording head | `method: ce_recording_pafa`, bag 4 cycles, eval 8 cycles, loss weight 0,5, batch 8 |
| Recording head PCSL0 | Recording head với `lambda_pcsl: 0` |
| Band/noise | Augmentation p=0,5; device rate 4–8 kHz; SNR 20–35 dB; consistency weight 0 |
| Band/noise + consistency | Như trên; consistency weight 0,5; ramp 10 epoch |
| SWA | `swa_start: 10`, `swa_interval: 1`; average online model/classifier/projector |
| LoRA | `lora_rank: 8` |
| Multi-scale | `multiscale: true`; shared backbone |
| LoRA + class-aware | Rank 8; class-aware method; PCSL 5; GPAL 0,0005 |

## Decision logs

| Ngày | Quyết định | Cấu hình/đầu ra | Nguồn |
| --- | --- | --- | --- |
| 23/10 | Primary baseline: BEATs + PAFA no SC; AuscultTrust: BEATs + PAFA trained static SC | Epoch 100, seed 0–4; serving seed 0 | `docs/decision-logs/2026-10-23-final-model.md` |
| 25/10 | Chọn mean + gain6 trên source CV | Epoch 67, Score 61.084, Δ +0.959 so với control | `logs/improvement-20261025/cv-selection-next.json` |
| 26/10 | LLRD qua screen/confirm source gate; chạy final official + 4 LODO | Epoch 21, 5 seed mỗi cohort | `build/campaigns/improvements-20261026/logs/improvements-20261026/programme.json` |
| 26/10 | Giữ registered 8 s preprocessing | Source recording Score 71.769 | `outputs/evaluation/improvements-20261026/preprocessing/summary.json` |
| 26/10 | Learned fusion không fit | Thiếu full complementary OOF và validation độc lập; equal weights đã chạy | `outputs/evaluation/next-addons-20261026/fusion-feasibility.json` |
| 26/10 | Target adaptation đánh giá tách riêng; không thay deployment | 33/78 patient split, 5 seed | `outputs/evaluation/next-addons-20261026/summary.json` |
| 27/10 | Hoàn tất 150/150 source-CV fit | 5 fold × 5 seed × 6 arm; `complete.json` và `summary.json` hash khớp | `build/campaigns/fivefold-20261026/outputs/evaluation/fivefold-20261026/` |
| Theo yêu cầu người dùng | Bỏ external pretraining | 0 fit | user decision 26/10 (`report.md` section 3) |
