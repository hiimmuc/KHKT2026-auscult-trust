# Fixed follow-up ensembles

Exploratory fixed static-SC comparisons. Historical LODO-v1 and new LODO-v2 training protocols differ; exact held-out cohort alignment is mandatory. No target-based member/epoch selection.

Equal weight per individual member; full-pool label-free k-rate matches historical PAFA normal rate; no threshold uses target labels

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

Common cohort was fixed without outcome selection across all seven members. Whole patients with unequal cycle sets are excluded from every group. Lost patient IDs and original/retained class counts are in receipt.json.

| Split | Seed | Retained cycles | Retained patients | Most patients lost by one member |
|---|---|---|---|---|
| official | 0 | 2756 | 49 | 0 |
| official | 1 | 2756 | 49 | 0 |
| official | 2 | 2756 | 49 | 0 |
| official | 3 | 2756 | 49 | 0 |
| official | 4 | 2756 | 49 | 0 |
| Meditron | 0 | 1370 | 63 | 0 |
| Meditron | 1 | 1370 | 63 | 0 |
| Meditron | 2 | 1370 | 63 | 0 |
| Meditron | 3 | 1370 | 63 | 0 |
| Meditron | 4 | 1370 | 63 | 0 |
| LittC2SE | 0 | 594 | 23 | 0 |
| LittC2SE | 1 | 594 | 23 | 0 |
| LittC2SE | 2 | 594 | 23 | 0 |
| LittC2SE | 3 | 594 | 23 | 0 |
| LittC2SE | 4 | 594 | 23 | 0 |
| Litt3200 | 0 | 481 | 10 | 0 |
| Litt3200 | 1 | 481 | 10 | 0 |
| Litt3200 | 2 | 481 | 10 | 0 |
| Litt3200 | 3 | 481 | 10 | 0 |
| Litt3200 | 4 | 481 | 10 | 0 |
| AKGC417L | 0 | 4216 | 31 | 0 |
| AKGC417L | 1 | 4216 | 31 | 0 |
| AKGC417L | 2 | 4216 | 31 | 0 |
| AKGC417L | 3 | 4216 | 31 | 0 |
| AKGC417L | 4 | 4216 | 31 | 0 |

Receipt status: complete. Pending artifacts/alignments: 0. See receipt.json for paired patient/seed CIs and recording coverage/uncertainty.
