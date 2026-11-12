# Vendored upstream code

Byte-identical copies of the upstream files this project imports, at the pinned commits in `SOURCES.json`. Only `*.py` files,
licence files and the files named below are kept; weights and datasets are not included. Do not edit these files: the
serving-bundle and prediction certificates hash them, and `auscult_trust/models/upstream.py` loads them by path.

Needed beyond Python sources: `patch-mix_contrastive_learning/data/icbhi_dataset/official_split.txt` (the official 60/40 split)
and `DASS/kernels/selective_scan` (the CUDA kernel built by `scripts/build_dass_env.sh`). `BEATs` inside `PAFA/` is Microsoft's
BEATs from https://github.com/microsoft/unilm/tree/master/beats, MIT licence.

| directory | upstream | commit | licence |
|---|---|---|---|
| `PAFA/` | https://github.com/wa976/PAFA.git | `e49e294d0db0` | none published upstream |
| `ADD-RSC/` | https://github.com/deegy666/ADD-RSC.git | `e2b0f8213cb7` | none published upstream |
| `patch-mix_contrastive_learning/` | https://github.com/raymin0223/patch-mix_contrastive_learning.git | `836b09fea1b7` | `LICENSE` in the folder |
| `QLung/` | https://github.com/RSC-Toolkit/QLung.git | `a1a01041855d` | none published upstream |
| `SG-SCL/` | https://github.com/kaen2891/stethoscope-guided_supervised_contrastive_learning.git | `2ed1bacc7121` | none published upstream |
| `Lung-SRAD/` | https://github.com/RSC-Toolkit/Lung-SRAD.git | `6df75ce07646` | none published upstream |
| `OPERA/` | https://github.com/evelyn0414/OPERA.git | `3622310e667a` | `LICENSE.txt` in the folder |
| `DASS/` | https://github.com/Saurabhbhati/DASS | `246c9f419cae` | `LICENSE` in the folder |

Redistribution: upstream publishes no licence for PAFA, ADD-RSC, QLung, SG-SCL and Lung-SRAD, so these copies are kept for
reproducibility on the project owner's responsibility. Replace a copy by a pinned git dependency if upstream objects.
