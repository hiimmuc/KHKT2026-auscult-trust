# Analysis notebooks

Read-only analysis of finished artefacts. Run from `notebooks/` with the analysis kernel (`.venv`, kernel `auscult-analysis`):
`../.venv/bin/jupyter nbconvert --to notebook --execute --inplace <name>.ipynb`. Shared paths, loaders and the arm table are in
`common.py`. Figures: `outputs/figures/<topic>/`; tables: `outputs/analysis/<topic>/` (both ignored by git).

| Notebook | Content | Figures / tables |
| --- | --- | --- |
| `00_dataset_eda.ipynb` | ICBHI size, split, class and device balance, durations, patients, device spectra, example cycles, LODO cohorts, KAUH, replay set | `eda` |
| `01_official_split.ipynb` | train / validation (source CV) / test on the official split: scores per arm, paired deltas, curves, confusion, per-class and per-device, reliability | `official` |
| `02_lodo.ipynb` | four-device LODO: Score and macro per arm, paired change, Sp/Se, curves, confusion | `lodo` |
| `03_kauh_screening.ipynb` | KAUH invariance (flip, TV), B2 coverage, window histograms, recording screening | `kauh_screening` |
| `04_replay_phantom.ipynb` | phantom replay: take quality, transfer functions, spectrograms, processing variants, selection | `replay` |

The replay results come from `scripts/replay_analysis.py` (`PYTHONPATH=. .venv-train/bin/python scripts/replay_analysis.py --takes ../data/raw/phantom_replay --out outputs/replay/phantom-replay-20261028`).
Campaign outputs are read from `build/campaigns/<campaign>/outputs/train/...` (per-unit `report.json` and `test_preds.npz`; copy them
from machine B if absent locally). Final units have no validation split; validation curves are the source-CV runs of `fivefold-20261026`.
