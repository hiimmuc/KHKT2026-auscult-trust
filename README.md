# AuscultTrust

Lung-sound classification that keeps its predictions and its split-conformal coverage when the stethoscope changes. Method: spectrum correction (SC) plus a conformal layer, on ICBHI with KAUH and phantom-replay checks.

AuscultTrust is the distilled student `kd_g4_k1`. The baseline is fine-tuned BEATs + PAFA without SC. Results: [report.md](report.md). Research prototype for sound classification; not a medical device, no diagnosis.

## Layout

| Path | Content |
| --- | --- |
| `src/processing/` | Audio, ICBHI/KAUH, patient splits, spectrum correction |
| `src/training/` | Training loop, encoders (AST, BEATs, HTS-AT, OPERA-CT, DASS), methods, distillation |
| `src/inference/` | Frozen prediction, test-time SC |
| `src/evaluation/` | Metrics, conformal, LODO, KAUH, screening, ensembles |
| `experiments/train/` | YAML recipes (compose left to right) |
| `scripts/` | Grid, campaigns, frozen evaluation, ensembles, replay analysis |
| `notebooks/` | Analysis of finished outputs |
| `api/`, `app/` | FastAPI service, web demo |
| `outputs/train/<run>/<cell>/<unit>/` | Reports, predictions (`outputs/index.csv` lists every unit) |
| `checkpoints/train/<run>/<cell>/<unit>/` | Weights |

`../data` holds datasets, `../repos` the read-only reference code.

## Reproduce

Setup: [docs/SETUP.md](docs/SETUP.md). Commands per stage: [docs/REPRODUCE.md](docs/REPRODUCE.md).

| Stage | Run |
| --- | --- |
| Encoders x methods, official split | `scripts/grid.py` (`grid-v1`) |
| LODO training | `scripts/grid.py lodo-jobs`, `scripts/lodo_h2_all.sh`, `scripts/lodo_tables.py` |
| SC at training | `sc.yaml` (+ `sc_static.yaml`) on any cell, official and LODO |
| SC at inference | `scripts/run_lodo_infer.sh`, `scripts/run_kauh.sh` |
| Ensembles | `scripts/ensemble.py`, `scripts/followup_ensembles.py`, `scripts/final_ensembles.py` |
| Distillation | `scripts/distill_campaign.py`, `scripts/student_final_campaign.py` |
| Phantom replay | `scripts/replay_analysis.py`, `notebooks/04_replay_phantom.ipynb` |

Tests: `uv run --no-project --python .venv/bin/python python -m pytest -q tests`.

## Docs

- [report.md](report.md): results by stage.
- [docs/README.md](docs/README.md): index. Registered study and amendments in `docs/registered/`, plans in `docs/plans/`, decisions in `docs/decision-logs/`.
- [app/README.md](app/README.md): demo deployment.
- [CLAUDE.md](CLAUDE.md): working rules.
