# AuscultTrust

Lung-sound classification that keeps its predictions and its split-conformal coverage when the stethoscope changes. Method: spectrum correction (SC) plus a conformal layer, on ICBHI with KAUH and phantom-replay checks.

AuscultTrust is the distilled student `kd_g4_k1`. The baseline is fine-tuned BEATs + PAFA without SC. Results: [report.md](report.md). Research prototype for sound classification; not a medical device, no diagnosis.

## Quick start

```bash
# 1. Environments (details: docs/SETUP.md sections 1-2)
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
uv venv .venv-train --python 3.10
uv pip sync --python .venv-train/bin/python --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128 requirements-train.lock
.venv-train/bin/python -m src.references --install

# 2. Data (ICBHI, KAUH, encoder weights) and tests
bash scripts/download_data.sh && python scripts/prepare_data.py
.venv/bin/python -m pytest -q tests

# 3. Train one model: BEATs + PAFA baseline, official split, seed 0
export RUN_ID=my-run PYTHONPATH=. C=experiments/train
.venv-train/bin/python -m src.training.main \
    --config $C/base.yaml,$C/encoder_beats.yaml,$C/grid/pafa_beats.yaml \
    --cell beats+ft+pafa --selection fixed --report_epochs 100 --seed 0

# 4. Inference on the official test set
.venv-train/bin/python -m src.training.export_probs outputs/train/$RUN_ID/beats+ft+pafa/seed0 --ckpt report_epoch_100.pth

# 5. App: API on :8000, web on :5173
uv pip install --python .venv-train/bin/python -r api/requirements.txt
PYTHONPATH=. .venv-train/bin/python -m api.bundle
PYTHONPATH=. .venv-train/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000 &
(cd app/web && npm ci && npm run dev)
```

## Guides

| Step | Where |
| --- | --- |
| Environment, data, checks | [docs/SETUP.md](docs/SETUP.md) sections 1-4 |
| Train one model (baseline, SC, LODO, other encoders) | SETUP section 5 |
| Inference, screening, KAUH, SC at inference | SETUP section 6 |
| Ensemble | SETUP section 7 |
| Distillation | SETUP section 8 |
| Run the app | SETUP section 9, [app/README.md](app/README.md) (deployment) |
| Sweeps and campaigns per stage (70-cell grid, LODO, ensembles, distillation, phantom replay) | [docs/REPRODUCE.md](docs/REPRODUCE.md) |
| Results by stage | [report.md](report.md) |
| Current state, open items | [docs/handoff/HANDOFF.md](docs/handoff/HANDOFF.md) |
| Docs index, registered study, plans, decisions | [docs/README.md](docs/README.md) |
| Working rules | [CLAUDE.md](CLAUDE.md) |

## Layout

| Path | Content |
| --- | --- |
| `src/processing/` | Audio, ICBHI/KAUH, patient splits, spectrum correction |
| `src/training/` | Training loop, encoders (AST, BEATs, HTS-AT, OPERA-CT, DASS), methods, distillation |
| `src/inference/` | Frozen prediction, test-time SC |
| `src/evaluation/` | Metrics, conformal, LODO, KAUH, screening, ensembles |
| `experiments/train/` | YAML recipes, composed left to right |
| `scripts/` | Grid, campaigns, frozen evaluation, ensembles, replay analysis |
| `notebooks/` | Analysis of finished outputs |
| `api/`, `app/` | FastAPI service, web demo |
| `outputs/train/<run>/<cell>/<unit>/` | Reports, predictions; `outputs/index.csv` lists every unit |
| `checkpoints/train/<run>/<cell>/<unit>/` | Weights |

`../data` holds datasets, `../repos` the read-only reference code.
