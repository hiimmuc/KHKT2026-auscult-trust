# Setup and step-by-step guide

| Platform | What it can do | Where |
| --- | --- | --- |
| Linux with NVIDIA GPU (reference) | everything, including full training and DASS | sections 1-10 |
| Windows with NVIDIA GPU | training, inference, tests, app; DASS not available | section 11 |
| Windows CPU only | tests, data preparation, frozen inference, one smoke training unit | section 11 |
| Google Colab (GPU) | setup, data, tests, smoke training, export | section 12, `notebooks/colab_quickstart.ipynb` |

Every command runs from the repository root. `PY=.venv-train/bin/python` for training and inference, `.venv/bin/python` for tests and analysis. Sweeps and per-stage campaign commands are in [REPRODUCE.md](REPRODUCE.md).

## 1. Requirements

- Linux, `git`, `uv`, about 10 GB of disk, an NVIDIA GPU (training one unit needs about 11 GB of GPU memory).
- Node 20 or newer, only for the web app.
- Directory layout: this repo sits in a wrapper directory.

```
KHKT-CVA-2026/
├── auscult-trust/   this repo
└── data/            raw/ processed/ splits/ models/ cache/        (AT_DATA overrides)
```

From a git worktree, set `AT_DATA` to the wrapper's `data/`. Upstream model code is vendored in `third_party/` (no separate clone).

## 2. Environments

Two environments. Do not merge them: training pins `timm==0.4.5` on Python 3.10.

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt           # tests, evaluation, notebooks
uv venv .venv-train --python 3.10
uv pip sync --python .venv-train/bin/python \
    --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128 requirements-train.lock     # training, inference, API
```

`requirements-train.lock` is the resolved Linux/CUDA 12.8 environment; `third_party/SOURCES.json` records the upstream commits of the vendored code. DASS needs a third environment: `bash scripts/build_dass_env.sh`.

## 3. Data

```bash
bash scripts/download_data.sh      # ICBHI, KAUH, encoder weights; skips finished files
python scripts/prepare_data.py     # builds processed/ and the committed splits; idempotent
```

`bash scripts/download_data.sh --icbhi-only` downloads ICBHI alone (about 2 GB), which is enough for the baseline and the smoke runs.

Check: ICBHI 920 wav, 6,898 cycles, 126 patients; KAUH 336 wav (112 patients x 3 filters). HTS-AT weights are a manual download (see the header of `scripts/download_data.sh`). BEATs weights: `../data/models/beats/BEATs_iter3_plus_AS2M.pt`. Never edit `../data/raw/` or the committed splits.

## 4. Verify the install

```bash
.venv/bin/python -m pytest -q tests          # 300+ tests, about 10 s, no GPU
bash scripts/validate.sh                      # references, tests, API contract, web build
```

## 5. Train one model

A unit is one fit: one recipe, one seed. A recipe is a list of YAML files; later files override earlier ones, and the cell name is the stems joined by `+` without `base`. Results go to `outputs/train/<RUN_ID>/<cell>/seed<S>/`, weights to `checkpoints/train/<RUN_ID>/<cell>/seed<S>/`. A finished unit is skipped; a different config in an existing unit is refused. Reuse the same `RUN_ID` to resume after a crash (`last.pth`).

**Baseline: BEATs + PAFA, no SC, official split, epoch 100.**

```bash
export RUN_ID=my-run PYTHONPATH=.
C=experiments/train
.venv-train/bin/python -m auscult_trust.training.main \
    --config $C/base.yaml,$C/encoder_beats.yaml,$C/grid/pafa_beats.yaml \
    --cell beats+ft+pafa --selection fixed --report_epochs 100 --seed 0
```

Outputs: `report.json` (per-epoch metrics), `test_preds.npz` (per-epoch softmax), `train_args.json`. Weights: `report_epoch_100.pth`.

**With spectrum correction (SC) at training.** Add the SC recipes. `sc.yaml` estimates each device's spectrum and maps it to the mean training spectrum; `sc_static.yaml` bounds the coefficient to 50-2000 Hz; `sc_gain.yaml` adds a random smooth per-mel-bin gain (6 dB SD).

```bash
.venv-train/bin/python -m auscult_trust.training.main \
    --config $C/base.yaml,$C/encoder_beats.yaml,$C/grid/pafa_beats.yaml,$C/sc.yaml,$C/sc_static.yaml,$C/sc_gain.yaml \
    --cell beats+ft+pafa+sc+sc_static+sc_gain --selection fixed --report_epochs 100 --seed 0
```

The unit saves `sc_state.npz` and `s_ref_sha256`; inference needs both.

**Leave-one-device-out (LODO).** Append `$C/lodo_<device>.yaml` (`meditron`, `littc2se`, `litt3200`, `akgc417l`) to `--config` and `+lodo_<device>` to `--cell`. The held-out device is never seen in training, SC estimation or calibration.

```bash
.venv-train/bin/python -m auscult_trust.training.main \
    --config $C/base.yaml,$C/encoder_beats.yaml,$C/grid/pafa_beats.yaml,$C/lodo_meditron.yaml \
    --cell beats+ft+pafa+lodo_meditron --selection fixed --report_epochs 100 --seed 0
```

**Other encoders.** Swap `encoder_beats.yaml` for `encoder_ast`, `encoder_htsat`, `encoder_opera_ct` or `encoder_dass`, and `grid/pafa_beats.yaml` for another method in `experiments/train/grid/`. Recipe table: [experiments/train/README.md](../experiments/train/README.md). Add `--epochs 2` for a smoke run.

## 6. Inference and evaluation of a trained unit

```bash
U=outputs/train/$RUN_ID/beats+ft+pafa/seed0
# 6.1 Freeze: softmax and embeddings on the official test set, tied to the weight hash.
.venv-train/bin/python -m auscult_trust.training.export_probs $U --ckpt report_epoch_100.pth        # -> $U/export/test.npz, model.sha256
# 6.2 Recording-level screening: 8 s windows, p(abnormal), 2-class split conformal, triage.
.venv-train/bin/python -m auscult_trust.pipelines.screening_infer $U --ckpt report_epoch_100.pth \
    --out outputs/screening/my-run/baseline/seed0.npz                                      # add --sc for SC-trained units
.venv-train/bin/python -m auscult_trust.evaluation.screening --pairs p0:p0
# 6.3 KAUH: filter-pair conformal coverage with frozen models (needs a screen.json of the run).
bash scripts/run_kauh.sh
```

**SC at inference only (no SC at training).** Test-time SC, A2 and TTA-EQ on trained LODO units, then the H2 tables:

```bash
RUN=lodo-ast SC_CELL=sc+sc_static bash scripts/run_lodo_infer.sh        # auscult_trust/pipelines/lodo_ttsc.py
bash scripts/lodo_h2_all.sh && .venv/bin/python scripts/lodo_tables.py > lodo_tables.md
```

**Frozen baseline vs SC comparison** (KAUH A/B2/B1 and screening, five paired seeds): `scripts/final_evaluation.py`. Exports carry `.provenance.json` sidecars that bind them to data, config, weights, SC state, source and environment; a stale sidecar is rejected and the export must be redone.

## 7. Ensemble

Softmax averaging of units that share a seed (and LODO fold). Equal weights are the default.

```bash
# Registered groups on grid-v1 / lodo-v1 (markdown to stdout):
PYTHONPATH=. .venv/bin/python scripts/ensemble.py > ensemble_tables.md
PYTHONPATH=. .venv/bin/python scripts/ensemble_sesp.py > ensemble_sesp.md      # Se at PAFA's Sp
```

Ensemble of your own units (last epoch, official test split):

```python
import numpy as np
from auscult_trust.training.reporting import per_device_report

units = [f"outputs/train/{RUN_ID}/beats+ft+pafa/seed{s}" for s in (0, 1, 2)]   # RUN_ID: your run id
Z = [np.load(f"{u}/test_preds.npz") for u in units]
p = np.mean([z["probs"][-1].astype(np.float64) for z in Z], 0)
r = per_device_report(Z[0]["labels"], p.argmax(1), Z[0]["device"])["all"]
print({k: round(float(r[k]), 2) for k in ("sp", "se", "score")})
```

Frozen ensembles on KAUH and screening: `scripts/final_ensembles.py`; fixed additions: `scripts/followup_ensembles.py template`, then `evaluate <manifest>`.

## 8. Distillation

One student learns from the equal-weight softmax of four teachers of the same seed (anchor `gain6-cv66`, `pcsl5_control`, `class_aware_light`, `multiscale`). The student is the SC recipe of section 5 with `--soft_targets`; its hard-label weight is zero (K1) and it stops at epoch 66.

```bash
S=0; T=outputs/train; K=checkpoints/train
mkdir -p $T/my-run/kd_g4_k1/seed$S
# 8.1 Export the teachers' soft targets on the training cycles (needs the four teacher units of seed $S).
.venv-train/bin/python -m auscult_trust.training.distillation \
    --units $T/followup-20261026-final/gain6-cv66/seed$S $T/target-20261027-final/pcsl5_control/seed$S \
            $T/target-20261027-final/class_aware_light/seed$S $T/target-20261027-final/multiscale/seed$S \
    --checkpoints $K/followup-20261026-final/gain6-cv66/seed$S/report_epoch_66.pth \
            $K/target-20261027-final/pcsl5_control/seed$S/report_epoch_78.pth \
            $K/target-20261027-final/class_aware_light/seed$S/report_epoch_4.pth \
            $K/target-20261027-final/multiscale/seed$S/report_epoch_68.pth \
    --out $T/my-run/kd_g4_k1/seed$S/teacher.npz
# 8.2 Fit the student on those targets.
RUN_ID=my-run .venv-train/bin/python -m auscult_trust.training.main \
    --config $C/base.yaml,$C/encoder_beats.yaml,$C/grid/pafa_beats.yaml,$C/sc.yaml,$C/sc_static.yaml,$C/sc_gain.yaml \
    --cell kd_g4_k1 --seed $S --epochs 100 --stop_epoch 66 --selection fixed --report_epochs 66 \
    --soft_targets $T/my-run/kd_g4_k1/seed$S/teacher.npz
```

The teacher epochs above are the ones of the registered final units; train your own teachers with section 5 and the recipes `experiments/train/{pcsl5_control,class_aware_light,multiscale}.yaml` (overlaid on the SC recipe), and pass their report epochs. The no-SC twin adds `$C/student_nosc.yaml`. For a LODO student, use only teachers trained without that device and add `--holdout_device <Device>`.

At scale: `scripts/distill_campaign.py prepare` then `run <manifest>`; `scripts/student_final_campaign.py` for `kd_g4_k1` and its twin; `scripts/student_final_report.py` for the report. Decision: [decision log](decision-logs/2026-10-29-final-model-distilled.md).

## 9. Run the app

The API serves frozen models from a bundle; with no bundle it starts, reports `unavailable` and answers 503. Details, deployment and security: [app/README.md](../app/README.md).

```bash
uv pip install --python .venv-train/bin/python -r api/requirements.txt
PYTHONPATH=. .venv-train/bin/python -m api.bundle                      # writes app/bundle/ (manifest, SC state, 12 sample WAVs)
PYTHONPATH=. .venv-train/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000
curl -s localhost:8000/v1/health                                       # {"status":"ok","pipelines":[...],"gpu":true}
```

In a second terminal:

```bash
cd app/web && npm ci && npm run dev        # http://127.0.0.1:5173, default API http://127.0.0.1:8000
```

Analyse one recording from the shell:

```bash
curl -s -F file=@recording.wav -F device=Litt3200 localhost:8000/v1/infer | python -m json.tool
```

`api.bundle` builds the pipelines listed in `PIPELINES` (`api/bundle.py`), currently BEATs + PAFA with and without static SC. The distilled student `kd_g4_k1` is not in the bundle yet: [Pending] add its unit and screening export to `PIPELINES`, then rebuild. Production build: `cd app/web && npm run build`.

## 10. Notebooks and index

```bash
cd notebooks && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace 0N_*.ipynb
PYTHONPATH=. .venv/bin/python scripts/index_results.py                 # outputs/index.csv: one row per unit
```

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `timm` version error | Use `.venv-train`. |
| ICBHI certificate error | `download_data.sh` falls back to the Dataverse mirror. |
| `official_split.txt` not found | Rerun `download_data.sh` (Patch-Mix CL repo missing). |
| CPU extraction hogs the machine | Prefix `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4`. |
| Unit refused: config differs | The unit exists with another config. Use a new `RUN_ID` or `--cell`. |
| Export rejected: stale provenance | Code or weights changed since the export. Redo the export. |
| GPU out of memory | Lower `--num_workers`, or run one job at a time (`MEM_GB` gates `scripts/sweep.py`). |

## 11. Windows

Use the commands below in PowerShell from the repository root. Bash scripts (`scripts/*.sh`) need Git Bash or WSL2; the data download needs `unzip` and `md5sum`, so WSL2 is the simplest route for section 3. Everything else is Python.

**Environments.** Install [uv](https://docs.astral.sh/uv/) (`winget install astral-sh.uv`), then:

```powershell
uv venv .venv --python 3.10
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
uv venv .venv-train --python 3.10
```

Training, GPU (NVIDIA driver with CUDA 12.8 support; check `nvidia-smi`):

```powershell
uv pip sync --python .venv-train\Scripts\python.exe --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128 requirements-train-windows-cu128.lock
```

Training, CPU only:

```powershell
uv pip sync --python .venv-train\Scripts\python.exe --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu requirements-train-windows-cpu.lock
```

Check the install: `.venv-train\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"`. The CPU lock prints `False`, as expected.

**Data.** Put the data next to the repo as in section 1, or set `AT_DATA`. Get ICBHI from the Dataverse mirror (`https://dataverse.harvard.edu/api/access/datafile/7127117`, md5 `f6a5d0ea40bbcd25e17754f9d5ddae48`) and unzip it to `..\data\raw\icbhi\ICBHI_final_database`. Then:

```powershell
.venv\Scripts\python.exe scripts\prepare_data.py
```

On Windows without Developer Mode, `prepare_data.py` copies the wav files instead of linking them. BEATs weights (`BEATs_iter3_plus_AS2M.pt`) go to `..\data\models\beats\`; get them from the [BEATs release](https://github.com/microsoft/unilm/tree/master/beats). KAUH, AST, DASS and OPERA weights are needed only for their experiments.

**Run.** Set the environment in the same window, then train. Use `--num_workers 0` on Windows (process spawning makes data workers slow and fragile):

```powershell
$env:PYTHONPATH="."; $env:RUN_ID="my-run"; $C="experiments\train"
.venv-train\Scripts\python.exe -m auscult_trust.training.main --config $C\base.yaml,$C\encoder_beats.yaml,$C\grid\pafa_beats.yaml --cell beats+ft+pafa --selection fixed --report_epochs 2 --epochs 2 --seed 0 --num_workers 0
.venv-train\Scripts\python.exe -m auscult_trust.training.export_probs outputs\train\my-run\beats+ft+pafa\seed0 --ckpt report_epoch_2.pth --num_workers 0
```

Two-epoch smoke run as above first. Full runs (100 epochs) need about 11 GB of GPU memory and take hours on a consumer GPU. On CPU, one training epoch of the baseline took 1409 s (about 24 min) on a 32-core Linux machine with `OMP_NUM_THREADS=4`, plus about 5 min for validation and export; a laptop will be slower. Treat CPU as a smoke-test device only.

**Known limits on Windows**

- DASS needs the CUDA selective-scan kernel (`scripts/build_dass_env.sh`, Linux only). Do not use `encoder_dass`.
- `scripts/campaign.py`, `followup_campaign.py`, `improvement_campaign.py` use `fcntl` (POSIX file locks). Run them on Linux or WSL2. Their tests are skipped on Windows (`tests/conftest.py`).
- No `latest` symlink: pass `--run <run_id>` to `auscult_trust.training.summary` explicitly.
- Long paths: nested cell names can exceed 260 characters. Run `git config --global core.longpaths true` before cloning, or keep the repo near the drive root.
- Tests: `.venv\Scripts\python.exe -m pytest -q tests`.

## 12. Google Colab

Open `notebooks/colab_quickstart.ipynb` in Colab (File > Upload notebook, or Drive). Then:

1. Runtime > Change runtime type > GPU.
2. Upload the code zip (see `DELIVERY.md`) to `MyDrive/auscult-trust-delivery.zip`.
3. Run the cells in order. The notebook checks for a GPU and stops if none is found, creates a Python 3.10 environment with the same pinned lock as section 2, downloads ICBHI, checks the BEATs weights, runs the tests, trains the two-epoch smoke unit and exports its predictions.
4. Place the BEATs file at `MyDrive/auscult-trust-data/models/beats/BEATs_iter3_plus_AS2M.pt` before the data cell; the notebook prints the exact path if it is missing.

With Drive mounted, data lives in `MyDrive/auscult-trust-data` and outputs in `MyDrive/auscult-trust-outputs`, so a session reset loses nothing. Set `RUN_FULL = True` in the training cell for the 100-epoch baseline (several hours on a T4).

The notebook was executed end-to-end on a Linux machine with the same steps (CPU override `ALLOW_CPU=1`, one epoch). It has not been run in a Colab session yet. [Pending]

