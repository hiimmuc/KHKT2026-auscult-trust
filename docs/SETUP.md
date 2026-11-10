# Setup

## Layout

`auscult-trust/` sits in a wrapper directory. Data and reference code live beside it.

```text
KHKT-CVA-2026/
├── auscult-trust/   this repo
├── data/            raw/ processed/ splits/ models/ cache/        (AT_DATA overrides)
└── repos/           read-only OPERA, Patch-Mix CL, SG-SCL, PAFA, ...  (AT_REPOS overrides)
```

Needs Linux, `uv`, `git`, about 10 GB of disk and an NVIDIA GPU (CUDA 12.8 build of `torch`).

## Environments

Two envs; do not merge them.

```bash
uv venv .venv && uv pip install -r requirements.txt                       # tests, evaluation, notebooks
uv venv .venv-train --python 3.10                                          # training (timm==0.4.5)
uv pip sync --python .venv-train/bin/python --extra-index-url https://download.pytorch.org/whl/cu128 requirements-train.lock
.venv-train/bin/python -m src.references --install                         # reference repos at pinned commits
```

`requirements-train.lock` is the resolved Linux/CUDA environment; `requirements-references.lock.json` pins the reference repos.

## Data

```bash
bash scripts/download_data.sh      # ICBHI, KAUH, encoder weights, reference repos (skips finished files)
python scripts/prepare_data.py     # idempotent
```

Expected: ICBHI 920 wav, 6,898 cycles, 126 patients; KAUH 336 wav (112 patients x 3 filters). HTS-AT weights are a manual download (see the header of `scripts/download_data.sh`).
BEATs weights: `data/models/beats/BEATs_iter3_plus_AS2M.pt`.

## Check

```bash
uv run --no-project --python .venv/bin/python python -m pytest -q tests
bash scripts/validate.sh           # references, tests, API contract, web build
```

From a git worktree set `AT_DATA` and `AT_REPOS` to the wrapper's `data/` and `repos/`.

## Run layout and provenance

One run = one `RUN_ID` (`export RUN_ID=...` again to resume). Results go to `outputs/train/<run>/<cell>/<unit>/`, weights to `checkpoints/train/<run>/<cell>/<unit>/`. A finished unit is skipped; a different config in an existing unit is refused. Every SC run saves `sc_state.npz` and `s_ref_sha256`. Exports write `.provenance.json` sidecars binding outputs to data, config, weights, SC state, source and environment; stale sidecars are rejected and need a re-export.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `timm` version error | Use `.venv-train`. |
| ICBHI certificate error | `download_data.sh` falls back to the Dataverse mirror. |
| `official_split.txt` not found | Rerun `download_data.sh` (Patch-Mix CL repo missing). |
| CPU extraction hogs the machine | Prefix `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4`. |
