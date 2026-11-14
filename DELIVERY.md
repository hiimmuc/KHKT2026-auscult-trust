# Delivery

The delivery is the source tree at one git commit, packaged with `git archive`. It contains code, tests, recipes, the report and the documentation needed to reproduce the results. It does not contain datasets, model weights or generated results.

## Build the package

```bash
git archive --format=zip --prefix=auscult-trust/ -o ../auscult-trust-delivery.zip HEAD
```

Run it from a clean commit (`git status` empty). Uncommitted changes are not in the archive.

## Included

- `auscult_trust/`, `api/`, `scripts/`, `experiments/` (training recipes), `tests/`, `third_party/` (vendored upstream code, byte-identical to the pinned commits in `third_party/SOURCES.json`).
- Lock files: `requirements.txt` (analysis), `requirements-train.lock` (Linux, CUDA 12.8), `requirements-train-windows-cu128.lock` and `requirements-train-windows-cpu.lock` (Windows, resolved from `requirements-train.txt`, not installed on Windows here).
- `notebooks/colab_quickstart.ipynb` (self-contained; runs setup, data, tests, smoke training, export) and the five analysis notebooks (outputs cleared; they read finished runs, see below).
- `app/` (web demo source), `docs/` (setup, reproduction, decision logs, registered protocol), `report.md`.
- Committed split file `third_party/patch-mix_contrastive_learning/data/icbhi_dataset/official_split.txt`.

## Not included, and why

| Item | Reason | How to obtain |
| --- | --- | --- |
| ICBHI, KAUH, phantom recordings | Dataset terms; not redistributed | `scripts/download_data.sh`, section 3 of `docs/SETUP.md` |
| BEATs, AST, DASS, OPERA, HTS-AT weights | Upstream licences and size | Links in `scripts/download_data.sh` and `docs/SETUP.md` |
| Trained outputs, checkpoints (`outputs/`, `checkpoints/`) | Weights and generated results stay out of git | Reproduce with `docs/SETUP.md` section 5, or ask for the weights archive (see open items) |
| `build/`, `logs/`, `.venv*`, `node_modules/`, `app/bundle/` | Generated | Created by the setup steps |
| Notebook outputs | Embedded local paths and images | Run the notebooks |

The analysis notebooks (`00`-`04`) read the finished runs. Without them they stop at the first cell with a message naming the missing runs. `00_dataset_eda.ipynb` reads only the raw data.

## Verification

Verified on Linux (Ubuntu, NVIDIA RTX 5000 Ada, Python 3.10.12):

| Check | Result |
| --- | --- |
| `pytest -q tests` (analysis env) | 305 passed, 1 skipped |
| `pytest -q tests api/test_contract.py` (train env) | 322 passed, 1 skipped |
| `scripts/validate.sh` (pip check, tests, web audit and build) | exit 0 |
| `colab_quickstart.ipynb` executed end to end on a clean copy of the tree (installs, data, tests, 1 training epoch, export), CPU override | 0 error outputs; test Score 34.00 after 1 epoch (smoke value, not a result) |

Not verified: Windows (no Windows machine here; the Windows commands are written from the lock files and the code paths), Colab (the notebook was executed locally, not in a Colab session), the GPU run of the notebook (the GPU was occupied by another process during testing: a 28 GB vLLM engine, which the notebook's CUDA run ran out of memory against).

## Known limits

- DASS encoder: CUDA kernel, Linux only.
- Campaign drivers in `scripts/` use `fcntl`: Linux or WSL2 only.
- A GPU with about 11 GB of memory is needed for one full training unit.
- `app/web/public/_worker.js` defaults `DEFAULT_UPSTREAM` to the author's private tunnel host. Set it (or the deployment env) to your own API before publishing; a decision for the owner, not changed here.
- `experiments/campaigns/*.json` (manifests of past campaigns) contain the author's local paths; `tests/infrastructure/test_campaign.py` reads them. Kept as a historical record.

## Open items

- Weights archive for `kd_g4_k1` (AuscultTrust) and the baseline (BEATs + PAFA, seed 0), with SHA-256 receipts. Not created. Needs a decision on where to host it.
