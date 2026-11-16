# Delivery

Source tree at one git commit, packaged with `git archive` from a clean tree:

```bash
git archive --format=zip --prefix=auscult-trust/ -o ../auscult-trust-delivery.zip HEAD
```

## Contents

- Code: `auscult_trust/`, `api/`, `scripts/`, `experiments/` (recipes), `tests/`, `app/`, `third_party/` (vendored upstream, pinned in `third_party/SOURCES.json`).
- Environments: `requirements.txt` (analysis), `requirements-train.lock` (Linux, CUDA 12.8), `requirements-train-windows-cu128.lock`, `requirements-train-windows-cpu.lock`.
- Notebooks: `colab_quickstart.ipynb` (setup, data, tests, smoke training, export) and five analysis notebooks with cleared outputs.
- Docs: `docs/`, `report.md`.

## Not included

| Item | How to obtain |
| --- | --- |
| ICBHI, KAUH, phantom recordings (dataset terms) | `scripts/download_data.sh`, `docs/SETUP.md` section 3 |
| Pretrained encoder weights (BEATs, AST, DASS, OPERA, HTS-AT) | Links in `scripts/download_data.sh` |
| `outputs/`, `checkpoints/` | Train with `docs/SETUP.md` section 5, or use the final weights below |
| `build/`, `logs/`, `.venv*`, `node_modules/`, `app/bundle/` | Created by setup |

Final weights (AuscultTrust `kd_g4_k1` seed 0, baseline BEATs + PAFA seed 0, with SHA-256): GitHub Release `weights-final`, asset `weight-final.zip`. Check with `sha256sum -c SHA256SUMS.txt` after unzipping.

The analysis notebooks `01`-`04` read finished runs and stop with a message naming the missing ones. `00` reads only raw data.

## Verified (Linux, Python 3.10.12)

| Check | Result |
| --- | --- |
| `pytest -q tests` (analysis env) | 305 passed, 1 skipped |
| `pytest -q tests api/test_contract.py` (train env) | 322 passed, 1 skipped |
| Same suites in the extracted archive (no data, no outputs) | 302 / 319 passed, 4 skipped |
| `scripts/validate.sh` | exit 0 |
| `colab_quickstart.ipynb`, clean tree, CPU, 1 epoch | 0 error outputs; test Score 34.00 (smoke value, not a result) |

Not verified: Windows (commands follow from the lock files and code paths), a real Colab session, the notebook on CUDA (the GPU was occupied during testing).

## Limits

- DASS needs a CUDA kernel: Linux only.
- Campaign drivers in `scripts/` use `fcntl`: Linux or WSL2.
- One full training unit needs about 11 GB of GPU memory.
- `experiments/campaigns/*.json` hold the author's local paths. They are historical records; `tests/infrastructure/test_campaign.py` reads them.
- The Pages worker needs `API_UPSTREAM` at deploy (`app/README.md`).
