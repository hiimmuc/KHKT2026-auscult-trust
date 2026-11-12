# Repository Guidelines

## Project Structure & Module Organization

`src/processing/` handles audio, datasets, patient splits, and spectrum correction; `src/training/` implements training/export; `src/inference/` shares frozen prediction and adaptation between serving and experiments; `src/evaluation/` contains metrics and conformal analysis. Training recipes live in `experiments/train/`, automation in `scripts/`, and Python tests in `tests/`. `api/` serves FastAPI inference; `app/web/` contains the Vite/TypeScript UI and browser checks. Consult `docs/SETUP.md`, `docs/REPRODUCE.md`, `report.md`, `docs/handoff/HANDOFF.md` and `app/README.md` for setup, reproduction commands, results, current state, and serving details. The current proposal is `docs/registered/research-proposal-v10-en.md`; the registered study is v9.

The parent directory is an unversioned wrapper: `../data/` holds datasets, and `../archive/` contains frozen Phase 0 work. Keep generated results in ignored `outputs/`, `checkpoints/`, and `logs/`.

## Build, Test, and Development Commands

Run from this repository unless specified:

- `uv pip install --python .venv/bin/python -r requirements.txt`: install analysis/test dependencies after creating `.venv`.
- `uv pip install --python .venv-train/bin/python -r requirements-train.txt`: install training dependencies into a separate Python 3.10 environment.
- `.venv/bin/python -m pytest -q tests`: run the research test suite.
- `bash scripts/run.sh screen`: start CV screening; subsequent campaign stages are `summary`, `final`, and `report`. Preserve `RUN_ID` when resuming.
- `PYTHONPATH=. .venv-train/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000`: serve a prepared model bundle; install `api/requirements.txt` first.
- `cd app/web && npm ci && npm run dev`: start the UI; `npm run build` type-checks and builds it.

## Coding Style & Naming Conventions

Follow nearby code: four-space Python indentation, snake_case functions/modules, and two-space TypeScript indentation with camelCase identifiers. Write comments and documentation in English. Prefer small changes and existing helpers. No repository-wide formatter or lint command is configured; the frontend build checks TypeScript.

## Testing Guidelines

Use pytest files named `test_*.py` and functions named `test_*`. Add regression coverage for changed numerical behavior and patient-split logic. API contract checks run with `.venv-train/bin/python -m pytest -q api/test_contract.py`; real-model tests require a bundle and GPU. Browser checks are documented in `app/README.md`. No numerical coverage threshold is configured.

## Commit & Pull Request Guidelines

History mixes descriptive subjects with prefixes such as `docs:` and `api:`. Use a concise, scoped subject. PRs should explain behavior changes, validation commands/results, and relevant issues or protocol decisions; include screenshots for UI changes.

## Research & Agent Rules

Preserve raw data and committed splits. Tune only on training-patient CV; retain negative results and log seeds, configuration, and artifact hashes. Never commit weights or generated results. Read `CLAUDE.md` for research conventions. When a CodeGraph index is available, query it before locating or reading code; never index automatically.

## Current Evidence and Artifacts (3 November 2026)

- Results are consolidated in `outputs/train/<run>/<cell>/<unit>/` (index: `outputs/index.csv`); weights of final units in `checkpoints/train/`. Old runs, snapshots and removed code are archived at `/storage/hdd/disk0/namdp15/auscult-trust-archive/`; per-machine mirrors are in `/storage/hdd/disk0/namdp15/auscult-remote-mirror/`.
- AuscultTrust is the distilled student `kd_g4_k1`; the baseline is BEATs + PAFA without SC (`docs/decision-logs/2026-10-29-final-model-distilled.md`). Results: `report.md`. State: `docs/handoff/HANDOFF.md`.
- Stage-12 source-CV gains (class-aware light, LoRA) did not transfer to official or LODO; do not present source gains as target-device improvement. Keep registered v9 results and later exploratory selections distinct.
- Verify completion receipts and SHA-256 before using or deleting a checkpoint.
