# Handoff (03/11/2026)

State of the repo after the consolidation of 03/11/2026. Read `../../report.md`, then `../REPRODUCE.md`.

## Done

- Every training campaign is finished; results are consolidated on this machine (`outputs/train/<run>/<cell>/<unit>/`, index `outputs/index.csv`).
- Checkpoints kept: final / report-epoch units of `grid-v1`, `lodo-v1`, `lodo-ast`, `lodo-bench`, `pcmcl-v1`, `sc-dynamic-completion-20261024`, `followup-20261026-final`, `target-20261027-final`, `distill-20261028-final`, `student-final-20261029-final`. Source-CV (`cv*`) checkpoints stay on the training machines and in `/storage/hdd/disk0/namdp15/auscult-remote-mirror/`.
- Code reduced to what reproduces the plan: grid, LODO, SC (train and inference), ensembles, distillation, replay, analysis notebooks, app. Removed: CLAP/HeAR encoders, noise-training arm, layer/device probes, SNR stress, grid-rerun and queue tooling, combo/fivefold/next campaigns.
- Archived (not deleted) at `/storage/hdd/disk0/namdp15/auscult-trust-archive/`: old outputs and checkpoints, campaign snapshots, ladders, old app bundles, removed docs.

## Open

1. Serving bundle: `app/bundle` still serves BEATs + PAFA static SC, not the student `kd_g4_k1`. Rebuild with `python -m api.bundle` after validating hashes and health.
2. Phantom H1, H3, co-primary H2: need transducer recordings. Weights and `s_ref` are frozen (`outputs/train/final-*/*/seed*/freeze.json`).
3. Frozen-evaluation provenance: refactored code changes the implementation hash, so old KAUH/screening exports under `outputs/kauh/final-*` and `outputs/screening/final-*` are rejected as stale by the strict workflows (`baseline/seed0` of `final-kd-g4-k1` was regenerated on 03/11). Re-export before reusing them.
4. Checkpoints of `lodo-v1` and `sc-dynamic-completion` units were copied from the machine mirrors; check `outputs/index.csv` (`ckpt` column) before inference runs.
5. Unreachable on 03/11: 10.148.183.171 (D) and 10.148.182.97 (laptop). Their results were already in the local mirror.

## Gotchas

- Never `pkill -f` / `pgrep -f` with a pattern that appears in your own command line: it kills the shell. Kill by PID.
- Do not run an unknown script with `--help`: `scripts/final_evaluation*.py` have no argument parser and start a run.
