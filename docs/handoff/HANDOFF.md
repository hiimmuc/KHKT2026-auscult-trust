# Handoff (16/11/2026)

Read `../../report.md`, then `../REPRODUCE.md`.

## State

- All training campaigns are finished. Results: `outputs/train/<run>/<cell>/<unit>/` (index `outputs/index.csv`).
- Storage: `outputs/`, `checkpoints/`, `build/`, `logs/` are links into `/storage/hdd/disk0/namdp15/auscult-trust-A/`. Runs sit in stage folders (`LAYOUT.md` there); `train/<run>` links keep certificate paths valid.
- Checkpoints kept: final and report-epoch units of `grid-v1`, `lodo-v1`, `lodo-ast`, `lodo-bench`, `pcmcl-v1`, `sc-dynamic-completion-20261024`, `followup-20261026-final`, `target-20261027-final`, `distill-20261028-final`, `student-final-20261029-final`. Units that no certificate, bundle or distillation chain references were pruned (`PRUNE-20261112-checkpoints.tsv`).
- `mirror/` and `archive/` were deleted on 14/11 after a hash dedupe against the store. The 12,174 files without a duplicate are listed in `REMAINING-DELETED-20261114.tsv`. Source-CV (`cv*`) checkpoints that existed only on old machines are gone.
- Final weights (AuscultTrust `kd_g4_k1` seed 0, baseline seed 0): GitHub Release `weights-final`. See `../../DELIVERY.md`.

## Open

1. Serving bundle still holds BEATs + PAFA static SC, not `kd_g4_k1`. Edit `PIPELINES` in `api/bundle.py`, run `python -m api.bundle`, validate hashes and health.
2. Phantom H1, H3 and the co-primary H2 need transducer recordings. Weights and `s_ref` are frozen (`outputs/train/final-*/*/seed*/freeze.json`).
3. Old KAUH and screening exports (`outputs/kauh/final-*`, `outputs/screening/final-*`) fail the strict provenance check since the refactor. Re-export before reuse.
4. Pages worker needs `API_UPSTREAM` set at deploy (`app/README.md`).

## Gotchas

- Never `pkill -f` or `pgrep -f` with a pattern that appears in your own command line. Kill by PID.
- `scripts/final_evaluation*.py` have no argument parser; `--help` starts a run.
