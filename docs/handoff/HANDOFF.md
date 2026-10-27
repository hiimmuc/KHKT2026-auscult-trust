# Handoff (27/10/2026 17:00): stage 12 source CV done, target evaluation planned, report v6

| Type | Date | Status |
| --- | --- | --- |
| handoff | 2026-10-27 | current |

For the next session. Read this file, then `CLAUDE.md` (repo rules), `report.md` v6 (results by plan stage; stage 12 =
section 17), `docs/updates/2026-10-27-target-evaluation-plan.md` (proposed next run) and
`docs/decision-logs/2026-10-23-final-model.md` (final decision). The proposal is v10
(`docs/updates/research-proposal-v10-en.md`); the registered study is v9 with amendments 1-5. Archived handoffs:
older handoffs were folded into `report.md` (sections 4 and 17) and removed on 27/10 (`docs/README.md`); the follow-up campaign note is
`logs/followup-20261026/HANDOFF.md`.

Session style asked by the user: terse replies, minimal code, one runnable check per change. Earlier sessions edited
through a git worktree and pushed to `origin main` with a `Claude-Session:` line; a worktree made from HEAD does not
contain the uncommitted tree (open issue 1), so edit the main checkout until it is committed.

## 1. State

- [Fact] Project definition (provisional, user 27/10): AuscultTrust = BEATs + PAFA + trained static SC + split conformal;
  baseline = the same without SC. Frozen evaluation and serving bundle unchanged since 23/10.
- [Fact] No process of ours runs on A, B, D or E except the demo API on A (port 8000, since 27/10 10:51). The stage-12
  campaigns are finished: 150 five-fold source-CV fits (6 arms x 5 folds x 5 seeds), 25 anchor finals (`gain6-cv66`),
  3-fold screens of event heads and of the five recipes + refined KD.
- [Fact] Stage-12 results are source-CV only. Class-aware light +1.458 (selected epoch 4), LoRA + class-aware +0.695, LoRA
  +0.607, SWA +0.076, multi-scale -0.321 against the epoch-66 anchor; the 3-fold screens passed none. Event heads (-1.637)
  and refined KD (+0.009) failed their source gates (gates recomputed on 27/10; the controllers had not recorded them).
- [Pending] Official, LODO and KAUH for the five recipes, and a PCSL-5 non-class-aware control. Plan, cost
  (about 54 slot-hours for tiers 0 + 1) and decisions: `docs/updates/2026-10-27-target-evaluation-plan.md`.

## 2. Where things are

- Repository on A: `/home/namdp15/Workspace/Personal/KHKT-CVA-2026/auscult-trust`. Same path on B `namdp15@10.148.183.8`, D
  `10.148.183.171`, E `10.148.182.179`, F `10.148.182.137` (shared with another user, not for new runs). Machine C is dropped.
- Outputs of historical runs: `outputs/train/<run>/<cell>/seed<s>/` on A (counts in `report.md` section 4.3). Campaign
  snapshots, outputs and checkpoints: `build/campaigns/<programme>/` (B canonical, E archive). The 25 anchor finals exist only on B
  (`build/campaigns/followup-20261026/outputs/train/followup-20261026-final/gain6-cv66*`).
- Mirrors with inventory and hashes: `/storage/hdd/disk0/namdp15/auscult-remote-mirror/manifest.json` (B 143.9 GB, D 40.8 GB, E
  60.3 GB, F 1.8 GB). `outputs/`, `checkpoints/` and the campaign `outputs`/`checkpoints` directories on A are symlinks to `/storage/hdd/disk0/namdp15/auscult-trust-A/` (moved 27/10; SSD 255 GB free).
- Certified five-fold summary: `build/campaigns/fivefold-20261026/outputs/evaluation/fivefold-20261026/summary.json`.
- Docs: `docs/reports/2026-10-27-improvement-results.md` (tables, Vietnamese), `docs/reports/2026-10-27-findings.md`
  (scope and RQs), `docs/reports/2026-10-24-*` audits.

## 3. Open issues found on 27/10

1. Working tree is uncommitted: 51 modified, 42 deleted (moved by the refactor), 75 new files. Commit in groups
   (refactor of `src/training` and `tests/`, campaign scripts and configs, app, documents) after the user agrees.
2. [Done 27/10] All `auscult-*` campaign services on A, B, D, E stopped and disabled (E's `next-improvements-...-screen-E` crash loop
   came from a frozen snapshot script that no longer matches its digest). Only `auscult-inference-api` (A) runs.
3. A Codex session was active on 27/10 (it edits docs and `AGENTS.md`). Check `git status` and file times before editing
   shared files; do not edit a frozen snapshot under `build/campaigns/`.
4. 50 audio-audit candidates still need human listening (`outputs/evaluation/source-audio-audit-20261026/review.csv`).
5. Phantom (H1, H3, co-primary H2) waits for hardware; weights and s_ref are frozen (`freeze.json`).

## 4. Commands

- Tests: `.venv/bin/python -m pytest -q tests` (317 pass) and `.venv-train/bin/python -m pytest -q tests api/test_contract.py`
  (334 pass); `cd app/web && npm run build`.
- Status of old grid / LODO lists: `bash scripts/grid_status.sh`. Tables: `scripts/lodo_tables.py`, `scripts/ensemble.py`,
  `scripts/ensemble_sesp.py`, `scripts/ensemble_e1_dynamic.py` (prefix `PYTHONPATH=. .venv-train/bin/python`).
- Frozen evaluation: `scripts/final_evaluation.py`, `scripts/final_ensembles.py` (`docs/QUICKSTART.md` section 7).
- Campaign tooling: `scripts/improvement_campaign.py` (core: jobs, gates, receipts, collection), `scripts/fivefold_campaign.py`
  (example of a programme), `scripts/improvement_unit.py` (fit + exports), `scripts/followup_campaign.py`.
- Recompute a source gate by hand: `PYTHONPATH=. .venv/bin/python` with `scripts.improvement_campaign.gate(candidate_reports, anchor_reports)`.

## 5. Gotchas

- Kill processes by PID only, never `pkill -f` (over ssh the pattern matches the ssh shell).
- A cut-off unit always reruns from epoch 0; never resume from `last.pth` for reported units.
- `--epochs 1` smoke runs compress the cosine schedule: their scores mean nothing.
- From a git worktree, set `AT_DATA` and `AT_REPOS` to the wrapper's `data/` and `repos/`, and link `.venv`, `.venv-train`, `outputs`.
  A worktree made from HEAD does not contain the uncommitted tree (open issue 1).
- Tool output compressed by hooks shows `...` and `<<ccr:...>>`; recover the real text before editing against it.
