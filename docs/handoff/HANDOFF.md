# Handoff (27/10/2026 18:30): stage 12 source CV done, target evaluation RUNNING (campaign target-20261027), report v6

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


## 0. Running campaign `target-20261027` (launched 27/10 18:12 +07, independent of any Claude session)

- What: official + four-device LODO + KAUH/screening, 5 seeds, for class_aware_light (epoch 4), lora_class_aware (43), lora (83),
  swa (43), multiscale (68), plus the PCSL-5 PAFA control (`pcsl5_control`). Phase `main` = 25 control CV fits + 125 finals
  (150 jobs); phase `control_final` = 25 control finals at the epoch the controller picks from the control's own mean curve.
  The anchor is not re-run: reuse the 25 `gain6-cv66` finals (B, `build/campaigns/followup-20261026/outputs/train/followup-20261026-final/`).
  Code path: `scripts/target_campaign.py` (controller, B) on top of `scripts/improvement_campaign.py` (core) and `scripts/improvement_unit.py`.
- Where: snapshot `build/campaigns/target-20261027/` (frozen; same path on A, B, D, E, F; never edit). Manifest and job lists:
  `logs/target-20261027/programme.json`, `logs/target-20261027/<phase>/<machine>.jobs.txt` (on B, the controller).
  Outputs and checkpoints are collected on B (canonical), archived on E; A's snapshot `outputs`/`checkpoints` are symlinks to the HDD.
- Services (systemd --user, enabled, linger on): controller `auscult-target-20261027-controller` on B; workers
  `auscult-target-20261027-<phase>-<A|B|D|E|F>` (2 jobs per GPU; F limited to 30 GB). They restart on failure and survive logout and reboot.
- Status: `ssh namdp15@10.148.183.8 "python3 -c \"import json;s=json.load(open('<snapshot>/logs/target-20261027/programme.json'));print(s['phase'],s['status'])\""`;
  `journalctl --user -u auscult-target-20261027-controller -n 30` on B. Expected total 8-9 h wall (main 6-7 h, control_final 1.5 h).
- Done when `outputs/evaluation/target-20261027/complete.json` exists on B (phase `complete`). It lists the control epoch and
  scores of control / anchor / class-aware light at epochs 4, 66 and the control epoch.
- Smoke before launch (2-epoch fits on B): LoRA, SWA, multiscale and control pass fit, export, screening and KAUH; export probabilities match training
  probabilities (max diff 2e-4 LoRA, 4e-4 multiscale, 5e-3 SWA, which also differs from epoch 11 by 0.69, so SWA weights are exported).
- Not yet done: the paired analysis (recipe vs anchor `gain6-cv66` and vs control: official, LODO per device and macro, worst device, set size,
  KAUH A/B2, screening) and its entry into `report.md` section 17. A cut-off unit must be set aside and rerun from epoch 0 (do not resume `last.pth`).
- Incident 27/10 22:36: controller crash-looped (83 restarts) on "Completion receipt references another training unit": receipts from A carry the HDD mount path.
  Fixed by `canonical_path` in `scripts/improvement_campaign.py` (commit 916d5d2); snapshot file and manifest updated on all machines with a
  `recovery_amendments` entry. While pushing files I overwrote the remote copies with empty ones (`ssh -n` plus a redirect: never combine them);
  files and manifest were restored (manifest from the 20:52 copy on A, commands verified against all `jobs.txt`). After the repair the first tick showed 44/150 complete (22:47).
- 28/10 09:20: F (shared, slow: 18/38 after 15 h) was stopped; its scheduler was killed by PID with its two running jobs left alone
  (`lora+lodo_meditron/seed1`, `swa/seed0`). The 18 unstarted F jobs were reassigned to A 2, B 5, D 6, E 5 (manifest `recovery_amendments` lists
  them; job lists on all machines rewritten; backups `programme.pre-f-move.json` on B and `*.jobs.old.txt` not kept). F's two jobs are still collected from F.
  The F service is stopped on purpose; do not restart it until its two jobs finish.
- If a service crash-loops: read its journal first; a hash mismatch on a snapshot file means someone edited the snapshot.

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
