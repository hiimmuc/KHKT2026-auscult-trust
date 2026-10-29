# Documentation index

| Type | Date | Status |
| --- | --- | --- |
| reference | 2026-10-27 | current |

Start with `../report.md` (results by plan stage) and `handoff/HANDOFF.md` (state, commands, open issues).

## Layout and naming

| Folder | Content | File name |
| --- | --- | --- |
| `.` | `SETUP.md`, `QUICKSTART.md`, this index, `TEMPLATE.md` | upper-case singleton names |
| `handoff/` | `HANDOFF.md`, the only current handoff | fixed name; older handoffs are folded into `report.md` and removed |
| `decision-logs/` | protocol decisions that bind later work | `YYYY-MM-DD-topic.md` |
| `reports/` | measured results and audits | `YYYY-MM-DD-topic.md` (date = evidence cut-off) |
| `updates/` | proposals, pre-registration and amendments, plans | versioned `topic-vN[-en].md`, `prereg-v9-amendment-N.md`, or `YYYY-MM-DD-topic.md` for a dated plan |

Rules: lower-case kebab-case; ASCII; date-first ISO dates for dated documents; no `_`; one header table
(Type, Date, Status) under the title (see `TEMPLATE.md`); new results go into `../report.md` first, a detailed report is
written only when the numbers do not fit there. Interim status notes, executed plans and old handoffs are deleted once
their content is in `report.md`.

## Current documents

- Registered: `updates/prereg-v8.md`, `updates/prereg-v9-amendment-1.md` to `-5.md`, `updates/research-proposal-v9-en.md`.
- Current proposal: `updates/research-proposal-v10-en.md`. Plans: `updates/stage-plan-v1.md`, `updates/lodo-plan-v1.md`,
  `updates/2026-10-27-target-evaluation-plan.md` (executed 27-28/10).
- Decisions: `decision-logs/2026-10-23-final-model.md` (final model), `2026-10-07-*`, `2026-10-14-db-scale.md`.
- Reports: see the table in `../report.md` header. Latest: `reports/2026-10-28-target-distillation-results.md`, `reports/2026-10-29-replay-phone-electret.md`.
- Analysis notebooks: `../notebooks/README.md`.

## Removed on 2026-10-27 (content folded into `../report.md`)

Reports: 08/10 sweep status, synced campaign and three-seed progress (24/10), KD preflight and first Litt3200 seed, nine-improvement
completion note, next-addons note, follow-up model and ensemble tables. Handoffs: 08/10, 14/10, 15/10 (grid monitoring and
`monitor_log`), 19/10, 23/10, 24/10 CPU analysis. Plans: improvement programme and its implementation plan, 48-hour sprint,
execution plan 24/10, implementation status and throughput plan 25/10, all-improvements hardware and execution plans,
next-improvements and follow-up plans, warm-up preflight, v8 migration plan, Vietnamese v9 change list.
Tracked files are in git history; all 29 files (including untracked ones) are archived in
`/storage/hdd/disk0/namdp15/auscult-trust-A/docs-removed-20261027.tgz`.
