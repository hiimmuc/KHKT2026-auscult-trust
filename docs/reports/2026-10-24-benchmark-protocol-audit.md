# Benchmark protocol and next experiment audit

| Type | Date | Status |
| --- | --- | --- |
| report | 2026-10-24 | current |

24 October 2026. Exploratory audit, not a model selection result. Read with the
[ensemble and finite-N evidence](2026-10-24-ensemble-finite-n.md) and
execution plan (removed 27/10, see `docs/README.md`).

## Published targets and comparability

The table records directly inspected primary sources. It is not an exhaustive leaderboard or certification of a current world-best result.

| Reference | Published four-class Score | What must be matched before comparison |
| --- | ---: | --- |
| [PAFA, Interspeech 2025](https://www.isca-archive.org/interspeech_2025/jeong25_interspeech.pdf) | 64.84 +/- 0.60 | Five runs; BEATs. Released code evaluates the official test loader each epoch and saves its best Score. Compare with a labelled test-selected column, not our fixed-epoch primary column. |
| [PC-MCL, January 2026](https://arxiv.org/html/2601.17080v1) | 65.37 +/- 0.73 | Official 60/40; five runs; three-label outputs with event-priority conversion, 10-second concatenated training inputs. Threshold conversion and inference on single cycles matter. |
| [ADD, Interspeech 2025](https://arxiv.org/html/2506.02505v1) | 65.53 | AST with its denoising components. The paper describes 8-second input and 64 mel filters; our grid's shared preprocessing is not automatically an exact reproduction. |
| [Meta-Ensemble, April 2026](https://arxiv.org/html/2604.24096v1) | 66.49 +/- 0.05 | CLAP/BTS, five base models, supervised fusion on reserved training data. Highest row uses a sample-level base/meta split; this permits patient overlap between those training subsets. It is not evidence of test-set leakage by itself. Patient-level rows differ. |
| [PAC-KD, October 2026](https://arxiv.org/html/2610.22826v1) | 65.28 +/- 0.29 | PAFA student, two separately trained binary event specialists, attribute-marginal distillation, five-epoch warm-up and matched controls. This differs from the running three-teacher soft-class-target pilot. |

[Fact] Meta-Ensemble's own patient-level table reports mean fusion 65.12 for the fixed split and 64.33 for its five-fold strategy. Its tables/discussion are not fully consistent about some fusion numbers. Do not treat its highest sample-level result as a patient-disjoint fusion target. The 72.7 number encountered on a secondary dataset page was not verified in these primary sources and is not admitted as the target.

[Interpretation] No inspected method establishes its official-split number as an expected LODO Score. Preserve separate official benchmark, unseen-patient and device-held-out evidence. Do not copy a paper's SOTA wording without a broader, protocol-matched comparison.

## Local PAFA recipe and split audit

[Fact] `experiments/train/grid/pafa_beats.yaml` matches the released PAFA script on the salient recipe settings: 100 epochs, batch 32, five-second cycles, no SpecAugment, PCSL weight 50 and GPAL weight 0.0005. `base.yaml` supplies Adam at 5e-5, weight decay 1e-6, cosine scheduling and moving-average beta 0.5. This is a settings check, not proof of identical floating-point training trajectories.

Local upstream evidence: `../repos/PAFA/scripts/beats_pafa.sh`; `../repos/PAFA/main.py:set_loader` builds the official-test dataset as `val_loader`, and `validate` saves the highest test Score. The primary local result uses its stored fixed epoch. The released seeds are 1-5; local campaign seeds are 0-4. Keep that distinction visible.

[Fact, later architectural audit] The scalar settings above do not make the local BEATs PAFA path an exact reproduction. The local alignment branch receives mean-pooled tokens, while the released BEATs PAFA projector learns token attention before pooling. A [source/gradient audit](2026-10-24-pafa-pooling-audit.md) verified the distinction and its presence in the frozen KD source. The linear classification branches are algebraically equivalent at fixed weights; the alignment gradients can differ. The next proposed source-CV budget now prioritizes this isolated pooling comparison before warm-up tuning. No active campaign or selected model was changed.

[Fact] Annotation/split reconstruction found 4,142 training cycles from 79 patients and 2,756 test cycles from 49 patients. Patients 156 and 218 occur on both sides: 117 training cycles and 120 test cycles. Excluding their test cycles leaves 2,636 cycles from 47 patients absent from the training set. LODO already removes these shared patients before splitting. Raw annotations and committed splits were not changed.

[Decision for reporting] Preserve the full official row for benchmark comparability, and add the unseen-patient evaluation row for each official model/ensemble/student. The latter uses the same trained model with an explicit evaluation mask; do not call it a newly untouched validation set. The KD official arm must not be described as wholly patient-disjoint if its full official-test row includes the two shared patients. For conformal evaluation, follow the existing exclusion before calibration/evaluation partitioning.

The current reference outputs illustrate the distinction: PAFA/static Score is 62.27 on the full official split and 62.03 on the unseen-patient subset. The static three-BEATs/two-HTS-AT mean is 63.19 and 63.21, respectively. Both are descriptive five-seed means, without target-side k adjustment. Full tables and artifact hashes are in the linked evidence report.

## Warm-up control bug and bounded successor

[Fact] The parser accepted `--warm_epochs 5` then unconditionally assigned 10 before computing the schedule. A schedule-level regression test reproduced the wrong learning rate and wrong exit from warm-up. The local fix respects a positive explicit duration; zero/unspecified retains the historical 10-epoch default. Negative durations with warm-up enabled are rejected. Tests cover the downstream learning rate during and after the requested duration, not just the parsed value.

Affected code: `src/training/config.py:parse_args`; consumer: `src/training/optim.py:warmup_learning_rate`; proof: `tests/training/test_warmup.py`. No frozen snapshot, historical artifact or remote job was modified. This is an experiment-control correction, not measured evidence of a Score improvement.

[Proposed next experiment, not launched] After the existing SC completion and KD pilot, compare the current PAFA recipe against the same recipe with five-epoch warm-up on the existing three source-training-patient CV folds. Keep every other setting fixed. Use validation curves only, earliest maximum of their mean for epoch selection, and retain the no-warm-up control. Do not add a temperature/loss/learning-rate sweep at the same time. Warm-up has a directly relevant published precedent and uses an existing optimizer path, but may improve, worsen or leave results unchanged.

A fresh isolated, hash-frozen campaign is required for this successor; the existing manifests bind the old source. Its launch and any refit are deferred until current jobs are observed and their resource use is known. Do not choose from held-out device labels or overwrite the selected serving bundle. A subsequent frozen candidate receives official and LODO evaluation, then SC/conformal assessment; claims remain exploratory after the prior benchmark inspections.

Validation: the schedule tests failed before the fix (3 failures, 1 default-behavior pass), then all four passed.
The full research suite passed with `.venv-train/bin/python -m pytest -q tests`: 117 passed, one NVML warning
from unavailable GPU driver visibility. Log: `logs/report-20261024/warmup-validation.log`. The native worktree
tool could not resolve the unversioned wrapper; the fix was tested in a manual detached worktree at
`/tmp/auscult-warmup-20261024`, then applied to the current source without replacing other work.

## Outstanding evidence

Remote process observation still fails before connection in this session (`socket: Operation not permitted`), and local NVIDIA driver communication is unavailable. The 10:00 remote status is historical, not a verified wait or permission to duplicate jobs. The two CPU analyses above completed against local artifacts.

Patient/seed finite-N intervals and an existing-arm SC comparison are now available in the
[uncertainty update](2026-10-24-sc-finite-n-uncertainty.md). Remaining work includes joint finite-N SC,
completion/verification of 100 dynamic-SC units and six KD units, matched student/teacher runtime measurements,
source-patient-CV optimization, broader literature checking, and the hardware experiments. Model superiority,
SOTA and device-robust conformal coverage are not established by this audit.
