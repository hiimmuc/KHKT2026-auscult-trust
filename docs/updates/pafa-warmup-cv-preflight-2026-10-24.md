# PAFA warm-up CV preflight, 24 October 2026

[Fact] A bounded source-only comparison is prepared and mechanically validated. No CV model has been trained or launched. The selected model, current 100-unit dynamic-SC campaign and six-unit KD pilot are unchanged.

[Later decision] The [pooling audit](../reports/pafa-pooling-audit-2026-10-24.md) identified a material architectural difference from released BEATs PAFA. Its isolated source-CV comparison now takes precedence after the active queues. These warm-up jobs remain prepared and unlaunched; do not combine the two interventions or deploy this historical preview as a current launch manifest.

## Conditions and selection

Control: current BEATs + PAFA recipe, no SC, no warm-up. Candidate: identical recipe with `experiments/train/warm5.yaml` enabling five-epoch warm-up. Both use 100 scheduled epochs, batch 32, five-second input, no SpecAugment, Adam 5e-5, the existing cosine schedule and moving-average update. This is an exploratory optimization comparison, not an exact reproduction of another paper or a new registered hypothesis.

Use seed 0 and the same three device-stratified training-patient CV folds for both conditions: six units total. Select the epoch with maximum mean validation Score, earliest on ties. Prefer the warm-up candidate only if its best mean CV Score exceeds the control; retain the control on a tie. Do not use official-test or held-out-device scores for this choice. One screening seed does not establish a trained-arm result; any later comparison requires fixed-recipe paired seed evaluation.

## Executed preflight

| Fold | Train patients | Validation patients | Train cycles | Validation cycles | Validation class counts N/C/W/B |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 | 52 | 27 | 2916 | 1226 | [518, 396, 153, 159] |
| 1 | 53 | 26 | 2776 | 1366 | [866, 330, 143, 27] |
| 2 | 53 | 26 | 2592 | 1550 | [679, 489, 205, 177] |

[Fact] Both conditions have identical patient assignments. Every training patient belongs to exactly one validation fold, with no training/validation overlap within a fold. Source cache labels/patients/devices match annotation order; cached images and pretrained weights are fingerprinted. Actual training and validation batches were loaded on CPU: training shape 32 x 1 x 498 x 128, finite values and valid labels, no teacher targets. The existing builder constructs the official-test loader, but this preflight never iterates it or evaluates test predictions. It uses zero loader workers for mechanical iteration; prepared jobs retain two workers.

The complete 100-epoch learning-rate schedule was checked at epoch 1, the last warm-up batch in epoch 5, epoch 6 and epoch 100; this was not a compressed one-epoch smoke. Warm-up starts at 5e-6, joins the existing cosine schedule after epoch 5 and finishes at 5e-8 in epoch 100. No classification-quality estimate is inferred from these checks.

## Campaign identity correction and proof

`scripts/campaign.py:identities` previously identified every same-cell/same-seed CV fold as `seed0`, rejecting valid folds as duplicates and checking the wrong output directories. It now identifies CV units as `cv0`, `cv1`, `cv2`, matching the training CLI. Non-CV seed identities are retained. Regression cases cover same-seed distinct folds, source freezing of all six jobs, default CV fold zero and disabled CV. Both new tests failed before the fix; the full research suite now passes: 119 passed, one NVML warning (`logs/report-20261024/cv-identity-validation.log`). Existing frozen snapshots were not patched.

Prepared list: `logs/campaigns/pafa-warmup-source-cv-20261024.txt`. Run ID: `pafa-warmup-source-cv-20261024`. Explicit `--cv_fold` values are present on all six commands. A source-freeze dry run admitted six unique units and fingerprinted 168 source/config files. The preview intentionally has no host assignment or deployment configuration and is not a launch manifest.

```bash
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 PYTHONPATH=. .venv-train/bin/python \
  outputs/evaluation/report-20261024/warmup-preflight/check.py
.venv-train/bin/python -m pytest -q tests
```

Artifacts: `outputs/evaluation/report-20261024/warmup-preflight/{manifest.json,source-freeze.json,check.py,check.log}`. Preserve the frozen proof with the report archive. The current GPU queues take priority. Assign machines, stage a new isolated source snapshot and freeze an actual deployment manifest only after the existing runs and resource state are observed. Do not silently append these jobs to the active queues.
