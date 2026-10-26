# BEATs PAFA pooling audit

[Fact] The current BEATs + PAFA implementation is a variant of the released BEATs PAFA training path. The loss implementation and salient scalar recipe settings match, but its alignment representation differs. This qualifies the settings-only comparison in the [benchmark audit](benchmark-protocol-audit-2026-10-24.md). It does not invalidate the locally measured variant's results or establish why its Score differs from the paper.

## Traced behavior

| Path | Classification branch | PAFA alignment branch |
| --- | --- | --- |
| Released `repos/PAFA/main.py`, BEATs + PAFA | Linear classifier on each transformer token, then mean logits | Learned token attention, weighted token sum, then projection MLP |
| Current `src/training/encoders/beats.py` and `methods/pafa.py` | Mean transformer tokens, then linear classifier | Same mean-pooled feature, then projection MLP with `attention=False` |

The reference wrapper `repos/PAFA/models/beats.py` returns all transformer tokens. Its `main.py` constructs `ProjectionHead(..., attention=True, norm_type='ln')` for BEATs. The local encoder returns `h.mean(1)`, and the local PAFA projector explicitly disables attention. AST's released PAFA path already disables this attention; do not generalize the BEATs finding to every encoder.

[Fact] Mean pooling commutes with a shared linear classifier, including its bias: `mean(W h_t + b) = W mean(h_t) + b`. Thus the classification branch is algebraically equivalent at fixed weights, apart from floating-point operation order. Learned attention in the alignment branch is different: its token weights depend on features and have trainable parameters. This can change both alignment features and gradients flowing into the encoder. The attention head is used for training alignment; it is not an extra attention head required for inference.

The current BEATs encoder and PAFA-method source hashes exactly match those in the completed official seed-0 KD student's input manifest. The new finding therefore applies to the frozen pilot's implementation, not just a proposed future edit. Historical PAFA arms were already described as local recipes; their complete numerical training trajectories have not been reconstructed here. The inspected final inference checkpoint does not retain a projector, so it alone cannot prove the historical projector's training behavior.

## Executed proof

`outputs/evaluation/report-20261024/pafa-pooling/check.py` runs the unchanged reference projector/loss on a deterministic synthetic CPU tensor `(4,5,8)` with two patients. It checks:

- Mean-logit and mean-feature classification agree to a maximum absolute error of 5.56e-17 in float64.
- With attention weights fixed to uniform, projected features and encoder-input gradients agree with the local mean-pooling path; maximum gradient error is 6.22e-15.
- The attention-weight gradient is nonzero even at the uniform-attention point, so the released head can learn away from uniform pooling.
- With deterministic nonuniform attention, projected features and encoder-input gradients differ. These are checks of mathematical behavior, not quality estimates.
- Input source hashes remain stable during the check and the two local training source hashes match the pilot manifest.

The assertion-based check passed. `summary.json` records numeric errors, synthetic losses, source hashes and the explicit synthetic-only limitation; `check.log` records completion. No dataset labels, trained-model predictions or held-out scores are used in this proof.

The exact pre-candidate encoder and PAFA-method source are archived under `pafa-pooling/source-before-candidate/src/`, with hashes verified against that original summary. The current source has subsequently gained the separate candidate described below; the original snapshot describes the earlier source state.

## Next source-only experiment

[Decision] Retain the selected serving model and finish the frozen dynamic-SC/KD queues. Never change the active pilot's PAFA implementation or treat this discovery as permission to restart its jobs. Its completed results remain evidence for the declared local variant.

Before pursuing further recipe tuning, prioritize a bounded pooling comparison: the current mean-pooled PAFA control versus the released token-attention alignment path, with the same three source-training-patient folds and seed 0. Keep the existing five-second, 100-epoch, no-SC/no-warm-up recipe and all loss weights fixed. Use an explicit new method/cell identity for token attention; preserve historical `ce_pafa` behavior and inference/checkpoint compatibility. Implement and test token/gradient behavior and actual source batches before launching. Match common backbone/classifier initialization and document projector initialization. No loss, learning-rate or attention-size search is included.

Select the earliest maximum of the three-fold mean validation Score, retain the control on ties, and fix the recipe/epoch before any subsequent official or LODO evaluation. Three folds at one screening seed do not establish a trained-arm improvement. Paired multi-seed frozen evaluation and SC/conformal assessment remain necessary.

The already prepared [warm-up comparison](../updates/pafa-warmup-cv-preflight-2026-10-24.md) stays unlaunched. The pooling comparison has a more direct reproduction rationale and takes precedence as the next six-unit source-CV budget after the active queues. Do not combine attention and warm-up into one intervention or silently reuse a completed control if its source, initialization, split or schedule differs. This is a revised proposed successor, not a new resource assignment or launch manifest.

[Hypothesis] Restoring the released alignment representation may improve classification. It may also worsen or leave it unchanged. No improvement, SOTA result or device-robustness claim follows from this audit.

## Candidate implementation and executed preflight

[Fact] `ce_pafa_attn` is now implemented as a separate training method, selected with `experiments/train/pafa_attn.yaml` after the existing BEATs PAFA configs. The encoder can return its transformer tokens explicitly; ordinary calls still return the same mean-pooled features, with unchanged checkpoint parameter keys. The candidate uses the unchanged reference attention projector and PAFA loss. Classification follows the reference's mean of per-token linear logits. Existing `ce_pafa`, PC-MCL/PAFA and frozen snapshots retain their behavior. The new candidate is restricted to BEATs and this comparison uses hard labels only.

Three new focused tests failed before implementation. They now verify token/default-output and checkpoint compatibility, released-path loss and encoder-gradient agreement, nonzero attention gradients, and rejection of unsupported encoders. The method also passes the existing generic loss/probability contract. Validation used an isolated worktree, then the narrow patch was applied to the main source. The main research suite passes: **123 passed**, one existing NVML warning (`logs/report-20261024/pafa-attention-main-validation.log`). The isolated suite had one environment-dependent skip; the main suite had none.

[Fact] The real source-data preflight passed all six loader conditions with matching training/validation patients across arms. The three folds contain 2,916/1,226, 2,776/1,366 and 2,592/1,550 train/validation cycles; validation covers all 4,142 source cycles exactly once in each arm. Actual finite training batches have shape `(32,1,498,128)`. The builder constructs the official-test loader, but this check never iterates it or computes test predictions.

At seed 0, full encoder and classifier initial-state digests match exactly between arms. Projector initial states differ naturally because the reference constructs additional attention parameters before its MLP; this is disclosed rather than claimed as an identical projector initialization. A real four-cycle source minibatch passed token/default-forward agreement, finite loss, backward gradients through BEATs patch embeddings, classifier and attention weights, and an Adam update changing attention weights. This was a CPU FP32 mechanical step with three CPU threads, not a quality estimate or GPU mixed-precision/full-batch resource test. The declared training schedule remains 100 epochs, batch 32, no warm-up and no SC.

Prepared list: `logs/campaigns/pafa-attention-source-cv-20261024.txt`; six unique `(run, cell, cv_fold)` identities, seed 0. A new source-freeze preflight fingerprinted 171 source/config files. Artifacts are `outputs/evaluation/report-20261024/pafa-attention-preflight/{check.py,check.log,manifest.json,source-freeze.json}`. Preserve this snapshot; the script refuses an existing manifest/job list. The preview has no host assignment or deployment configuration. **No CV job was launched** and no selected model or active queue was modified. Observe capacity and stage a new deployment snapshot after the active campaigns before execution.
