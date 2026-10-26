# ICBHI, LODO and KAUH improvement programme

24 October 2026. Research plan and resource estimate requested by the user. No jobs are launched by this document. All new work is exploratory; registered v9 criteria and historical results remain intact.

**Superseded for execution:** the user subsequently limited availability to four RTX 5090s for two days and one RTX 6000 PRO for one day. Use the [48-hour sprint](improvement-sprint-48h-2026-10-24.md). This larger programme is deferred.

## Scope and recommended commitment

Improve four-class ICBHI classification, generalization to held-out devices, and recording-level KAUH performance. Cover every direction proposed in the discussion through bounded experiments, then perform five-seed evaluation of the strongest candidates. More compute supports replication and broader coverage; it does not guarantee a higher Score or device robustness.

The full commitment covers **20 families / 40 conditions**, including **35 source-screened candidate conditions**, a common control, source-only fusion, two separately reported target-adaptation regimes and two conformal comparators. The first delivery point budgets **1,147 main model-training slots** and evaluates four finalists in depth. **Completion of all proposed work continues to full evaluation of all source candidates**, bringing the main budget to **1,797–2,697 slots**, plus teachers, SSL and adaptation. Some processing-only conditions can share trained weights; slots and costs are conservative ceilings before verified deduplication.

Given the user's willingness to provide hardware, recommended provision for the full scope is **sixteen independent training workers**, each with a fast 32–48+ GB CUDA GPU, plus temporary access to **one four-GPU 80 GB server** for SSL. The SSL server can be reused for independent trials afterward; twenty GPUs need not remain occupied throughout. The existing 32 GB machines remain useful. Plan **4–8 calendar weeks**, conditional on ready data, usable execution access and annotation review. Early classification comparisons should become available in **3–7 days**. The eight-worker first-delivery option is also priced, but it is not the stopping point for “complete all.”

The 25 October report deadline cannot accommodate this full programme. Submit verified existing evidence at that deadline, retain incomplete studies as pending, and run this as the subsequent improvement programme toward the 31 January research cut-off.

Implementation work packages: [implementation plan](improvement-programme-implementation-2026-10-24.md). Earlier evidence: [protocol audit](../reports/benchmark-protocol-audit-2026-10-24.md), [PAFA pooling audit](../reports/pafa-pooling-audit-2026-10-24.md), [frozen KAUH evaluation](../reports/final-pafa-static-execution.md).

## Baseline and outcomes

Keep current BEATs + PAFA as the classifier control and BEATs + PAFA + trained static SC as the selected AuscultTrust reference. Include fresh paired controls when source, initialization, split or schedule changes. The attention implementation is ready and passed 123 research tests plus a real source CPU gradient step; its improvement is untested.

| Evaluation | Primary outcome for this exploratory programme | Required companion outcomes |
| --- | --- | --- |
| ICBHI official | Fixed source-validation-selected epoch, four-class Score | Sp, Se, HS, class recalls, per-device results, five seeds; full official and separately labelled unseen-patient subset |
| LODO | Four-device macro Score, with a separate worst-device view | Every fold; macro excluding AKGC417L; Sp/Se, class support, coverage, set size; source-only selection |
| KAUH external | Recording-level binary balanced accuracy, macro over filters | Sensitivity, specificity, AUROC, filter-specific results, paired flip/TV, coverage, singleton/uncertain rates, set size |
| Practicality | Fresh-process latency, loaded memory and allocator peaks | Batch 1/32, fixed hardware/precision/workload, model count, checkpoint size and complete timing scope |

Proposed engineering goals are +1 Score point on official ICBHI, +1 macro Score point on LODO, and +2 balanced-accuracy points on KAUH, with no concealed loss on a device or sensitivity. These are planning targets, not predicted gains, powered effect sizes, equivalence margins or replacements for registered gates. Report paired patient/seed intervals, including negative results; do not declare improvement from a seed mean alone. Keep distinct candidates when one improves classification and another improves transfer; model selection uses validation only.

## Candidate register

Every candidate has a declared parent control. P01–P17 use source data; P18 consumes source out-of-fold predictions; P19 changes the adaptation regime and is reported separately; P20 changes uncertainty sets rather than raw classifier accuracy. Do not combine factors during their first tests.

| ID | Family and bounded conditions | Count | Main purpose |
| --- | --- | ---: | --- |
| P01 | Released PAFA token-attention alignment | 1 | Resolve the known reproduction difference |
| P02 | Patient alignment within patient/class groups; class-conditional alignment across patients | 2 | Preserve normal/abnormal differences within a patient |
| P03 | Classification attention pooling; top-k token pooling | 2 | Retain brief abnormal events |
| P04 | Shared-backbone 1+5 s views; 5+8 s views | 2 | Combine transient detail and context |
| P05 | Layerwise LR decay; gradual unfreezing; pretrained-weight anchoring | 3 | Reduce overfitting and representation drift |
| P06 | Five-epoch warm-up; LR 2.5e-5; LR 1e-4, each versus the original 5e-5/no-warm control | 3 | Improve optimization and training duration |
| P07 | Auxiliary crackle/wheeze heads with the four-class head | 1 | Preserve event structure, including both events |
| P08 | Patient-balanced sampling; bounded class weighting | 2 | Reduce patient domination and rare-class neglect |
| P09 | Smooth EQ; bandwidth simulation; moderate noise; their fixed mixture | 4 | Teach plausible acquisition variation |
| P10 | Same-audio prediction consistency on P09's mixture | 1 | Isolate consistency beyond augmentation alone |
| P11 | Source device/class group-robust objective | 1 | Improve difficult source groups without ignoring support |
| P12 | Static-SC control; half log-gain correction; static gain clipped to ±12 dB | 3 | Test regularization of correction; matched training/inference |
| P13 | Joint cycle/recording training with attention pooling; with top-k pooling | 2 | Train the recording-level screening task |
| P14 | Matched five-second screening windows; harmonized resampling/fading at the existing window duration | 2 | Test processing/length mismatch; reuse weights where valid |
| P15 | Independently reviewed source-training annotations, as a versioned overlay | 1 | Test label/boundary quality without changing raw data |
| P16 | BEATs lung-audio SSL continuation: checkpoints at 10k and 50k updates | 2 | Adapt representation using additional training audio |
| P17 | Hard+soft class KD; event-specialist KD; patient-crossfit teacher KD | 3 | Test targets beyond the frozen pilot |
| P18 | A nonnegative, regularized source out-of-fold fusion model, with equal weights as control | 1 | Use complementary errors without target-label fitting |
| P19 | Unlabelled target adaptation; few-labelled-patient adaptation | 2 regimes | Quantify adaptation under a different, explicit protocol |
| P20 | Source class-conditional LAC; source-device-robust LAC | 2 | Compare coverage and efficiency after classifier freeze |

P01–P17 total 35 conditions. P05/P08/P09/P10/P11 constants and group-support rules must be locked in a per-family spec before its first run. Their implementation package may use source-only mechanical/preliminary pilots to choose a feasible setting, but no undeclared sweep is included in this budget. The programme does not cover every combination of every factor.

## Data and selection rules

1. Preserve raw audio, original annotations and committed splits. Labels reviewed for P15 are an overlay applied only to the current fit's eligible training patients. Validation/test labels stay original; suspected evaluation-label problems are reported separately.
2. Official selection uses the existing three patient-disjoint official-training folds. Do not use either official-test patients or conformal calibration patients for selection. Report the two shared official patients explicitly and retain the unseen-patient evaluation view.
3. For LODO, reconstruct each outer fold first, excluding its held-out-device patients and reserving its calibration patients. Candidate selection then holds out each of the **three remaining source devices**, using only the outer training patients. A patient with the inner held-out device cannot enter inner training. No outer test or calibration labels enter this procedure. This requires new loader/split support; it is not already provided by the prepared six-job preview.
4. All 36 source recipes, including control, receive this inner-device screen. Do not shortlist LODO recipes using official validation patients that belong to an outer held-out cohort. Source-defined finalists and source-selected per-fold procedures are named separately. No SC mode, threshold, member or epoch is chosen using the held-out device's results.
5. Select the earliest maximum of mean validation Score over a predeclared checkpoint grid: epochs 1–10, then 15,20,...,100. Recording-specific heads use recording balanced accuracy as their declared secondary selection objective, with cycle results reported too. The new checkpoint grid is a disclosed exploratory rule; it does not change existing registered or prepared-run rules.
6. Freeze candidate definitions before screening, then advance at most eight candidates plus control for official stability, and three plus control independently inside each outer LODO fold. Lock four combination templates before their fits, selecting their constituent settings independently from each endpoint's eligible source validation. Before final evaluation, freeze four official finalists plus control and four source-ranked candidates plus control **independently in each outer fold**. Never aggregate validation from other outer folds to choose the current outer fold's method: those other source pools can contain its test patients. The four LODO positions are source-selection procedures and can choose different families by fold; do not present them as one fixed architecture. The 250-fit budget is unchanged. A common fixed-recipe Stage-1-to-LODO evaluation is an additional, separately labelled option if required.
7. KAUH remains a frozen external evaluation for the primary programme. Windows, three software-filter renderings and repeated predictions from one patient remain one patient cluster for splitting and uncertainty. The primary registered eight-second/mean-aggregation results remain intact. New processing/MIL rules are selected on ICBHI training patients, not KAUH outcomes.
8. P19 is a separate adaptation study. Use five fixed outer patient folds, distinct adaptation/calibration/evaluation patients, and keep all filter renderings of each patient together. Unlabelled budgets: 5,20,all eligible adaptation recordings; labelled budgets: 2,5,10 adaptation patients. Three model seeds give 2 regimes × 3 budgets × 5 folds × 3 seeds = 90 fits. Record infeasible budgets explicitly. These results are not frozen zero-shot KAUH results; because the dataset has already been inspected, new-cohort validation remains desirable.
9. P16 SSL uses a separately audited additional lung-audio training corpus, excluding ICBHI/KAUH evaluation recordings and duplicates. Do not perform corpus-wide pretraining on a CV validation cohort. Use HF_Lung_V1 only after confirming provenance, usable labels/metadata, recording/patient overlap and allowed use for this new exploratory extension. No hospital Tier-2 data is included.
10. Crossfit teachers exclude the predicted patient's entire data. For every student CV/inner-device fit, teacher training also excludes the student's validation, outer calibration and test patients. Full-source historical teachers cannot be reused inside a CV fold if they saw that fold's validation patients.
11. Compute 1,000 paired patient/seed bootstrap draws; recalibrate in each draw when estimating conformal contrasts. Repeated windows, bootstrap draws and filter renderings are not extra independent patients. Existing registered null gates and physical-test criteria are unchanged. Exploratory many-candidate comparisons do not become independent confirmatory claims.
12. Fit SC reference spectra and any learned normalization on the current fit's training patients only, after partitioning. Validation/target coefficients may use their declared unlabelled pools, labelled explicitly as transductive; this does not permit including those patients in the training reference. Fix finite-N adaptation/evaluation patient partitions before estimating either SC or k. Existing whole-training-dataset preprocessing cannot be assumed to implement this new nested protocol without verification.

## Phases and experiment counts

Counts are budgeted slots before valid reuse. A main fit is at most 100 epochs; heavier methods may cost more per epoch. Existing frozen queues finish or are explicitly rescheduled by their owner; this plan does not mutate them.

| Phase | Work and gate | Main/auxiliary fit slots | Worker GPU-hours before reserve |
| --- | --- | ---: | ---: |
| 0 | Accept current SC/KD outputs, collect needed checkpoints/targets, complete matched pilot runtime | Existing remainder | 40–80 |
| 1 | Split/OOF/checkpoint safeguards, corpus audit, representative real GPU smokes | Mechanical only | 20–40 |
| 2 | All 35 source candidates + control, 3 folds, seeds 0–1 | 216 | 150–350 |
| 3 | The same 36 recipes, 4 outer devices × 3 source-only inner devices, seed 0 | 432 | 250–650 |
| 4 | Official top 8 + control: 3 folds × seeds 2–4; LODO top 3 + control per outer fold: 3 inner devices × seeds 1–2 | 81 + 96 = 177 | 120–300 |
| 5 | Four predeclared combinations: official 3 folds × 2 seeds, plus outer 4 × inner 3 × 1 seed | 24 + 48 = 72 | 70–180 |
| 6 | Four finalists + control per endpoint/outer fold, no SC/static SC, official + 4 LODO folds, seeds 0–4 | 5 × 2 × 5 × 5 = 250 | 250–650 |
| 7 | Fold-correct specialist/crossfit teachers and target exports; executed alongside phases 2–6 | At least 198 teachers, conditional reserve for up to 450 more | 150–650 |
| 8 | Frozen KAUH, fusion, conformal, joint finite-N SC+k, runtime and bootstrap | Reuse trained checkpoints | 80–200 |
| 9 | Separate target adaptation, original external results retained | 90 | 30–90 |

The five main model phases total **216+432+177+72+250 = 1,147 slots**. P14 may share the trained parent; teacher/model reuse is permitted only with exact patient/config/source identity. Phase 7's 198 initial teachers comprise 54 official patient-crossfit teachers, 108 nested-LODO crossfit teachers, and 36 event specialists. If KD advances, the conditional teacher reserve must be refined from the actual selected graph before deployment; extra teachers beyond the reserve require a budget update, not hidden free compute.

P16 SSL is separate: two pretraining seeds, 50k updates each, retaining 10k/50k checkpoints. Start with 5 s clips, global batch 128 over four GPUs. At an **assumed, unmeasured 1–3 seconds/update**, this is 111–333 native 80 GB GPU-hours; budget 100–400, or **130–520 with reserve**. One pretraining seed supplies the initial two source conditions; the second checks pretraining-seed sensitivity if SSL advances. Five fine-tuning seeds alone do not measure pretraining-seed variation.

## Timing evidence and uncertainty

Measured metadata in `logs/grid_smoke_B.json` gives BEATs PAFA 7.06 GB allocator peak, 0.130 s/step and an extrapolated 0.602 h for 100 epochs on the RTX 5090. Machine A's smoke gives 7.06 GB, 0.202 s/step and 0.885 h. These are **smoke extrapolations**, not end-to-end reserved-GPU service guarantees. Recent A dynamic-AST logs also show material differences by fold and concurrency; do not transfer their epoch time to a new BEATs architecture.

For planning, allow about 0.5–1.0 worker GPU-hour for a standard source-CV fit, 0.75–1.5 for a full official fit, and roughly 1.5–3 times the relevant cost for multiview/consistency/specialist variants. Inner-device fits are smaller but vary by class/device support. Each phase range reflects a mixture of these costs, caching and teacher reuse; it is not a measured per-method forecast.

Worker ranges total **1,160–3,190 RTX-5090-equivalent GPU-hours**. Adding a 30% contingency gives **1,508–4,147**, rounded to **1,500–4,200 worker GPU-hours**. SSL native 80 GB GPU-hours are separate: do not add unlike-device hours and claim measured equivalent throughput. Benchmark the proposed GPUs and replace these estimates before bulk execution.

Assuming dedicated equivalent workers with 75% useful scheduling utilization:

| Independent worker GPUs | Compute-only elapsed time for the first delivery | First-delivery calendar estimate |
| ---: | --- | --- |
| 4 | 21–58 days | Roughly 6–12 weeks |
| 8 | 11–29 days | Roughly 4–8 weeks |
| 16 | 5–15 days | Roughly 3–6 weeks; development/review increasingly dominate |

These calendar estimates assume the four-GPU SSL allocation is separate or becomes available in parallel. They assume one primary implementer can continuously implement/review the methods while queued experiments run, and reviewers/corpus access are available. They include analysis and integration; they are not a guarantee of uninterrupted work or of a positive outcome. With only four total GPUs, reserve SSL time in addition to worker-pool time or revise the schedule explicitly.

## Hardware to provision

| Resource | Recommended allocation | Reason |
| --- | --- | --- |
| Independent trials | 16 fast CUDA GPUs for the full scope; ≥32 GB each, 48+ GB preferred for heavier variants | Parallel fold/seed jobs; begin with one training process/GPU; 8 workers can produce the earlier finalist delivery |
| SSL server | 4 × 80 GB CUDA GPUs on one node with fast GPU interconnect | Fixed global batch and two-view/masked pretraining; reusable for later fits |
| Worker CPUs/RAM | About 8 effective CPU cores and 16–24 GB available RAM per active job; ~256 GB RAM per four-GPU server | Cached features, audio transforms and independent loaders; preserve headroom |
| Analysis server | 32–64 effective CPU cores, 128 GB RAM | Patient bootstrap, corpus audit and report generation |
| Fast storage | 2–4 TB local NVMe per four-GPU server; 8–12 TB shared archive | Audio/features, selected/checkpoint-grid weights, resume state and immutable evidence |
| Networking | 10 GbE minimum preferred, 25 GbE useful for shared storage | Checkpoint/artifact collection without making inference wait on the network |
| Human review | Two independent trained sound reviewers plus adjudication access | P15 cannot be replaced by GPU capacity |

The existing PAFA smoke does not certify new-method VRAM. Keep physical PAFA batch 32 where specified: gradient accumulation over smaller batches is **not mathematically equivalent** for its patient-centroid loss. Use activation checkpointing or larger VRAM if needed, then verify loss/gradient behavior. Distributed PAFA requires an explicit decision about cross-rank centroids and gradient-correct gathering; do not silently change its objective. Ordinary fits should remain single-GPU initially.

Retain selected checkpoints, final resume/RNG state, all negative-result reports and prediction/provenance artifacts. Retain the declared checkpoint grid until the cross-fold epoch is locked. Newly generated unselected temporary weights may be cleaned only after checksummed acceptance; never delete historical/raw assets to fit this programme. Monitor disk below a 30 GiB floor. The current A filesystem has only about 93 GiB free and cannot serve as the sole new archive.

## Calendar and deliverables

- **Days 1–3:** verify execution access and dependencies; accept current campaign tail; lock data/selection rules; run physical-batch GPU smokes and update all cost ranges. Begin attention/current-PAFA paired CV. Audit SSL corpus and annotation review queue in parallel.
- **Days 3–10:** implement and screen the remaining families in independent batches. Start SSL once its corpus audit passes. Export source out-of-fold predictions and retain the checkpoint grid. Publish the first honest candidate/control comparison as soon as its paired folds are complete.
- **Weeks 2–3:** complete source-device screens, stability seeds and annotation adjudication; lock four combinations using validation evidence. Do not choose from displayed official/KAUH/outer-LODO results.
- **Weeks 3–5:** complete the combination ablations and frozen five-seed finalist evaluations, including source-selected LODO procedures. Generate KAUH, conformal, finite-N and engineering comparisons.
- **Weeks 5–8 reserve:** finish separately labelled adaptation studies, pretraining-seed checks, failed-run recovery, independent review and report/proposal integration. Retain both stronger and weaker results. Deliver a reproducible recommendation and explicit unmet targets if no candidate meets the goals.

For the existing physical hypotheses, separately reserve **2–5 working days after phantom hardware/setup are ready**. With three transducers, one 1,200-clip pass at 5 s plus 1 s gaps is about six hours of pure playback; the 100-clip replace/repeat measurements, setup, calibration, QC and reruns add time. This calculation is illustrative and must be replaced with actual clip durations/setup. Hardware: the registered phantom/replay setup, at least two physical transducers, repeatable mounting and recording chain. GPU experiments cannot substitute for H1/H3 or physical-device H2 evidence.

## Full-scope endpoint: five-seed evaluation of every candidate

For full official/four-LODO/five-seed evaluation of **every source candidate**, budget **36 × 5 splits × 5 seeds = 900 final fits** at their declared frontend. Adding separate no-SC/static-SC diagnostics where meaningful gives an upper bound of **1,800** final fits before deduplication, replacing the early milestone's 250. The main programme total becomes **1,147−250+900 = 1,797**, or at most **2,697** slots with the full frontend diagnostics. Do not construct contradictory SC combinations just to fill a matrix. This full endpoint is the requested scope; the earlier finalist table supplies useful results before the whole programme finishes.

For that larger option, reserve approximately **2,500–10,000 worker GPU-hours including contingency**, plus the separate SSL allocation. At 75% utilization that is about 18–70 compute days on eight workers or 9–35 on sixteen. Allow **6–12 calendar weeks with eight workers**, or **4–8 with sixteen**, with implementation/review running alongside training. Refine teacher counts before committing: crossfit distillation can add many fold-specific teachers. No claim of significance follows from trying more candidates; freeze the deployment choice using source validation before unlocking this larger evaluation table.

## Launch gates and remaining unknowns

The latest tracker at 16:00 local reports 39/100 dynamic-SC and 5/6 KD units; the earlier integrity audit covered 28/4. New tracker counts are not full checkpoint acceptance or fresh process-liveness proof. Preserve the current campaign's 26 October admission deadline unless its owner changes it.

The current agent environment still lacks usable GPU/SSH access and the needed KD student checkpoint/teacher-target bytes at their expected local paths. Additional physical hardware alone does not grant this session execution access. T0 for the new programme is **usable worker access + verified datasets + frozen source/deployment manifests**, not this document's timestamp. Complete the outstanding pilot before attributing engineering benefits to KD.

Before launch, lock per-family constants, inspect actual CPU/cgroup/GPU allocations, run five representative full-schedule epochs and validation/export, measure peak memory and step time, and regenerate estimates. Store an immutable identity `(stage, recipe, outer_fold, inner_fold, seed)`. Existing `cv0` naming does not encode seed: use separate run IDs per screening seed and explicit outer-fold cell identities to prevent collisions. Randomize/balance recipe order across hardware while pairing controls by seed/fold; do not confound one method with one GPU type.

Planning arithmetic and historical timing hashes: `outputs/planning/improvement-programme-20261024/{estimate.py,estimate.json,full_scope.py,full-scope.json}`. The scripts assert condition/run counts and totals, with first-delivery and full-scope estimates kept distinct. No infrastructure stress test, remote deployment, new training or driver change was performed to create this estimate.
