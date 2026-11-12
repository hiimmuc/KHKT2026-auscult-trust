# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

Current results: [`report.md`](report.md). State and open items: [`docs/handoff/HANDOFF.md`](docs/handoff/HANDOFF.md). Commands per stage: [`docs/REPRODUCE.md`](docs/REPRODUCE.md). Demo: [`app/README.md`](app/README.md).

Main line: **proposal v10** (`docs/registered/research-proposal-v10-en.md`). Registered study: **proposal v9** with pre-registration `docs/registered/prereg-v8.md` and amendments 1-5; registered criteria unchanged. Registered title (fixed): *Nghiên cứu giải pháp cải thiện độ tin cậy của mô hình phân loại âm thanh phổi dưới sự dịch chuyển thiết bị bằng hiệu chỉnh phổ và dự đoán Conformal cho hỗ trợ sàng lọc bệnh đường hô hấp*.

Model (29/10/2026): **AuscultTrust** = distilled student `kd_g4_k1` (BEATs + PAFA, trained static SC, gain6, epoch 66, soft targets from the equal-weight G4 teacher ensemble) + split-conformal layer. **Baseline** = BEATs + PAFA, fine-tuned, no SC. The registered v9 core (AST + Patch-Mix CL) is a separate, historical study; its P0/P1 and gate G below describe it. Phantom work is pending hardware.

Data in `../data/raw/`: ICBHI (920 wav, 6898 cycles, 126 patients), KAUH (336 wav = 112 patients x 3 filters), phantom replay takes. Vendored upstream code in `third_party/` (see `third_party/README.md`).

Dates: research cut-off 31/01/2027.

## Project (v9)

One question: does spectrum correction (SC) let a benchmark-level lung-sound classifier keep its predictions and its conformal coverage when the stethoscope changes, without labels from the new stethoscope?

- Arms: "SC pipeline" = AuscultTrust weights + SC at test; "baseline pipeline" = P0 weights, no SC; isolation contrast = P0 weights with vs without test-time SC (reported, no verdict).
- **G (Stage 1 gate):** ΔScore (AuscultTrust - P0) over 10 seeds (0-9), CV epoch, 95% t-interval lower bound > -1.5 points (margin = about half the published AST-FT to Patch-Mix CL gap). Seeds 0-4 are the benchmark report. If G fails: Stage 2 uses P0 + test-time SC.
- **H1 (invariance):** SC lowers the prediction flip rate between devices on the same sound. Primary: phantom transducer pairs [pending]. Cycle 4-class argmax, bootstrap by source patient.
- **H2 (coverage):** SC lowers |Δ| (Δ = prior-matched coverage - 0.90) on an unseen device, set size not larger. Primary: ICBHI LODO, D = mean over tested folds of |Δ_base| - |Δ_SC|, two-level bootstrap (seeds, patients) with recalibration in every draw; co-primary phantom. Null gate: a fold is tested only if |Δ_base| > 95th percentile of |Δ| over 200 exchangeable re-splits of source-device patients with the same sizes.
- **H3 (mechanism, pending):** explained fraction R²_gain / R²_ceiling of the paired log-mel shift, median over transducer pairs >= 0.8 (ceiling from a remove-and-replace repeat of the same transducer).
- **Positive controls (must pass, no claim):** KAUH-A flip rate drops with SC; KAUH-B2 |Δ| drops with SC on pairs that pass the null gate. KAUH filters are vendor software and near-linear, so SC should succeed by construction; a failure means broken SC code or a non-linear filter.
- **Findings, no direction:** F1 per-device Sp/Se/Score/HS; F2 LODO benchmark (P0, P4, AuscultTrust; optional AST-CE, SG-SCL) and class-mix vs device split of coverage error; F3 screening view (referral rate, screening Se/Sp; B1 ICBHI-calibrated threshold on KAUH); F4 SC practicality (n unlabelled target recordings, class-mix sensitivity, arithmetic vs geometric reference); F5 comparators, each with and without SC: A2 test-time statistics, TTA-EQ (cold start, no device info: mean softmax over K = 8 random smooth per-bin gain views from the P3 distribution, 6 dB SD, fixed seeds 1000-1007; calibration uses the same ensemble; view disagreement reported as a label-free shift score), k-shot recalibration. Comparators get no verdict; H2 is SC vs baseline. Multi-rate resampling consensus was rejected (low-pass only, no-op on 4 kHz data).
- Prior work [46] (Koo et al., arXiv 2605.29862v2): federated device leave-out on ICBHI + SPRSound, Score only, 3 seeds. Its deterministic device-mean removal failed on AKGC417L (Sp 11.49) and Yunting: treat SC risk R4 as high. Its stochastic spectral perturbation helped most (supports P1P3).
- Not claimed: a new algorithm; SOTA Score; disease diagnosis (screening endpoint = abnormal lung sound); guarantees on patients; KAUH software filters as real hardware.

### Stage 1 (model)

- Official 60/40 split, 4-class primary, 2-class view of the same predictions, Sp/Se/Score/HS per device and pooled, 5 seeds (0-4, same in every arm).
- Arms: P0 Patch-Mix CL; P1 = P0 + SC (`--spectrum_correction`); P1+P3 = SC + random per-bin gain (6 dB SD); P4 Freq-MixStyle p = 0.5 (competitor). AuscultTrust = P1 or P1+P3 by mean 3-fold CV Score (tie: P1). v8 variants P2, P6, P7, P8, P9 are not candidates; report them in an appendix if run.
- SC: STFT n_fft 1024, hop 512, 16 kHz; reference = arithmetic mean of training-device mean spectra; `sc_mode` (`--sc_mode`, `src/processing/correction.py:limit_coefficients`): `dynamic` (default) clips every bin to +-20 dB (4 kHz devices have no energy above 2 kHz); `static` leaves 50-2000 Hz unclipped, no-op outside. Not a reported ablation: at CV screening run both, keep the higher mean CV Score per SC arm, tie -> `dynamic`. Device from file name; target-device spectrum from unlabelled clips only.
- Primary result: refit all four arms at their CV epoch (`--selection fixed`), seeds 0-4, test once; seeds 5-9 for P0 and AuscultTrust (gate G). Literature column: best epoch on test, labelled optimistic. Stage-1 SC estimates test-device spectra from unlabelled official-test clips (transductive, as in [3]).
- Hand-off to Stage 2: frozen checkpoint + SHA-256, saved s_ref and coefficients, exported softmax (`export_probs.py`).

### Stage 2 (experiments; conformal score LAC, alpha 0.1)

- Unit follows labels: ICBHI and phantom = cycle, 4-class; KAUH = recording, normal vs abnormal (screening view: 8 s windows, recording p(abnormal) = mean over windows of 1 - p(normal); max as sensitivity). Triage: {normal} no referral, {abnormal} refer, {both} uncertain, empty re-record.
- E0 reference: official-test patients split into calibration/evaluation halves, stratified by device, 20 re-splits (`calibration_eval_splits`).
- Test 1 LODO: hold out all recordings of one device (Meditron, LittC2SE, Litt3200, AKGC417L); train on the rest minus 20% calibration patients (stratified by device); a patient with any recording on the held-out device leaves training and calibration (their held-out-device recordings stay in test); recipe and epoch count frozen from Stage 1; 5 seeds. AKGC417L fold is class-mix dominated (training keeps at most 321 of 1,864 crackle cycles): report, flag, add a macro mean without it. Arms: P0, P4, AuscultTrust (trained); P0 + test-time SC, TTA-EQ on P0 and AuscultTrust (inference only). Secondary protocol matching [46] (reference only): every multi-device patient removed; folds AKGC417L, Meditron, pooled Littmann.
- Test 2 KAUH, frozen Stage-1 model, no KAUH training. A: paired invariance across Bell/Diaphragm/Extended (flip rate, TV distance, set change) = positive control. B1: ICBHI threshold (E0 calibration halves) applied per filter (deployment check, F3). B2: 5-fold patient rotation, calibrate on source-filter recordings of the other 4 groups, test target filter of the held-out group, 6 ordered pairs = positive control. C: A and B again with SC (target spectrum per filter from unlabelled recordings) and with A2.
- Test 3 phantom [pending]: 1,200 ICBHI official-test cycles replayed per transducer; calibration/test halves and bootstrap by source patient; sweep ratio Y_B/Y_A (no reference sensor); 100-clip remove-and-replace repeat per transducer for the ceiling; conformal calibrated on transducer A clips (phantom-internal, allowed), tested on B clips.
- Stage 2 excludes the 2 patients that the published official split puts on both sides. ICBHI recording label for the screening view: abnormal if any cycle has a crackle or wheeze.

Arm names in code and configs: P0 = `baseline`, P1 = `sc`, P1+P3 = `sc_gain`, P4 = `freq_mixstyle`; gate G = the non-inferiority check in `summary.py`; Stage 1 = the `train` experiment.

## Commands

Two envs: analysis/tests `.venv` and training `.venv-train` (Python 3.10, `timm==0.4.5`). Data: `bash scripts/download_data.sh`, `python scripts/prepare_data.py`. Details: `docs/SETUP.md`, `docs/REPRODUCE.md`.

- Tests: `uv run --no-project --python .venv/bin/python python -m pytest -q tests` (single: append `tests/training/test_training.py::test_summary_counts_match_metrics`).
- Registered AST campaign (one `RUN_ID`): `bash scripts/run.sh screen|summary|final|report`. Grid, LODO, SC, ensembles, distillation, replay: `docs/REPRODUCE.md`. Same `RUN_ID` skips finished units and resumes interrupted ones from `last.pth`. One unit: `PYTHONPATH=. .venv-train/bin/python -m src.training.main --config experiments/train/base.yaml,experiments/train/<arm>.yaml [--cv_folds 3 --cv_fold k | --selection fixed --report_epochs E] --seed S`.
- Freeze/export: `python -m src.training.export_probs <outputs unit dir> --ckpt report_epoch_<E>.pth` -> softmax, embeddings, `export/model.sha256`.
- Earlier-phase tools are in `../archive/auscult-trust-phase0`; code removed on 03/11/2026 is in git history (`git show d722ae8`).
- CPU-heavy extraction: prefix `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4`.

## Code map

Run from repo root, imports are `src.*` (`pyproject.toml` sets `pythonpath`).

- `src/paths.py` (REPO, DATA = `AT_DATA` or `../data`, THIRD_PARTY = `third_party/`, OUTPUTS, CHECKPOINTS), `src/runs.py` (run directories `outputs|checkpoints/<exp>/<run_id>/<cell>/<unit>/`, `meta.json`, `twin_dir`). `src/artifacts.py` owns atomic writes and `sha256_file`; `src/provenance.py` owns environment and prediction certificates.
- `src/processing/cycles.py` owns cycle preparation; `src/inference/` owns frozen prediction and test-time adaptation. API and experiment workflows use these shared modules; evaluation primitives belong in `src/evaluation/calibration.py`, `recordings.py`, and `metrics.py`.
- Docstrings: Google style. Code, comments, scripts and configs are self-contained: no references to documents, versions, sections or stage numbers; plain names.

## Data tiers (hard rules)

| Tier | Data | Use |
| --- | --- | --- |
| 0 | ICBHI, KAUH (HF_Lung_V1 Phase 0 only) | Train, select, calibrate, shift experiments |
| 1 | Lung phantom recordings | Evaluate only. Weights and s_ref frozen and hashed before any phantom recording. No model selection. Phantom-internal conformal calibration (transducer A clips) is allowed for phantom tests only |
| 2 | Hospital patients | Out of scope for v9. Never read or load |

- Splits are patient-disjoint. Always. Calibration patients are never used for training, selection or evaluation.
- KAUH: 3 filter renderings of one recording (Bell/Diaphragm/Extended) share a patient and a split; unit = whole recording; label = recording-level sound label mapped to normal vs abnormal; `Bronchial`-only recording excluded. KAUH device = Littmann 3200 (same model as ICBHI Litt3200).
- ICBHI: device is confounded with class mix and site. 4 patients (112, 158, 218, 226) recorded on two devices.
- Target labels are used only for evaluation, V5 k-shot and V4-oracle rows. SC and A2 use unlabelled target audio only.
- Never join datasets across different people.

## Experiment rules
>
- >= 5 seeds for trained arms, same seeds across arms (paired). Report mean +- SD. A gain smaller than seed SD is not a result.
- Report Sp, Se, Score, HS per device and pooled; coverage, |coverage - nominal|, set size, singleton and empty-set rate; flip rate and TV distance for paired-device data. Compare coverage with the binomial CI of nominal for the actual test size.
- Patient-level bootstrap for CIs (1,000), paired across SC/no-SC on the same resampled patients. Never cycle-level. Phantom: clip bootstrap.
- Log config, seed, data hash, checkpoint hash, s_ref hash for every run.
- Tune on grouped CV over training patients. Never on test. Test-selected numbers only in the labelled literature column.
- Negative results are kept and reported.

## Working conventions

- `third_party/*` is vendored upstream code, byte-identical to the pinned commits in `third_party/SOURCES.json` (hashed by certificates and bundles). Never edit it; load it by path (`src/training/upstream.py`).
- Docs: results in `report.md`, detail in `docs/reports/`, handoff in `docs/handoff/` (written when a session runs out of context), registered documents in `docs/registered/`, plans in `docs/plans/`, decisions in `docs/decision-logs/`.
- Results: `outputs|checkpoints/<exp>/<run_id>/<cell>/<unit>/`; `<exp>` = `train`, `<cell>` = variant (config stems joined by `+`, no `base`), `<unit>` = `cv<k>` or `seed<s>`, `run_id` = `YYYYMMDD-HHMMSS[_$RUN_TAG]` or a `RUN_ID` set to resume one campaign. Never write elsewhere, never reuse a unit with another config.
- Frozen Phase 0 code lives in `../archive`; the wrapper holds `../data`, `../docs`.
- Install a missing package before reaching for a workaround.
- Simplest change that works. No speculative abstractions.
- Do not modify `data/raw/` or committed split files.
- Do not invent dataset statistics, citations, or results. Mark unknowns as TODO.
- Claims in docs use tags: [Fact], [Interpretation], [Hypothesis], [Assumption], [Pending].
- Code comments and docs in English.
