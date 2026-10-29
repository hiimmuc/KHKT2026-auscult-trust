# Lung-phantom replay: ICBHI audio -> speaker in the phantom -> electret / phone recording

| Type | Date | Status |
| --- | --- | --- |
| report | 2026-10-29 | current (exploratory; no registered criterion changed) |

Setup: each ICBHI recording is played by a speaker placed inside a lung phantom and recorded at the phantom surface by two
capture devices, an electret microphone and a phone microphone (chain: ICBHI audio -> speaker -> phantom -> electret / phone).
Question: how far do the frozen AuscultTrust predictions move from the predictions on the original audio, and which automatic signal-processing steps
bring them back? Evidence cut-off: 29/10/2026 05:15 (+07).

Data (tier 1 phantom data: evaluate only, weights frozen): Drive folder `1hyR7WLlpzxCkyaBDQtcPkvW8NaMS0E_S`, subfolders `phone` and `electret`: 20 ICBHI recordings each (patients 101-107,
four ICBHI devices, 20 takes per capture device, 48 kHz mono, 17-37 s). Stored in `../data/raw/phantom_replay/`. The `pipeline test`
folder (serving-app screenshots) is not used. The takes were provided already recorded; no new playback was done in this session. Phantom geometry, speaker and microphone
placement are not described in the data: treated as given [Assumption].

Code: `scripts/replay_analysis.py`; results `outputs/replay/phantom-replay-20261028/` (`takes.csv`, `summary.csv`, `cycles.csv.gz`,
`selection.json`, `transfer_functions.npz`, processed audio in `audio/<capture>/<variant>/`); notebook `notebooks/04_replay_phantom.ipynb`;
tables `outputs/analysis/replay/`; figures `outputs/figures/replay/`.

## 1. Comparison with the original dataset audio

Every take is aligned to its original (band-limited GCC-PHAT over the whole take) and then to each annotated cycle (+-50 ms).

| Measure (median over 20 takes) | electret | phone |
| --- | ---: | ---: |
| Playback SNR (dB) | 16.7 | 8.2 |
| Alignment peak (SD above other lags) | 41 | 11 |
| Clock drift (ppm) | -4 | 2 |
| Per-cycle residual delay SD (ms) | 0.55 | 0.19 |
| Magnitude-squared coherence, 50-2000 Hz, PSD-weighted | 0.15 | 0.02 |
| Log-spectral distance to original (dB, level-free) | 5.3 | 6.6 |

[Fact] Alignment is reliable (no drift, sub-millisecond per-cycle residual), so the low coherence is not an alignment error: the
playback path (speaker, phantom, capture device) removes most of the waveform-level agreement. [Fact] The transfer function relative to the
200-1000 Hz level (`transfer_functions.npz`, figure `transfer_functions.png`) shows +10 to +15 dB below 100 Hz, a roll-off above
4 kHz, and for the phone a +10 to +20 dB rise at 1-4 kHz with narrow lines. [Fact] Spectrograms (`spectrogram_*.png`) show stationary
narrow-band lines (phone) and strong periodic low-frequency clicks (electret) that are absent from the original; their source
(electronics, mains, the speaker) is not identified [Pending].

## 2. Processing variants (automatic, no labels)

Every variant applies the static SC of the model; the coefficient is estimated from unlabelled takes of the capture device only
(except `*_rec`, one take, and `eq`, which needs paired originals).

| Variant | Steps |
| --- | --- |
| raw | aligned take, no SC |
| raw_sc / raw_scd | static / dynamic SC from all takes of the capture device |
| raw_sc_rec | static SC from the take itself (serving default for an unknown device) |
| dn_sc / dn_scd | 50-2000 Hz band-pass, loudness to the ICBHI median RMS, spectral (Wiener) denoise with a minimum-statistics floor, then SC |
| tdn_sc / tdn_scd | as dn, preceded by removal of stationary narrow-band lines (bins above the frequency-smoothed median by > 6 dB) |
| eq / dn_eq / tdn_eq | inverse capture-device transfer function (geometric mean of the other 19 takes against their originals, boost <= 30 dB, leave-one-recording-out) then the ICBHI device test coefficient |
| dn_sc_tta | dn_sc with TTA-EQ (8 fixed-seed 6 dB gain views) |

## 3. Result (distilled AuscultTrust, 5-seed softmax ensemble; 20 takes, all cycles)

Flip = share of cycles whose 4-class prediction differs from the same model's prediction on the original audio; flip2 = same for
normal vs abnormal; Score = ICBHI 4-class Score against the cycle labels.

| Variant | Flip electret | Flip phone | flip2 electret | flip2 phone | Score electret | Score phone |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | 0.547 | 0.581 | 0.436 | 0.441 | 45.2 | 43.6 |
| raw_sc | 0.503 | 0.430 | 0.402 | 0.352 | 49.2 | 53.6 |
| raw_sc_rec | 0.492 | 0.458 | 0.385 | 0.385 | 52.4 | 48.5 |
| dn_sc | 0.413 | 0.458 | 0.313 | 0.341 | 55.6 | 53.9 |
| dn_scd | 0.430 | 0.374 | 0.324 | 0.302 | 54.8 | 58.5 |
| **tdn_sc** | **0.408** | 0.380 | **0.313** | 0.291 | **55.8** | 58.6 |
| tdn_scd | 0.436 | **0.358** | 0.324 | **0.285** | 54.1 | **60.9** |
| eq | 0.480 | 0.402 | 0.363 | 0.291 | 52.7 | 57.7 |
| dn_eq | 0.436 | 0.363 | 0.335 | 0.302 | 55.7 | 60.7 |
| tdn_eq | 0.436 | 0.346 | 0.341 | 0.296 | 55.1 | 60.6 |

Source: `outputs/analysis/replay/pooled_20_takes_kd_g4_k1_ens5.csv` (built from `cycles.csv.gz`). Original audio of the training-split
patients (103, 105, 106, 107) was seen in training, so its Score (91.7 pooled) is not a reference; the 9 test-split takes per capture
device (patients 101, 102, 104; 182 cycles) give 63.8 on the original (seed 0) and 26-37 raw / 44-51 after the best processing (single-seed rows of `summary.csv`).

[Fact] Unprocessed replay changes more than half of the cycle predictions (flip 0.55 and 0.58) and costs 27-38 Score points on the
test-split takes (single seed: 26-37 against 63.8). [Fact] SC alone reduces the flip rate by 0.04 (electret) and 0.15 (phone). [Fact] Tone removal, denoising and the
capture-device SC reduce it further, to 0.41 (electret) and 0.36-0.38 (phone): a Score of 54-61 on the pooled takes against 44-45 raw.
[Fact] The leave-one-out equaliser (`eq`, `tdn_eq`) is no better than the label-free `tdn_sc`; it needs paired originals and gives
no extra gain at this sample size. [Fact] `dn_eq` and `dn_eq12` are identical: the equaliser boost never exceeds 12 dB.
[Fact] TTA-EQ on top of `dn_sc` changes nothing (flip 0.42-0.52 either way).
[Interpretation] What remains (flip about 0.4, flip2 about 0.3) comes from content the playback path destroys (coherence 0.02-0.15,
phone SNR 8 dB), not from spectrum shape; spectrum correction and denoising cannot restore it.

## 4. Automatic choice and its limit

[Fact] Selecting the variant by the lowest flip rate on the training-split takes and evaluating on the test-split takes picks a
different variant for each model and capture device (`selection.json`: `tdn_sc` for the phone, `raw_scd` or `tdn_eq` for the
electret) and does not beat the fixed rule `tdn_sc`. The flip rate of the training-split takes is almost flat across variants (0.38-0.48),
so it carries little signal. [Recommendation] Use the fixed label-free chain `band-pass -> tone removal -> denoise -> device static SC`
(`tdn_sc`) for takes from an unknown microphone; the dynamic SC (`tdn_scd`) is equal within noise.

## 5. Caveats

- 20 takes per capture device, 9 per device on the test split (182 cycles); differences of 0.03 in flip rate are inside the noise.
- One phantom, one speaker position per take set; the transfer functions describe this chain (speaker -> phantom -> capture device),
  not a stethoscope on a patient. The phantom is a replay of recorded sound, so it carries no new physiology.
- [Deviation from the tier-1 rule "evaluate only, no model selection"] The signal-processing chain was chosen on these takes (tone
  threshold 6 dB, Wiener over-subtraction 1.5, noise floor at the 15th percentile, variant ranking). The model weights, the SC
  reference and the coefficients' bounds were not touched, but the processing numbers are optimistic for new phantom takes and
  must be frozen and re-tested on a new capture before any claim.
- Variant selection and Score use the same takes. No labels were used by the label-free variants; labels only score them.
