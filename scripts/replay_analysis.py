"""Replay study: ICBHI recordings played through a speaker and re-recorded by two capture devices (phone, electret).

Every take is aligned to its original ICBHI recording (band-limited GCC-PHAT), measured against it (delay, clock drift,
SNR, coherence, log-spectral distance, transfer function), cut into the original annotated cycles, processed by a set of
signal-processing variants, and scored by frozen models (distilled AuscultTrust `kd_g4_k1` and the gain6 anchor, seeds 0-4).

    PYTHONPATH=. .venv/bin/python scripts/replay_analysis.py --takes ../data/raw/phantom_replay --out outputs/replay/phantom-replay-20261028

Variants (wave transform -> spectrum correction before the model). Every replay cycle is cut after a per-cycle residual
alignment (+-50 ms GCC-PHAT against the original).
    original      original recording, its ICBHI device test coefficient (the model's own test condition)
    original_nosc original recording, no SC
    raw           aligned take, no processing, no SC
    raw_sc        aligned take, static SC coefficient estimated from all takes of that capture device (unlabelled; AuscultTrust)
    raw_sc_rec    aligned take, static SC estimated from this take only (the serving default for an unknown device)
    raw_scd       aligned take, SC as raw_sc but dynamic (every bin clipped to +-20 dB) instead of static
    bp_sc         50-2000 Hz band-pass, loudness set to the ICBHI median RMS, then device static SC
    dn_sc         bp + spectral (Wiener) denoising with a minimum-statistics noise floor, then device static SC
    dn_scd        dn with dynamic (+-20 dB) device SC
    dn_sc_tta     dn_sc with TTA-EQ: mean softmax over 8 fixed-seed smooth gain views (6 dB SD)
    eq            bp, then the inverse capture-device transfer function (boost <= 30 dB) estimated leave-one-recording-out
                  from the other takes of the same capture device against their originals, then the original ICBHI device's
                  test coefficient
    dn_eq         dn, then eq
    dn_eq12       dn_eq with the equaliser boost capped at 12 dB (less noise amplification)
    tdn_sc/_scd   tone removal (stationary narrow-band lines) + denoise, then device static / dynamic SC
    tdn_eq        tone removal + denoise, then the inverse capture transfer function and the ICBHI device coefficient
Each model also gets a seed-ensemble row (`<model>_ens5`: mean softmax over seeds 0-4).

The `eq` variants use paired original audio of the other recordings (a phantom-internal calibration of the capture chain);
the evaluated recording never enters its own correction. Patients 103, 105, 106, 107 belong to the official training split,
101, 102, 104 to the test split: summaries are reported for both groups.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, coherence, sosfiltfilt, welch

from src.paths import CHECKPOINTS, DATA, OUTPUTS
from src.processing.audio import SR, load_wav
from src.processing.correction import _stft, apply_spectrum_correction, mean_spectrum, spectrum_coefficients
from src.processing.cycles import cut_pad
from src.processing.icbhi import DEVICES
from src.training.config import load_args
from src.inference.model import images_of, load_frozen, predict_images

ICBHI = DATA / "raw" / "icbhi" / "ICBHI_final_database"
MODELS = {
    "kd_g4_k1": ("distill-20261028-final/kd_g4_k1", 66),
    "anchor": ("followup-20261026-final/gain6-cv66", 66),
}
TEST_PATIENTS = {"101", "102", "104"}
BAND = (50.0, 2000.0)
N_FFT, HOP = 1024, 512
FREQS = np.fft.rfftfreq(N_FFT, 1 / SR)
INBAND = (FREQS >= 100) & (FREQS <= 2000)


def gcc_lag(ref, rec, max_lead=0.5, band=(100.0, 2000.0)):
    """Delay (s) of `rec` behind `ref` searched over the whole take, and the peak height in SDs of the other lags."""
    n = 1 << int(np.ceil(np.log2(len(ref) + len(rec))))
    R, X = np.fft.rfft(ref, n), np.fft.rfft(rec, n)
    f = np.fft.rfftfreq(n, 1 / SR)
    g = X * np.conj(R)
    g = np.where((f >= band[0]) & (f <= band[1]), g / np.maximum(np.abs(g), 1e-12), 0)
    cc = np.fft.irfft(g, n)
    lags = np.r_[np.arange(0, max(len(rec) - len(ref) // 4, SR)), np.arange(-int(max_lead * SR), 0)]
    vals = cc[lags]
    k = int(np.argmax(vals))
    rest = vals[np.abs(lags - lags[k]) > int(0.005 * SR)]
    return lags[k] / SR, float((vals[k] - rest.mean()) / (rest.std() + 1e-12))


def shift(x, lag, n):
    """Advance `x` by `lag` s, zero-filled, cut or padded to `n` samples."""
    k = int(round(lag * SR))
    y = x[k:] if k >= 0 else np.r_[np.zeros(-k, x.dtype), x]
    return np.r_[y, np.zeros(max(n - len(y), 0), x.dtype)][:n].astype(np.float32)


def local_lag(ref, x, max_shift=0.05):
    """Residual delay (s) of the aligned `x` behind `ref` within +-`max_shift`, by band-limited GCC-PHAT."""
    n = 1 << int(np.ceil(np.log2(2 * len(ref))))
    f = np.fft.rfftfreq(n, 1 / SR)
    g = np.fft.rfft(x, n) * np.conj(np.fft.rfft(ref, n))
    g = np.where((f >= 100) & (f <= 2000), g / np.maximum(np.abs(g), 1e-12), 0)
    cc = np.fft.irfft(g, n)
    k = int(max_shift * SR)
    lags = np.r_[np.arange(0, k + 1), np.arange(-k, 0)]
    vals = cc[lags]
    return lags[int(np.argmax(vals))] / SR, float((vals.max() - vals.mean()) / (vals.std() + 1e-12))


def drift_ppm(ref, x, win=3.0, hop=1.0):
    """Clock drift (ppm) = slope of the residual delay over 3 s windows of the aligned take (peaks above 4 SD only)."""
    t, d = [], []
    for s in np.arange(0, len(ref) / SR - win, hop):
        i = slice(int(s * SR), int((s + win) * SR))
        lag, pk = local_lag(ref[i], x[i])
        if pk > 4:
            t.append(s + win / 2)
            d.append(lag)
    return 1e6 * np.polyfit(t, d, 1)[0] if len(t) >= 4 else np.nan


def snr_db(take, lag, n):
    """Playback power over the noise floor (pre-roll if it lasts 0.3 s, else the quietest 10% of 32 ms frames)."""
    k = int(lag * SR)
    seg = take[max(k, 0): max(k, 0) + n]
    if k > 0.3 * SR:
        noise = take[: k - int(0.05 * SR)]
        pn = np.mean(noise ** 2)
    else:
        fr = seg[: len(seg) // 512 * 512].reshape(-1, 512)
        pn = np.percentile((fr ** 2).mean(1), 10)
    return 10 * np.log10(np.mean(seg ** 2) / max(pn, 1e-12))


def psd(x):
    return welch(x, SR, nperseg=N_FFT, noverlap=N_FFT - HOP)[1]


def lsd_db(ref, rec):
    """Log-spectral distance (dB) of mean-normalised Welch spectra, 100-2000 Hz (level-free shape distance)."""
    a, b = 10 * np.log10(psd(ref)[INBAND] + 1e-12), 10 * np.log10(psd(rec)[INBAND] + 1e-12)
    return float(np.sqrt(np.mean(((a - a.mean()) - (b - b.mean())) ** 2)))


def bandpass_level(x, rms):
    y = sosfiltfilt(butter(4, BAND, "bandpass", fs=SR, output="sos"), x)
    return (y * rms / max(np.sqrt(np.mean(y ** 2)), 1e-9)).astype(np.float32)


def denoise(x, alpha=1.5, floor=0.1):
    """Wiener-style spectral subtraction; noise magnitude per bin = 15th percentile over frames (minimum statistics)."""
    import librosa
    S = _stft(x, N_FFT, HOP)
    mag = np.abs(S)
    noise = np.percentile(mag, 15, axis=1, keepdims=True)
    gain = np.sqrt(np.maximum(1 - alpha * noise ** 2 / np.maximum(mag ** 2, 1e-12), floor ** 2))
    return librosa.istft(S * gain, hop_length=HOP, length=len(x)).astype(np.float32)


def remove_tones(x, thresh_db=6.0, width=9):
    """Attenuate stationary narrow-band lines (hum, speaker or phone tones): bins whose temporal median magnitude exceeds
    the frequency-smoothed median by more than `thresh_db` are pulled down to the smoothed level."""
    import librosa
    from scipy.ndimage import median_filter
    S = _stft(x, N_FFT, HOP)
    med = np.median(np.abs(S), axis=1)
    ratio = med / np.maximum(median_filter(med, size=width, mode="nearest"), 1e-12)
    gain = np.where(ratio > 10 ** (thresh_db / 20), 1 / ratio, 1.0)
    return librosa.istft(S * gain[:, None], hop_length=HOP, length=len(x)).astype(np.float32)


def annotations(stem):
    a = np.loadtxt(ICBHI / f"{stem}.txt", ndmin=2)
    return [(s, e, int(c) + 2 * int(w)) for s, e, c, w in a]


def cycle_offsets(ref, x, ann, max_shift=0.05):
    """Per-cycle residual delay (s) of the aligned take, searched within +-50 ms; 0 when the peak is weak."""
    out = []
    for s, e, _ in ann:
        i = slice(int(s * SR), max(int(e * SR), int(s * SR) + SR // 2))
        lag, pk = local_lag(ref[i], x[i], max_shift)
        out.append(lag if pk > 4 else 0.0)
    return out


def cycles(x, ann, args, offsets=None):
    out = []
    for k, (s, e, _) in enumerate(ann):
        o = int(round((offsets[k] if offsets else 0.0) * SR))
        a, b = max(int(s * SR) + o, 0), max(int(e * SR) + o, 1)
        seg = torch.from_numpy(np.ascontiguousarray(x[a:b]))[None]
        out.append(cut_pad(seg, args.desired_length, SR, args.pad_types)[0].numpy())
    return out


def weighted_coherence(ref, x):
    """Magnitude-squared coherence averaged over 50-2000 Hz with the original's PSD as weight (where the lung sound is)."""
    f, coh = coherence(ref, x, SR, nperseg=N_FFT, noverlap=N_FFT - HOP)
    w = psd(ref)[: len(coh)] * ((f >= BAND[0]) & (f <= BAND[1]))
    return float(np.sum(coh * w) / np.sum(w))


def score(p, y):
    pr = p.argmax(1)
    sp = (pr[y == 0] == 0).mean() if (y == 0).any() else np.nan
    se = (pr[y > 0] == y[y > 0]).mean() if (y > 0).any() else np.nan
    return sp, se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--takes", required=True, help="folder with <capture>/<stem>_<capture>.wav")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    captures = sorted(p.name for p in Path(a.takes).iterdir() if p.is_dir() and any(p.glob("*_" + p.name + ".wav")))
    stems = sorted({f.stem.rsplit("_", 1)[0] for c in captures for f in (Path(a.takes) / c).glob("*.wav")})
    orig = {s: load_wav(ICBHI / f"{s}.wav", SR) for s in stems}
    ref_rms = float(np.median([np.sqrt(np.mean(v ** 2)) for v in (load_wav(f, SR) for f in sorted(ICBHI.glob("*.wav"))[::8])]))

    # Align and measure
    rows, aligned = [], {}
    for c in captures:
        for s in stems:
            take = load_wav(Path(a.takes) / c / f"{s}_{c}.wav", SR)
            lag, peak = gcc_lag(orig[s], take)
            x = shift(take, lag, len(orig[s]))
            aligned[c, s] = x
            rows.append(dict(capture=c, stem=s, patient=s[:3], device=s.split("_")[-1], split="test" if s[:3] in TEST_PATIENTS else "train",
                             take_sec=len(take) / SR, original_sec=len(orig[s]) / SR, lag_sec=lag, gcc_peak_sd=peak,
                             drift_ppm=drift_ppm(orig[s], x), snr_db=snr_db(take, lag, len(orig[s])),
                             coherence=weighted_coherence(orig[s], x), lsd_db=lsd_db(orig[s], x),
                             rms_ratio_db=20 * np.log10(np.sqrt(np.mean(x ** 2)) / np.sqrt(np.mean(orig[s] ** 2)))))
    takes = pd.DataFrame(rows)

    # Capture transfer functions |H| = sqrt(PSD_take / PSD_original) and leave-one-recording-out inverse EQ
    tf = {(c, s): np.sqrt(psd(bandpass_level(aligned[c, s], 1.0)) / np.maximum(psd(bandpass_level(orig[s], 1.0)), 1e-14))
          for c in captures for s in stems}
    np.savez_compressed(out / "transfer_functions.npz", freqs=FREQS, **{f"{c}__{s}": v for (c, s), v in tf.items()})

    def inverse_eq(c, s, max_gain_db=30.0):
        """Inverse of the geometric-mean |H| of the other takes; boost capped at `max_gain_db`, cut at -30 dB, 50-2000 Hz only."""
        h = np.exp(np.mean([np.log(np.maximum(tf[c, t], 1e-6)) for t in stems if t != s], 0))
        g = np.clip(-20 * np.log10(h), -30.0, max_gain_db)
        return np.where((FREQS >= BAND[0]) & (FREQS <= BAND[1]), 10 ** (g / 20), 1.0)

    # Variants per take (whole recording, then cycles)
    waves = {}
    for c in captures:
        bp = {s: bandpass_level(aligned[c, s], ref_rms) for s in stems}
        dn = {s: bandpass_level(denoise(bp[s]), ref_rms) for s in stems}
        tn = {s: bandpass_level(remove_tones(bp[s]), ref_rms) for s in stems}
        tdn = {s: bandpass_level(denoise(tn[s]), ref_rms) for s in stems}
        waves[c] = dict(raw={s: aligned[c, s] for s in stems}, bp=bp, dn=dn, tn=tn, tdn=tdn,
                        tdn_eq={s: apply_spectrum_correction(tdn[s], inverse_eq(c, s)).astype(np.float32) for s in stems},
                        eq={s: apply_spectrum_correction(bp[s], inverse_eq(c, s)).astype(np.float32) for s in stems},
                        dn_eq={s: apply_spectrum_correction(dn[s], inverse_eq(c, s)).astype(np.float32) for s in stems},
                        dn_eq12={s: apply_spectrum_correction(dn[s], inverse_eq(c, s, 12.0)).astype(np.float32) for s in stems})
    # per-cycle residual delay (+-50 ms) of every take, from the band-passed take against the original
    offsets = {(c, s): cycle_offsets(orig[s], waves[c]["bp"][s], annotations(s)) for c in captures for s in stems}
    takes["cycle_offset_ms_sd"] = [1000 * np.std(offsets[r.capture, r.stem]) for r in takes.itertuples()]

    records, probs = [], []
    for model, (cell, ep) in MODELS.items():
        for seed in a.seeds:
            unit = OUTPUTS / "train" / cell / f"seed{seed}"
            args = load_args(unit)
            st = np.load(unit / "sc_state.npz")
            ref_spec = st["reference"]
            dev_coef = {d: st[f"test_coefficients_{i}"] if f"test_coefficients_{i}" in st.files else st[f"train_coefficients_{i}"]
                        for i, d in enumerate(DEVICES)}

            def sc_from(ws, mode=None):
                return spectrum_coefficients({"x": mean_spectrum(ws)}, reference=args.sc_reference, sc_mode=mode or args.sc_mode,
                                             limit_freq_low=args.sc_limit_freq_low, limit_freq_high=args.sc_limit_freq_high,
                                             limit_freq_diff=args.sc_limit_freq_diff, reference_spectra={"ref": ref_spec})["x"]

            model_nn, clf = load_frozen(args, CHECKPOINTS / "train" / cell / f"seed{seed}" / f"report_epoch_{ep}.pth")
            batch = []  # (capture, variant, stem, cycle waves)
            for s in stems:
                ann = annotations(s)
                cyc = cycles(orig[s], ann, args)
                d = s.split("_")[-1]
                batch.append(("original", "original", s, [apply_spectrum_correction(w, dev_coef[d]).astype(np.float32) for w in cyc]))
                batch.append(("original", "original_nosc", s, cyc))
            for c in captures:
                V = waves[c]
                cut = {v: {s: cycles(V[v][s], annotations(s), args, offsets[c, s]) for s in stems} for v in V}
                pool = {v: [w for s in stems for w in cut[v][s]] for v in ("raw", "bp", "dn", "tdn")}
                coef_dev = {v: sc_from(pool[v]) for v in pool}
                coef_dyn = {v: sc_from(pool[v], "dynamic") for v in ("raw", "dn", "tdn")}

                def sc(ws, coef):
                    return [apply_spectrum_correction(w, coef).astype(np.float32) for w in ws]
                for s in stems:
                    d = s.split("_")[-1]
                    raw = cut["raw"][s]
                    batch += [(c, "raw", s, raw), (c, "raw_sc", s, sc(raw, coef_dev["raw"])), (c, "raw_sc_rec", s, sc(raw, sc_from(raw))),
                              (c, "raw_scd", s, sc(raw, coef_dyn["raw"])), (c, "bp_sc", s, sc(cut["bp"][s], coef_dev["bp"])),
                              (c, "dn_sc", s, sc(cut["dn"][s], coef_dev["dn"])), (c, "dn_scd", s, sc(cut["dn"][s], coef_dyn["dn"])),
                              (c, "tdn_sc", s, sc(cut["tdn"][s], coef_dev["tdn"])), (c, "tdn_scd", s, sc(cut["tdn"][s], coef_dyn["tdn"]))]
                    batch += [(c, v, s, sc(cut[v][s], dev_coef[d])) for v in ("eq", "dn_eq", "dn_eq12", "tdn_eq")]
            flat = [w for *_, ws in batch for w in ws]
            images = images_of(flat, args)
            p = predict_images(images, model_nn, clf, method=args.method)
            # TTA-EQ (8 fixed-seed smooth gain views, 6 dB SD) on the denoised + SC replay cycles
            from src.inference.adaptation import TTA_DB, TTA_SEED0, gain_curves
            from src.models.encoders import get_encoder
            sd = TTA_DB * get_encoder(args.encoder).db_scale
            i, extra = 0, []
            for c, v, s, ws in batch:
                if v == "dn_sc":
                    im = images[i: i + len(ws)]
                    pt = np.mean([predict_images(im, model_nn, clf, gain_curves(len(im), im.shape[-1], sd, TTA_SEED0 + k), method=args.method)
                                  for k in range(8)], 0)
                    extra.append(((c, "dn_sc_tta", s, ws), pt))
                i += len(ws)
            batch += [b for b, _ in extra]
            p = np.concatenate([p] + [q for _, q in extra])
            i = 0
            for c, v, s, ws in batch:
                ann = annotations(s)
                for k in range(len(ws)):
                    records.append(dict(model=model, seed=seed, capture=c, variant=v, stem=s, cycle=k, label=ann[k][2],
                                        split="test" if s[:3] in TEST_PATIENTS else "train"))
                    probs.append(p[i])
                    i += 1
            del model_nn, clf
            torch.cuda.empty_cache()
            print(model, seed, "done", flush=True)

    cyc = pd.DataFrame(records)
    P = np.array(probs)
    for j, name in enumerate(["p_normal", "p_crackle", "p_wheeze", "p_both"]):
        cyc[name] = P[:, j]
    pc = ["p_normal", "p_crackle", "p_wheeze", "p_both"]
    ens = cyc.groupby(["model", "capture", "variant", "stem", "cycle", "label", "split"], as_index=False)[pc].mean()
    ens["model"] = ens.model + "_ens5"
    ens["seed"] = -1
    cyc = pd.concat([cyc, ens], ignore_index=True)
    cyc["pred"] = cyc[pc].to_numpy().argmax(1)

    # Agreement with the original-audio prediction of the same model, seed and cycle; Score against labels
    key = ["model", "seed", "stem", "cycle"]
    ref = cyc[cyc.variant == "original"].set_index(key)[["pred", "p_normal", "p_crackle", "p_wheeze", "p_both"]]
    j = cyc.join(ref, on=key, rsuffix="_orig")
    j["flip"] = (j.pred != j.pred_orig).astype(float)
    pc = ["p_normal", "p_crackle", "p_wheeze", "p_both"]
    j["tv"] = 0.5 * np.abs(j[pc].to_numpy() - j[[c + "_orig" for c in pc]].to_numpy()).sum(1)
    j["flip2"] = ((j.pred > 0) != (j.pred_orig > 0)).astype(float)
    j["abn"] = (j.pred > 0) != (j.label > 0)
    summ = []
    for (m, c, v, sp_), g in j.groupby(["model", "capture", "variant", "split"]):
        per_seed = []
        for seed, h in g.groupby("seed"):
            sp, se = score(h[pc].to_numpy(), h.label.to_numpy())
            per_seed.append(dict(flip=h.flip.mean(), flip2=h.flip2.mean(), tv=h.tv.mean(), sp=sp, se=se, score=50 * (sp + se),
                                 acc2=1 - h.abn.mean()))
            pass
        df = pd.DataFrame(per_seed)
        summ.append(dict(model=m, capture=c, variant=v, split=sp_, n_cycles=len(g) // g.seed.nunique(),
                         **{k + "_mean": df[k].mean() for k in df}, **{k + "_sd": df[k].std(ddof=1) for k in df}))
    summ = pd.DataFrame(summ)

    # Automatic variant choice: lowest flip rate on the training-split takes, evaluated on the test-split takes
    choice = {}
    for (m, c), g in summ[(summ.capture != "original") & (summ.split == "train") & ~summ.variant.isin(["raw"])].groupby(["model", "capture"]):
        best = g.sort_values(["flip_mean", "tv_mean"]).iloc[0].variant
        t = summ[(summ.model == m) & (summ.capture == c) & (summ.variant == best) & (summ.split == "test")].iloc[0]
        r = summ[(summ.model == m) & (summ.capture == c) & (summ.variant == "raw_sc") & (summ.split == "test")].iloc[0]
        choice[f"{m}/{c}"] = dict(chosen=best, test_flip=t.flip_mean, test_score=t.score_mean,
                                  raw_sc_test_flip=r.flip_mean, raw_sc_test_score=r.score_mean)

    takes.to_csv(out / "takes.csv", index=False)
    j.to_parquet(out / "cycles.parquet") if _has_parquet() else j.to_csv(out / "cycles.csv.gz", index=False)
    summ.to_csv(out / "summary.csv", index=False)
    (out / "selection.json").write_text(json.dumps(dict(choice=choice, ref_rms=ref_rms, captures=captures, stems=stems,
                                                        test_patients=sorted(TEST_PATIENTS)), indent=1, default=float))
    for c in captures:  # processed audio for listening
        for v, ws in waves[c].items():
            d = out / "audio" / c / v
            d.mkdir(parents=True, exist_ok=True)
            import soundfile as sf
            for s, w in ws.items():
                sf.write(d / f"{s}.wav", np.clip(w, -1, 1), SR, subtype="PCM_16")
    print(json.dumps(choice, indent=1, default=float))


def _has_parquet():
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
