"""Test-time spectrum correction, moment matching, and fixed-seed gain views."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from src.processing.correction import apply_spectrum_correction, mean_spectrum, spectrum_coefficients, isa_match
from src.artifacts import atomic_npz
from src.provenance import prediction_inputs, record_prediction, verify_prediction

TTA_SEED0, TTA_DB, TTA_KNOTS = 1000, 6.0, 6


def test_time_sc(waves, filters, sc_state, args):
    """Spectrum-correct every window towards the saved reference, one coefficient curve per KAUH filter."""
    ref = sc_state["reference"]
    spectra = {
        f: mean_spectrum([w for w, g in zip(waves, filters) if g == f]) for f in sorted(set(filters))
    }
    coef = spectrum_coefficients(
        spectra,
        reference=args.sc_reference,
        sc_mode=args.sc_mode,
        limit_freq_low=args.sc_limit_freq_low,
        limit_freq_high=args.sc_limit_freq_high,
        limit_freq_diff=args.sc_limit_freq_diff,
        reference_spectra={"ref": ref},
    )
    return [apply_spectrum_correction(w, coef[g]).astype(np.float32) for w, g in zip(waves, filters)]


def gain_curves(n_total, n_freq, sd, seed, knots=TTA_KNOTS):
    """One smooth random per-bin gain curve per cycle index (same draw as `random_bin_gain`), shape (n_total, n_freq)."""
    rng = np.random.default_rng(seed)
    grid, kn = np.linspace(0, 1, n_freq), np.linspace(0, 1, knots)
    return np.stack([np.interp(grid, kn, rng.normal(0, sd, knots)) for _ in range(n_total)]).astype(np.float32)


def a2_images(images_test, images_source):
    """A2 on (n, 1, time, freq) images: per-freq-bin moments of the test images mapped onto the source images'."""
    def stats(x):
        flat = x[:, 0].reshape(-1, x.shape[-1]).double()
        return flat.mean(0).numpy(), flat.std(0).numpy() + 1e-6

    t, s = stats(images_test), stats(images_source)
    out = isa_match(images_test[:, 0].numpy(), t, s)
    return torch.from_numpy(out.astype(np.float32))[:, None]


def source_stats(unit, args):
    """Per-freq-bin mean and SD of the training images of the unit's recipe (A2 source statistics), cached on disk."""
    effective = {**vars(args), "holdout_device": None, "cv_folds": 0}
    context = dict(unit=unit, checkpoint=None, dataset="cycles",
                   options={"source_statistics": json.loads(json.dumps(effective, default=str))})
    inputs = prediction_inputs(**context)
    f = Path(unit) / "inference/source_stats.npz"
    if f.exists():
        verify_prediction(f, inputs)
        z = np.load(f)
        return z["mean"], z["sd"]
    from src.training.dataset import ICBHIDataset

    a = argparse.Namespace(**effective)
    ds = ICBHIDataset(True, None, a, print_flag=False)
    acc, acc2, n = 0.0, 0.0, 0
    for i in range(0, len(ds.images), 256):
        x = np.asarray(ds.images[i : i + 256], dtype=np.float64)[..., 0]
        acc, acc2, n = acc + x.sum((0, 1)), acc2 + (x ** 2).sum((0, 1)), n + x.shape[0] * x.shape[1]
    mean = acc / n
    sd = np.sqrt(acc2 / n - mean ** 2) + 1e-6
    f.parent.mkdir(parents=True, exist_ok=True)
    atomic_npz(f, mean=mean, sd=sd)
    record_prediction(f, inputs, **context)
    return mean, sd
