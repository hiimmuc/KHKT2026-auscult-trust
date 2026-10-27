"""Frozen-model inference on KAUH 8 s windows, with optional test-time spectrum correction (SC).

    python -m src.evaluation.kauh_infer outputs/train/<run>/<cell>/seed<s> --ckpt report_epoch_<E>.pth \
        --out outputs/kauh/<run>/<name>/seed<s>.npz [--sc --sc_state outputs/train/<run>/<sc cell>/seed<s>/sc_state.npz]

No KAUH data is used for training or selection. With `--sc`, the reference spectrum `s_ref` comes from the saved SC state of an
SC training run (identical across seeds: it is the mean of the ICBHI training devices' spectra), the coefficient of each KAUH
filter from the mean spectrum of that filter's own unlabelled windows, bounded with the `sc_mode` of that run. Windows keep
the recording-level label (an [Assumption] noted in `src.processing.kauh.windows`); KAUH audio is 4 kHz, resampled to 16 kHz.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from src import runs, artifacts
from src.paths import DATA
from src.processing.audio import SR, crop_pad, load_wav
from src.processing.kauh import file_table, windows
from src.training.config import load_args
from src.training.encoders import get_encoder
from src.inference.model import load_frozen, images_of, predict_images
from src.inference.adaptation import test_time_sc, source_stats

KAUH_ROOT = DATA / "raw" / "kauh"
WINDOW_SEC = 8.0


def window_waves(win):
    """16 kHz waveform (8 s, zero-padded) of every window row of `kauh.windows`."""
    cache, out = {}, []
    for r in win.itertuples(index=False):
        if r.wav not in cache:
            cache = {r.wav: load_wav(r.wav, SR)}  # rows are grouped by recording: keep one at a time
        x = cache[r.wav]
        i = int(round(r.start * SR))
        out.append(crop_pad(x[i : i + int(WINDOW_SEC * SR)], int(WINDOW_SEC * SR)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("unit_dir", help="outputs/<exp>/<run_id>/<cell>/<unit> of the trained model (holds train_args.json)")
    ap.add_argument("--ckpt", required=True, help="weight file in the unit's checkpoints dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sc", action="store_true", help="test-time spectrum correction")
    ap.add_argument("--sc_state", help="sc_state.npz of an SC training run (reference spectrum); default: the unit's own")
    ap.add_argument("--sc_args", help="train_args.json of the SC run (coefficient bound); default: the unit's own")
    ap.add_argument("--a2", action="store_true", help="A2: per-filter image moments mapped onto the training images'")
    ap.add_argument("--tta_eq", type=int, default=0, help="TTA-EQ views (6 dB SD per-bin gain, seeds 1000 + k)")
    a = ap.parse_args()
    unit = Path(a.unit_dir)
    from src.artifacts import atomic_npz
    from src.provenance import prediction_inputs, record_prediction
    context = dict(unit=unit, checkpoint=runs.twin_dir(unit) / a.ckpt, dataset="kauh", sc=a.sc,
                   sc_unit=Path(a.sc_state).parent if a.sc_state else unit,
                   options={"a2": a.a2, "tta_eq": a.tta_eq,
                            "sc_state": a.sc_state, "sc_args": a.sc_args})
    inputs = prediction_inputs(**context)
    args = load_args(unit)
    sc_args = argparse.Namespace(**json.load(open(a.sc_args or unit / "train_args.json")))
    win = windows(file_table(KAUH_ROOT), WINDOW_SEC).reset_index(drop=True)
    waves = window_waves(win)
    if a.sc:
        state = dict(np.load(a.sc_state or unit / "sc_state.npz"))
        waves = test_time_sc(waves, list(win["filter"]), state, sc_args)
    ckpt = runs.twin_dir(a.unit_dir, "checkpoints") / a.ckpt
    model, classifier = load_frozen(args, ckpt)
    images = images_of(waves, args)
    if a.a2:
        from src.processing.correction import isa_match

        src = source_stats(unit, args)
        filters = win["filter"].to_numpy()
        for f in np.unique(filters):
            m = filters == f
            flat = images[m].reshape(-1, images.shape[-1]).astype(np.float64)
            images[m] = isa_match(images[m], (flat.mean(0), flat.std(0) + 1e-6), src).astype(np.float32)
    view_probs = None
    recording = {}
    if a.tta_eq:
        from src.inference.adaptation import TTA_DB, TTA_SEED0, gain_curves

        sd = TTA_DB * get_encoder(args.encoder).db_scale
        view_probs = np.stack([predict_images(images, model, classifier, gain_curves(len(images), images.shape[-1], sd, TTA_SEED0 + k), method=args.method)
                               for k in range(a.tta_eq)])
        probs = view_probs.mean(0)
        if args.method == "ce_recording_pafa":
            from src.evaluation.recording_head import recording_predictions
            views = [recording_predictions(images, win, model, classifier,
                         gain_curves(len(images), images.shape[-1], sd, TTA_SEED0 + k))
                     for k in range(a.tta_eq)]
            recording = views[0]
            recording["recording_view_probs"] = np.stack([v["recording_probs"] for v in views])
            recording["recording_probs"] = recording["recording_view_probs"].mean(0)
    else:
        probs = predict_images(images, model, classifier, method=args.method)
        if args.method == "ce_recording_pafa":
            from src.evaluation.recording_head import recording_predictions
            recording = recording_predictions(images, win, model, classifier)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_npz(
        out,
        probs=probs,
        **recording,
        label=win["label"].to_numpy(),
        patient=win["patient"].to_numpy(),
        stem=win["stem"].to_numpy(),
        filter=win["filter"].to_numpy(),
        window=win["window"].to_numpy(),
        sc=np.array(a.sc),
        a2=np.array(a.a2),
        tta_eq=np.array(a.tta_eq),
        **({"view_probs": view_probs.astype(np.float16)} if view_probs is not None else {}),
        ckpt_sha256=np.array(artifacts.sha256_file(ckpt)),
    )
    record_prediction(out, inputs, **context)
    print(out, probs.shape, "sc" if a.sc else "no sc", "a2" if a.a2 else "", "tta{}".format(a.tta_eq) if a.tta_eq else "")


if __name__ == "__main__":
    main()
