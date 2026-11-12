"""ICBHI screening windows through a frozen model: 8 s windows of the official-test recordings and their softmax.

    python -m src.pipelines.screening_infer outputs/train/screen-ast/<cell>/seed<s> --ckpt report_epoch_<E>.pth \
        --out outputs/screening/<condition>/seed<s>.npz [--sc]

Every official-test recording (without the 2 patients on both official sides) is cut into max(1, floor(duration / 8 s)) windows
evenly spread, as for KAUH; waveform loading as in training (soundfile, torchaudio resampling to 16 kHz, linear fade, repeat
padding). Recording label: abnormal if any annotated cycle has a crackle or a wheeze. With `--sc` (models trained with SC) every
window is corrected with the saved test coefficient of its device (`sc_state.npz`, estimated from the unlabelled official-test
cycles, as in the registered test pipeline).
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from src.inference.model import predict
from src.processing.correction import apply_spectrum_correction
from src.processing.cycles import cut_pad, prepare_recording, read_annotations
from src.processing.icbhi import DEVICES
from src.processing.splits import both_sides_patients
from src.training.config import load_args

SR, WIN = 16000, 8.0


def icbhi_windows(args):
    """Waveforms and table (stem, patient, device, label, window) of the official-test recordings' 8 s windows."""
    wav_dir = os.path.join(args.data_folder, "audio_test_data")
    official = dict(line.strip().split("\t") for line in open(os.path.join(args.data_folder, "official_split.txt")).read().splitlines())
    ann = read_annotations(wav_dir)
    both = both_sides_patients(Path(args.data_folder) / "official_split.txt")
    waves, rows = [], []
    for stem in sorted(s for s, side in official.items() if side == "test" and s.split("_")[0] not in both):
        x, sr = sf.read(os.path.join(wav_dir, stem + ".wav"), dtype="float32", always_2d=True)
        x = torch.from_numpy(x.T.copy())
        x = prepare_recording(x, sr, SR)
        dur = x.shape[1] / SR
        a = ann[stem]
        label = int(((a["Crackles"] > 0) | (a["Wheezes"] > 0)).any())
        for w, s in enumerate(np.linspace(0, max(dur - WIN, 0), max(1, int(dur // WIN)))):
            i = int(round(s * SR))
            waves.append(cut_pad(x[:, i : i + int(WIN * SR)], int(WIN), SR, "repeat")[0].numpy())
            rows.append({"stem": stem, "patient": stem.split("_")[0], "device": DEVICES.index(stem.split("_")[-1]),
                         "label": label, "window": w})
    return waves, pd.DataFrame(rows)


def infer(a):
    from src.artifacts import atomic_npz
    from src.provenance import prediction_inputs, record_prediction
    from src.runs import twin_dir
    unit = Path(a.unit_dir)
    context = dict(unit=unit, checkpoint=twin_dir(unit) / a.ckpt, dataset="screening", sc=a.sc)
    inputs = prediction_inputs(**context)
    args = load_args(unit)
    waves, win = icbhi_windows(args)
    if a.sc:
        st = np.load(unit / "sc_state.npz")
        coef = {d: st["test_coefficients_{}".format(d)] for d in win.device.unique()}
        waves = [apply_spectrum_correction(w, coef[d]).astype(np.float32) for w, d in zip(waves, win.device)]
    from src import runs, artifacts

    ckpt = runs.twin_dir(unit, "checkpoints") / a.ckpt
    recording = {}
    if args.method == "ce_recording_pafa":
        from src.inference.model import images_of, load_frozen, predict_images
        from src.pipelines.recording_head import recording_predictions
        model, classifier = load_frozen(args, ckpt)
        images = images_of(waves, args)
        probs = predict_images(images, model, classifier, method=args.method)
        recording = recording_predictions(images, win, model, classifier)
    else:
        probs = predict(waves, args, ckpt)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_npz(out, probs=probs, **{k: win[k].to_numpy() for k in win.columns}, **recording, sc=np.array(a.sc),
                        ckpt_sha256=np.array(artifacts.sha256_file(ckpt)))
    record_prediction(out, inputs, **context)
    print(out, probs.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("unit_dir")
    ap.add_argument("--ckpt")
    ap.add_argument("--out")
    ap.add_argument("--sc", action="store_true")
    infer(ap.parse_args())


if __name__ == "__main__":
    main()
