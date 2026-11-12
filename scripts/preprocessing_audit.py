"""Source-only preprocessing audit; optional frozen source-CV recording comparison.

Run with python -m scripts.preprocessing_audit; CPU is the default device.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from math import gcd

from auscult_trust.paths import DATA, REPO
from auscult_trust.processing.cycles import cut_pad, prepare_recording
from auscult_trust.processing.icbhi import DEVICES


def source_rows(data_folder):
    """Read annotations only for official source-training recordings."""
    data_folder = Path(data_folder)
    split = dict(line.split() for line in (data_folder / "official_split.txt").read_text().splitlines())
    rows = []
    for stem in sorted(stem for stem, side in split.items() if side == "train"):
        path = data_folder / "audio_test_data" / (stem + ".wav")
        annotation = np.loadtxt(path.with_suffix(".txt"), ndmin=2)
        for start, end, crackle, wheeze in annotation:
            rows.append(dict(stem=stem, wav=str(path), patient=stem.split("_")[0],
                             device=DEVICES.index(stem.split("_")[-1]), start=float(start),
                             end=float(end), label=int(crackle) + 2 * int(wheeze)))
    return rows


def prepare_variant(data, sr, resampler="torchaudio", fade=True):
    """Hold decode/amplitude fixed while varying resampler or recording fade."""
    if resampler == "torchaudio":
        if fade:
            return prepare_recording(data, sr, 16000)
        from torchaudio.transforms import Resample
        return Resample(sr, 16000)(data) if sr != 16000 else data.clone()
    if resampler != "scipy":
        raise ValueError("unknown resampler")
    divisor = gcd(sr, 16000)
    wave = torch.from_numpy(resample_poly(data.numpy(), 16000 // divisor, sr // divisor, axis=-1).copy())
    return prepare_recording(wave, 16000, 16000) if fade else wave


def recording_windows(wave, seconds):
    """Registered ICBHI linspace coverage with only duration varied."""
    duration = wave.shape[-1] / 16000
    starts = np.linspace(0, max(duration - seconds, 0), max(1, int(duration // seconds)))
    return [cut_pad(wave[:, int(round(start * 16000)):int(round(start * 16000)) + seconds * 16000],
                    seconds, 16000, "repeat")[0].numpy() for start in starts]


def energy(wave):
    return float(np.square(np.asarray(wave, dtype=np.float64)).sum())


def audit(data_folder, max_recordings=16):
    rows = source_rows(data_folder)
    stems = sorted(set(row["stem"] for row in rows))
    effects, rates = [], {}
    for stem in stems:
        subset = [row for row in rows if row["stem"] == stem]
        info = sf.info(subset[0]["wav"])
        rates[str(info.samplerate)] = rates.get(str(info.samplerate), 0) + 1
        if stem not in stems[:max_recordings]:
            continue
        decoded, sr = sf.read(subset[0]["wav"], dtype="float32", always_2d=True)
        native = torch.from_numpy(decoded.T.copy())
        prepared = prepare_variant(native, sr)
        unfaded = prepare_variant(native, sr, fade=False)
        scipy = prepare_variant(native, sr, "scipy", fade=False)
        for row in subset:
            start, end = min(int(row["start"] * 16000), prepared.shape[-1]), min(int(row["end"] * 16000), prepared.shape[-1])
            cycle = prepared[:, start:end]
            if not cycle.shape[-1]:
                raise ValueError(f"empty annotated source cycle: {stem} {row['start']}:{row['end']}")
            five, eight = (cut_pad(cycle, seconds, 16000, "repeat") for seconds in (5, 8))
            effects.append(dict(stem=stem, start=row["start"], end=row["end"], native_sample_rate=sr,
                                native_cycle_samples=min(int(row["end"]*sr), native.shape[-1])-min(int(row["start"]*sr), native.shape[-1]),
                                resampled_cycle_samples=end-start,
                                samples_5=int(five.shape[-1]), samples_8=int(eight.shape[-1]),
                                energy_raw=energy(cycle), energy_5=energy(five), energy_8=energy(eight),
                                recording_fade_energy_ratio=energy(prepared) / max(energy(unfaded), 1e-30),
                                resampler_rms_difference=float(torch.sqrt((unfaded-scipy).square().mean()))))
    durations = np.array([row["end"]-row["start"] for row in rows])
    return dict(source_recordings=len(stems), source_patients=len(set(row["patient"] for row in rows)),
                source_cycles=len(rows), native_sample_rates=rates,
                cycle_duration_seconds=dict(min=float(durations.min()), median=float(np.median(durations)), max=float(durations.max())),
                cropped_at_5=int((durations > 5).sum()), cropped_at_8=int((durations > 8).sum()),
                measured_recordings=min(max_recordings, len(stems)), measured_cycles=len(effects), effects=effects)


def validation_rows(rows, args):
    from auscult_trust.processing.splits import cv_split
    if args.selection != "cv" or not args.cv_folds or args.test_fold != "official" or args.holdout_device:
        raise ValueError("requires an official-source patient-CV unit")
    train, val = cv_split(np.array([row["patient"] for row in rows]), np.array([row["device"] for row in rows]), args.cv_folds, args.cv_fold)
    assert not {rows[i]["patient"] for i in train} & {rows[i]["patient"] for i in val}
    return [rows[i] for i in val]


def compare_unit(unit, device="cpu"):
    from auscult_trust.training.config import load_args
    from auscult_trust.inference.model import load_frozen, images_of, predict_images
    from auscult_trust.processing.correction import apply_spectrum_correction
    from auscult_trust.evaluation.metrics import sp_se
    from auscult_trust.artifacts import sha256_file
    from auscult_trust.runs import twin_dir
    args = load_args(unit)
    rows = validation_rows(source_rows(args.data_folder), args)
    checkpoint = twin_dir(unit, "checkpoints") / "best.pth"
    if not checkpoint.is_file():
        return {"status": "pending", "reason": "source-CV best.pth not available", "checkpoint": str(checkpoint)}
    model, classifier = load_frozen(args, checkpoint, device=device)
    coefficients = None
    state_path = Path(unit) / "sc_state.npz"
    if args.spectrum_correction:
        with np.load(state_path) as state:
            coefficients = {d: state[f"validation_coefficients_{d}"].copy() for d in {row["device"] for row in rows}}
    variants = [("registered8", 8, "torchaudio", True), ("duration5", 5, "torchaudio", True),
                ("scipy8", 8, "scipy", True), ("no_recording_fade8", 8, "torchaudio", False)]
    results = {}
    stems = sorted(set(row["stem"] for row in rows))
    for name, seconds, resampler, fade in variants:
        probabilities, labels = [], []
        for stem in stems:
            subset = [row for row in rows if row["stem"] == stem]
            decoded, sr = sf.read(subset[0]["wav"], dtype="float32", always_2d=True)
            wave = prepare_variant(torch.from_numpy(decoded.T.copy()), sr, resampler, fade)
            waves = recording_windows(wave, seconds)
            if coefficients is not None:
                waves = [apply_spectrum_correction(w, coefficients[subset[0]["device"]]).astype(np.float32) for w in waves]
            probs = predict_images(images_of(waves, args), model, classifier, method=args.method)
            probabilities.append(float((1-probs[:, 0]).mean()))
            labels.append(int(any(row["label"] > 0 for row in subset)))
        sp, se = sp_se(np.asarray(labels), np.asarray(probabilities) >= .5)
        results[name] = dict(seconds=seconds, resampler=resampler, recording_fade=fade,
                             specificity=float(sp), sensitivity=float(se), binary_recording_score=float((sp+se)/2),
                             abnormal_probability=probabilities)
    return dict(status="complete", checkpoint=str(checkpoint), checkpoint_sha256=sha256_file(checkpoint),
                patients=sorted(set(row["patient"] for row in rows)), stems=stems,
                labels=labels, sc="fixed saved validation coefficients" if coefficients is not None else "none",
                sc_state_sha256=sha256_file(state_path) if coefficients is not None else None, variants=results)


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data-folder", type=Path, default=DATA / "processed/patchmix_icbhi")
    parser.add_argument("--max-recordings", type=int, default=16)
    parser.add_argument("--unit-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--out", type=Path, default=Path("outputs/evaluation/improvements-20261026/preprocessing"))
    cli = parser.parse_args()
    if cli.max_recordings < 1:
        parser.error("--max-recordings must be positive")
    if cli.device == "cpu":
        # Preprocess helpers auto-select CUDA, so CPU runs also hide it.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif not torch.cuda.is_available():
        parser.error("--device cuda requires an available CUDA GPU")
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    summary = audit(cli.data_folder, cli.max_recordings)
    traced = ["experiments/train/grid/pafa_beats.yaml", "auscult_trust/processing/cycles.py", "auscult_trust/processing/audio.py",
              "auscult_trust/pipelines/screening_infer.py", "auscult_trust/pipelines/kauh_infer.py"]
    summary["source_sha256"] = {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in traced}
    summary["training_recipe_yaml"] = (REPO / traced[0]).read_text()
    effects = summary["effects"]
    summary["measured_effects"] = dict(
        mean_energy_8_over_5=float(np.mean([row["energy_8"] / max(row["energy_5"], 1e-30) for row in effects])),
        median_recording_fade_energy_ratio=float(np.median([row["recording_fade_energy_ratio"] for row in effects])),
        max_resampler_rms_difference=max(row["resampler_rms_difference"] for row in effects))
    summary["trace"] = dict(training="pafa_beats.yaml: 5s annotated cycles; torchaudio resample to 16kHz, 62.5ms recording fade; first crop or repeat+62.5ms fade-out",
        screening="registered 8s ICBHI recording windows: same resample/fade, linspace starts, repeat padding",
        kauh="registered 8s windows: scipy resample_poly to 16kHz, start-aligned zero padding, no fade",
        normalization="soundfile float32 PCM for training/screening; scipy WAV integer/max PCM for KAUH; no per-wave peak normalization")
    summary["frozen_source_comparison"] = compare_unit(cli.unit_dir, device=cli.device) if cli.unit_dir else {"status": "pending", "reason": "supply source-CV --unit-dir with best.pth; no target evaluation used"}
    if cli.device == "cuda":
        torch.cuda.synchronize()
    summary["audit_runtime"] = dict(device=cli.device, torch_version=torch.__version__,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        wall_seconds=time.perf_counter()-started,
        frozen_model_sha256=summary["frozen_source_comparison"].get("checkpoint_sha256"))
    cli.out.mkdir(parents=True, exist_ok=True)
    (cli.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    report = ["# Source preprocessing audit", "", *[f"- {key}: {value}" for key, value in summary["trace"].items()], "",
              f"Official source only: {summary['source_recordings']} recordings, {summary['source_patients']} patients, {summary['source_cycles']} cycles.",
              f"Native sample rates: {summary['native_sample_rates']}. Cycles longer than 5s: {summary['cropped_at_5']}; longer than 8s: {summary['cropped_at_8']}.",
              f"Decoded effects measured on first {summary['measured_recordings']} sorted source recordings ({summary['measured_cycles']} cycles); per-cycle energies and resampler RMS are in summary.json.", "",
              f"Measured effects: {summary['measured_effects']}. Energy sums reflect padding/repetition as well as duration; these are waveform effects, not performance estimates.",
              "Duration comparison changes the registered linspace window schedule and repeat-padding budget together. Resampler/fade contrasts use the same 8s schedule. Saved validation SC curves stay fixed across variants.",
              "The no-recording-fade contrast retains the separate repeat-padding fade-out. Configuration text and traced source SHA-256 hashes are saved in summary.json.",
              "Registered 8s screening/KAUH remain unchanged. Source recording label is any annotated abnormal cycle; probabilities are mean window p(abnormal), threshold 0.5.",
              "Frozen source comparison: " + summary["frozen_source_comparison"]["status"] + ". No official-test or KAUH labels used."]
    (cli.out / "report.md").write_text("\n\n".join(report)+"\n")
    print(cli.out / "summary.json")


if __name__ == "__main__":
    main()
