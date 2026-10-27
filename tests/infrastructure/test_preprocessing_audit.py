"""Synthetic preprocessing proof and strict official-source audit boundaries."""
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from scripts.preprocessing_audit import audit, energy, prepare_variant, recording_windows, source_rows, validation_rows
from src.processing.cycles import cut_pad, individual_cycles


@pytest.mark.parametrize("sr", [4000, 16000, 44100])
def test_resample_length_amplitude_and_fade(sr):
    wave = torch.from_numpy((.5*np.sin(2*np.pi*200*np.arange(sr)/sr)).astype(np.float32))[None]
    plain = prepare_variant(wave, sr, fade=False)
    faded = prepare_variant(wave, sr)
    scipy = prepare_variant(wave, sr, "scipy", fade=False)
    assert plain.shape == faded.shape == scipy.shape == (1, 16000)
    assert .34 < float(plain.square().mean().sqrt()) < .36
    assert float(plain.abs().max()) < .55
    assert energy(faded) < energy(plain)
    assert faded[0, 0] == faded[0, -1] == 0
    torch.testing.assert_close(faded[:, 1000:-1000], plain[:, 1000:-1000])
    assert float((plain-scipy).square().mean().sqrt()) < .005


def test_crop_repeat_and_registered_window_samples():
    source = torch.arange(9*16000).float()[None] / (9*16000)
    torch.testing.assert_close(cut_pad(source, 5, 16000, "repeat"), source[:, :80000])
    windows = recording_windows(source, 8)
    assert len(windows) == 1
    assert all(len(w) == 128000 for w in windows)
    short = torch.ones(1, 32000)
    five, eight = (cut_pad(short, seconds, 16000, "repeat") for seconds in (5, 8))
    assert five.shape[-1] == 80000 and eight.shape[-1] == 128000
    assert five[0, -1] == eight[0, -1] == 0
    assert energy(eight) > energy(five) > energy(short)


def dataset(tmp_path):
    audio = tmp_path / "audio_test_data"
    audio.mkdir()
    stems = ["101_1b1_Al_sc_Meditron", "102_1b1_Al_sc_Meditron", "103_1b1_Al_sc_Meditron"]
    (tmp_path / "official_split.txt").write_text("\n".join(stem + "\ttrain" for stem in stems[:2]) + "\n" + stems[2] + "\ttest\n")
    for stem in stems:
        sf.write(audio / (stem+".wav"), np.ones(8000, dtype=np.float32)*.25, 4000, subtype="FLOAT")
        (audio / (stem+".txt")).write_text("0.25\t1.5\t0\t1\n" if stem != stems[2] else "do not read official-test annotations")
    return audio, stems


def test_source_only_metadata_and_cycle_effects(tmp_path):
    dataset(tmp_path)
    rows = source_rows(tmp_path)
    assert len(rows) == 2 and {r["patient"] for r in rows} == {"101", "102"}
    result = audit(tmp_path, 1)
    assert result["source_cycles"] == result["source_recordings"] == 2
    assert result["native_sample_rates"] == {"4000": 2}
    effect = result["effects"][0]
    assert effect["native_cycle_samples"] == 5000
    assert effect["resampled_cycle_samples"] == 20000
    assert effect["samples_5"] == 80000 and effect["samples_8"] == 128000
    assert effect["energy_8"] > effect["energy_5"] > effect["energy_raw"]


def test_native_annotation_bounds_and_label_independent_audio(tmp_path):
    import pandas as pd
    audio, stems = dataset(tmp_path)
    args = SimpleNamespace(sample_rate=16000, desired_length=5, pad_types="repeat", n_cls=4)
    annotations = pd.DataFrame([[.25, 20., 0, 0]], columns=["Start", "End", "Crackles", "Wheezes"])
    first = individual_cycles(annotations, audio, stems[0], args, raw=True)
    annotations.loc[0, ["Crackles", "Wheezes"]] = [1, 1]
    second = individual_cycles(annotations, audio, stems[0], args, raw=True)
    assert first[0][0].shape[-1] == 28000
    torch.testing.assert_close(first[0][0], second[0][0])
    assert first[0][1] == 0 and second[0][1] == 3


def test_validation_uses_same_patient_fold_and_rejects_target_unit(tmp_path):
    dataset(tmp_path)
    rows = source_rows(tmp_path)
    args = SimpleNamespace(selection="cv", cv_folds=2, cv_fold=0, test_fold="official", holdout_device="")
    selected = validation_rows(rows, args)
    assert len(selected) == 1
    assert selected[0]["patient"] in {"101", "102"}
    args.selection = "fixed"
    with pytest.raises(ValueError, match="patient-CV"):
        validation_rows(rows, args)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cli_device_forwarding_and_runtime_provenance(tmp_path, monkeypatch, device):
    import sys
    import scripts.preprocessing_audit as script
    dataset(tmp_path)
    seen = []
    monkeypatch.setattr(script, "compare_unit", lambda unit, device="cpu": seen.append(device) or
                        {"status": "complete", "checkpoint_sha256": "frozen-model"})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(sys, "argv", ["audit", "--data-folder", str(tmp_path), "--out", str(tmp_path / "out"),
                                    "--unit-dir", str(tmp_path), "--device", device])
    script.main()
    import json
    import os
    runtime = json.loads((tmp_path / "out/summary.json").read_text())["audit_runtime"]
    assert seen == [device]
    assert runtime["device"] == device and runtime["frozen_model_sha256"] == "frozen-model"
    assert len(runtime["script_sha256"]) == 64 and runtime["wall_seconds"] > 0
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ("" if device == "cpu" else "0")


def test_cuda_request_fails_without_gpu(monkeypatch):
    import sys
    import scripts.preprocessing_audit as script
    monkeypatch.setattr(sys, "argv", ["audit", "--device", "cuda"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit) as error:
        script.main()
    assert error.value.code == 2
