from auscult_trust import provenance
import json

import pytest

from auscult_trust import artifacts


def test_artifact_primitives_do_not_import_run_or_model_modules():
    import subprocess
    import sys
    subprocess.run([sys.executable, "-c", "import sys; import auscult_trust.artifacts; "
                    "assert not {'auscult_trust.runs', 'auscult_trust.provenance', 'torch'} & sys.modules.keys()"], check=True)


def test_atomic_failure_preserves_previous_artifact(tmp_path):
    path = tmp_path / "report.json"
    artifacts.atomic_json(path, {"old": True})
    with pytest.raises(RuntimeError):
        with artifacts.atomic_path(path) as temporary:
            temporary.write_text("partial")
            raise RuntimeError("interrupted")
    assert json.loads(path.read_text()) == {"old": True}
    assert list(tmp_path.iterdir()) == [path]


def test_prediction_certificate_rejects_stale_inputs_and_modified_output(tmp_path):
    output = tmp_path / "predictions.npz"
    output.write_bytes(b"prediction")
    inputs = {"checkpoint": "a", "data": "b", "sc": "c", "code": "d"}
    with pytest.raises(ValueError, match="Missing provenance"):
        provenance.verify_prediction(output, inputs)
    provenance.record_prediction(output, inputs)
    provenance.verify_prediction(output, inputs)
    for key in inputs:
        with pytest.raises(ValueError, match="Stale"):
            provenance.verify_prediction(output, {**inputs, key: "changed"})
    output.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        provenance.verify_prediction(output, inputs)


def test_fingerprint_tracks_data_sc_code_and_checkpoint(tmp_path, monkeypatch):
    for name in ("repo/auscult_trust", "refs", "data", "audio/audio_test_data", "unit"):
        (tmp_path / name).mkdir(parents=True)
    monkeypatch.setattr(provenance, "REPO", tmp_path / "repo")
    monkeypatch.setattr(provenance, "THIRD_PARTY", tmp_path / "refs")
    monkeypatch.setattr(provenance, "DATA", tmp_path / "data")
    monkeypatch.setattr(provenance, "environment", lambda: {"packages": {"numpy": "fixed"}})
    unit, checkpoint = tmp_path / "unit", tmp_path / "model.pth"
    artifacts.atomic_json(unit / "train_args.json", {"data_folder": str(tmp_path / "audio"), "encoder": "beats"})
    files = [checkpoint, unit / "sc_state.npz", tmp_path / "audio/official_split.txt",
             tmp_path / "audio/audio_test_data/a.wav", tmp_path / "repo/auscult_trust/preprocess.py"]
    for path in files:
        path.write_bytes(b"original")
    original = provenance.prediction_inputs(unit, checkpoint, "screening", True)
    for path in files:
        path.write_bytes(b"changed")
        assert provenance.prediction_inputs(unit, checkpoint, "screening", True) != original
        path.write_bytes(b"original")


def test_pipeline_rejects_modified_weights_before_model_load(tmp_path, monkeypatch):
    from api import engine
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"original")
    expected = artifacts.sha256_file(checkpoint)
    checkpoint.write_bytes(b"changed")
    monkeypatch.setattr(engine, "load_frozen", lambda *a: pytest.fail("loaded unverified model"))
    with pytest.raises(ValueError, match="integrity"):
        engine.Pipeline("test", {"checkpoint": checkpoint.name, "checkpoint_sha256": expected, "runtime": {"file": "runtime.json"}}, tmp_path)


def test_legacy_bundle_is_rejected(tmp_path):
    from api.engine import Engine
    with pytest.raises(ValueError, match="Legacy"):
        Engine({"pipelines": {}}, tmp_path)


@pytest.mark.parametrize("target", ["arguments", "correction"])
def test_pipeline_checks_arguments_and_correction_before_load(tmp_path, monkeypatch, target):
    from api import engine
    checkpoint, arguments, correction = [tmp_path / name for name in ("model.pth", "train_args.json", "sc.npz")]
    for path in (checkpoint, arguments, correction):
        path.write_bytes(b"original")
    entry = {"checkpoint": checkpoint.name, "checkpoint_sha256": artifacts.sha256_file(checkpoint),
             "runtime": {"file": arguments.name, "sha256": artifacts.sha256_file(arguments)}, "spectrum_correction": True,
             "sc": {"state": correction.name, "sha256": artifacts.sha256_file(correction)}}
    (arguments if target == "arguments" else correction).write_bytes(b"changed")
    monkeypatch.setattr(engine, "load_frozen", lambda *a: pytest.fail("loaded unverified inputs"))
    with pytest.raises(ValueError, match="integrity"):
        engine.Pipeline("test", entry, tmp_path)


def test_atomic_npz_and_immutable_freeze(tmp_path):
    import numpy as np
    path = tmp_path / "predictions.npz"
    artifacts.atomic_npz(path, probs=np.array([[0.2, 0.8]]))
    with np.load(path) as saved:
        assert np.array_equal(saved["probs"], [[0.2, 0.8]])
    frozen = tmp_path / "freeze.json"
    artifacts.immutable_json(frozen, {"data": "a"})
    artifacts.immutable_json(frozen, {"data": "a"})
    with pytest.raises(ValueError, match="Frozen record changed"):
        artifacts.immutable_json(frozen, {"data": "b"})
    assert json.loads(frozen.read_text()) == {"data": "a"}
