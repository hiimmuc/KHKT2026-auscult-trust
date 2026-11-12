import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from auscult_trust.artifacts import atomic_json
from auscult_trust.training.state import capture_rng, restore_rng, completed, publish_completion, report_epoch


def test_rng_resume_reproduces_next_draws():
    original = capture_rng()
    try:
        saved = capture_rng()
        def draw():
            return random.random(), np.random.random(4), torch.rand(4), torch.rand(4, device="cuda") if torch.cuda.is_available() else None
        expected = draw()
        draw()
        restore_rng(saved)
        actual = draw()
        assert expected[0] == actual[0]
        assert np.array_equal(expected[1], actual[1])
        assert torch.equal(expected[2], actual[2])
        if expected[3] is not None:
            assert torch.equal(expected[3], actual[3])
        with pytest.raises(ValueError, match="no RNG"):
            restore_rng(None)
    finally:
        restore_rng(original)


def test_report_epoch_sorts_numerically(tmp_path):
    atomic_json(tmp_path / "report.json", {"fixed": {"100": {}, "20": {}}})
    assert report_epoch(str(tmp_path)) == 20


def test_completion_rejects_partial_or_modified_results(tmp_path):
    out, checkpoints = tmp_path / "unit", tmp_path / "weights"
    out.mkdir()
    checkpoints.mkdir()
    assert not completed(out)
    atomic_json(out / "report.json", {"history": [{"epoch": 1}]})
    with pytest.raises(ValueError, match="Uncertified"):
        completed(out)
    (checkpoints / "last.pth").write_bytes(b"restart state")
    assert not completed(out, checkpoints)
    (checkpoints / "report_epoch_1.pth").write_bytes(b"weights")
    publish_completion(out, checkpoints)
    assert completed(out)
    (checkpoints / "report_epoch_1.pth").write_bytes(b"changed")
    with pytest.raises(ValueError, match="integrity"):
        completed(out)


def test_config_mismatch_is_checked_before_completed_skip(tmp_path, monkeypatch):
    from auscult_trust.training import main
    args = SimpleNamespace(exp="train", cell="test", unit="seed0", seed=1)
    atomic_json(tmp_path / "train_args.json", {"seed": 0})
    atomic_json(tmp_path / "report.json", {"history": [1]})
    monkeypatch.setattr(main, "parse_args", lambda: args)
    monkeypatch.setattr(main.runs, "unit_dir", lambda *a: tmp_path)
    with pytest.raises(ValueError, match="different config"):
        main.main()


def test_resumed_epoch_matches_uninterrupted_training():
    import copy
    original = capture_rng()
    try:
        torch.manual_seed(42)
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(0.3), torch.nn.Linear(4, 1))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        data = torch.arange(18, dtype=torch.float32).reshape(6, 3) / 18
        def epoch():
            for i in torch.randperm(len(data)):
                optimizer.zero_grad()
                scale = random.random() + np.random.random()
                model(data[i] * scale).square().sum().backward()
                optimizer.step()
        epoch()
        checkpoint = copy.deepcopy((model.state_dict(), optimizer.state_dict(), capture_rng()))
        epoch()
        expected = copy.deepcopy(model.state_dict())
        model.load_state_dict(checkpoint[0])
        optimizer.load_state_dict(checkpoint[1])
        restore_rng(checkpoint[2])
        epoch()
        assert all(torch.equal(value, model.state_dict()[key]) for key, value in expected.items())
    finally:
        restore_rng(original)
