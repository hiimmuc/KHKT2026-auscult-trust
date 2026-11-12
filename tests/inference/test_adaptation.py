"""A2 statistics must belong to one unit and its exact preprocessing inputs."""
from types import SimpleNamespace

import numpy as np
import pytest

from auscult_trust import provenance
from auscult_trust.inference import adaptation


def test_source_statistics_are_unit_scoped_and_reject_stale_inputs(tmp_path, monkeypatch):
    from auscult_trust.training import dataset
    images = np.arange(48, dtype=np.float32).reshape(2, 4, 6, 1)
    calls = []
    def build_dataset(train, transform, args, print_flag):
        assert train and args.holdout_device is None and args.cv_folds == 0
        calls.append(args.seed)
        return SimpleNamespace(images=images)
    def inputs(**context):
        return {"unit": str(context["unit"]), "options": context["options"]}
    monkeypatch.setattr(dataset, "ICBHIDataset", build_dataset)
    monkeypatch.setattr(adaptation, "prediction_inputs", inputs)
    monkeypatch.setattr(provenance, "prediction_inputs", inputs)
    args = SimpleNamespace(seed=0, holdout_device="Litt3200", cv_folds=5)
    a, b = tmp_path / "run-a/cell/seed0", tmp_path / "run-b/cell/seed0"
    mean, sd = adaptation.source_stats(a, args)
    np.testing.assert_allclose(mean, images[..., 0].mean((0, 1)))
    np.testing.assert_allclose(sd, images[..., 0].std((0, 1)) + 1e-6)
    adaptation.source_stats(a, args)
    assert calls == [0]
    adaptation.source_stats(b, args)
    assert calls == [0, 0]
    assert (a / "inference/source_stats.npz.provenance.json").exists()
    args.seed = 1
    with pytest.raises(ValueError, match="Stale"):
        adaptation.source_stats(a, args)
    assert calls == [0, 0]
