"""QLung's quality score must stay finite for silence and reuse its completed cache."""
from types import SimpleNamespace

import numpy as np
import torch

from src.training.config import parse_args
from src.training.methods import qlung


def test_silent_quality_fallback_and_cache(tmp_path, monkeypatch):
    from src.processing import cycles
    dataset = SimpleNamespace(args=parse_args(["--method", "qlung"]), split="train", filenames=["silent"], labels=[0])
    reference = SimpleNamespace(compute_spectral_entropy=lambda _: float("nan"), compute_rms_energy=lambda _: 0,
                                compute_quality_score_entropy_rms=lambda e, r, **kw: e)
    monkeypatch.setattr(qlung, "load_reference", lambda *a: reference)
    monkeypatch.setattr(cycles, "individual_cycles", lambda *a: [(torch.zeros(1, 4096), 0)])
    method = qlung.Method()
    quality = method.quality_scores(dataset, {"silent": []}, tmp_path, tmp_path)
    assert np.array_equal(quality, [1.0])
    def unexpected(*args):
        raise AssertionError("recomputed a completed cache")
    monkeypatch.setattr(cycles, "individual_cycles", unexpected)
    assert np.array_equal(method.quality_scores(dataset, {}, tmp_path, tmp_path), quality)
