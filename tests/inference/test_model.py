"""Frozen inference contracts without datasets, pretrained weights or a GPU."""
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from src.inference.model import load_frozen, predict_images
from src.training.config import parse_args
from src.training.methods import get_method
from src.models.factory import build_inference_model


class Encoder(nn.Module):
    final_feat_dim = 8

    def __init__(self):
        super().__init__()
        self.features = nn.Linear(1, self.final_feat_dim)
        self.mlp_head = nn.Linear(self.final_feat_dim, 4)

    def forward(self, x):
        return self.features(x.mean((2, 3)))


@pytest.mark.parametrize("method", ["ce", "ce_pafa", "pcmcl", "qlung"])
def test_frozen_checkpoint_matches_direct_prediction_without_training_components(tmp_path, monkeypatch, method):
    from src.models import factory
    from src.training import modeling
    args = parse_args(["--method", method])
    spec = SimpleNamespace(build=lambda _: Encoder(), image_shape=(16, 32))
    monkeypatch.setattr(factory, "get_encoder", lambda _: spec)
    def forbidden(*args, **kwargs):
        pytest.fail("Frozen inference constructed training-only components")
    monkeypatch.setattr(type(get_method(method)), "projector", forbidden)
    monkeypatch.setattr(type(get_method(method)), "losses", forbidden)
    monkeypatch.setattr(modeling, "set_optimizer", forbidden)
    original, head = build_inference_model(args, device="cpu")
    path = tmp_path / "model.pth"
    torch.save({"model": original.state_dict(), "classifier": head.state_dict()}, path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model, classifier = load_frozen(args, path)
    assert not model.training and not classifier.training
    assert next(model.parameters()).device.type == "cpu"
    images = np.random.default_rng(7).normal(size=(35, 16, 32)).astype(np.float32)
    curves = np.random.default_rng(8).normal(size=(35, 32)).astype(np.float32)
    with torch.no_grad():
        x = torch.from_numpy(images[:, None] + curves[:, None, None])
        expected = torch.cat([get_method(method).probabilities(head(original(batch)))
                              for batch in x.split(32)]).numpy()
    actual = predict_images(images, model, classifier, curves=curves, method=method)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    assert actual.shape == (35, 4)


def test_api_import_does_not_patch_torch_or_import_experiment_workflows():
    subprocess.run([sys.executable, "-c", """
import sys
import torch
torch.cuda.is_available = lambda: False
tensor_cuda, module_cuda = torch.Tensor.cuda, torch.nn.Module.cuda
import api.engine
assert torch.Tensor.cuda is tensor_cuda
assert torch.nn.Module.cuda is module_cuda
assert not {'src.pipelines.kauh_infer', 'src.pipelines.lodo_ttsc',
            'src.evaluation.screening'} & sys.modules.keys()
"""], check=True)
