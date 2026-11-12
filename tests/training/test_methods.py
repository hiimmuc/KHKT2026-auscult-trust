"""Method contracts on synthetic batches; no datasets or pretrained checkpoints."""
from types import SimpleNamespace
import subprocess
import sys

import pytest
import torch
from torch import nn

from src.training.config import parse_args
from src.training.methods import NAMES, get_method
from src.training.methods.base import Batch, StepContext


class ToyEncoder(nn.Module):
    final_feat_dim = 8

    def __init__(self):
        super().__init__()
        self.features = nn.Linear(1, self.final_feat_dim)
        self.mlp_head = nn.Linear(self.final_feat_dim, 4)

    def forward(self, images, y=None, patch_mix=False, **kwargs):
        if kwargs.get('return_tokens'):
            return self.features(images.flatten(2).transpose(1, 2))
        features = self.features(images.mean((2, 3)))
        if patch_mix:
            index = torch.arange(len(y)).roll(1)
            return 0.6 * features + 0.4 * features[index], y, y[index], 0.6, index
        return features


@pytest.fixture
def cpu_reference(monkeypatch):
    # Reference loss files hard-code Tensor.cuda(); run their unchanged algebra on CPU in CI.
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **kw: self)
    original_to = nn.Module.to
    def to_cpu(module, *args, **kwargs):
        if args and str(args[0]).startswith("cuda"):
            args = ("cpu", *args[1:])
        return original_to(module, *args, **kwargs)
    monkeypatch.setattr(nn.Module, "to", to_cpu)


@pytest.mark.parametrize("name,extra", [(name, ['--encoder', 'beats'] if name in ('ce_pafa_attn', 'ce_pafa_events') else []) for name in NAMES] + [
    ("pcmcl", ["--pcmcl_pafa", "1"]),
    ("patchmix_cl", ["--target_type", "grad_block"]),
    ("patchmix_cl", ["--target_type", "grad_flow"]),
    ("patchmix_cl", ["--target_type", "project_block"]),
    ("patchmix_cl", ["--encoder", "dass", "--cl_mode", "dual"]),
    ("patchmix_cl", ["--consistency_weight", "0.2"]),
    ("sg_scl", ["--grl_schedule", "progress"]),
])
def test_method_loss_backpropagates_and_returns_four_class_probabilities(name, extra, cpu_reference):
    torch.manual_seed(7)
    args = parse_args(["--method", name, *extra, *(['--num_workers', '0'] if name == 'ce_device_pafa' else [])])
    args.class_nums, args.proj_dim = [2, 2, 2, 2], 8
    method = get_method(name)
    model = method.wrap_encoder(ToyEncoder(), SimpleNamespace(image_shape=(16, 32)), args)
    parts = method.components(model, args)
    images = torch.randn(8, 1, 16, 32)
    labels = torch.arange(8) % 4
    if method.multilabel:
        from src.training.methods.pcmcl_data import multilabel
        labels = torch.tensor([multilabel(int(y)) for y in labels]).float()
    quality = (torch.arange(8) % 2).float() if method.multilabel else torch.full((8,), 0.8)
    batch = Batch(images, labels, torch.arange(8) % 2, torch.arange(8) // 2, quality)
    if name == 'ce_recording_pafa':
        batch.images = images[:, None].repeat(1, 2, 1, 1, 1)
        batch.labels = torch.cat([labels[:, None].repeat(1, 2), (labels != 0)[:, None]], dim=1)
        batch.quality = torch.ones(8, 2, dtype=torch.bool)
    elif name == 'ce_device_pafa':
        batch.images = torch.stack([images, images + 0.1], dim=1)
    context = StepContext(model, parts.classifier, parts.projector, parts.losses, args, 1, 0, 1,
                          method.trainable_frontend)
    result = method.step(context, batch)
    assert result.logits.shape == (8, 4) and result.labels.shape == (8,)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in parts.classifier.parameters())
    raw = torch.randn(8, 3 if method.multilabel else 4)
    probabilities = method.probabilities(raw)
    assert probabilities.shape == (8, 4)
    assert torch.allclose(probabilities.sum(1), torch.ones(8))


def test_ce_has_no_upstream_method_dependencies():
    subprocess.run([sys.executable, "-c", "import src.training.upstream as u; u.THIRD_PARTY = u.THIRD_PARTY / 'missing'; "
                    "from types import SimpleNamespace; from torch import nn; "
                    "from src.training.methods import get_method; "
                    "get_method('ce').components(SimpleNamespace(mlp_head=nn.Linear(2,4)), SimpleNamespace())"], check=True)


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="Unknown training method"):
        get_method("typo")


def test_gradient_reversal_changes_only_backward():
    from src.training.methods.sg_scl import GradientReversal
    x = torch.tensor([2.0], requires_grad=True)
    result = GradientReversal.apply(x, 0.4)
    assert torch.equal(result, x)
    result.sum().backward()
    assert torch.allclose(x.grad, torch.tensor([-0.4]))


@pytest.mark.parametrize("name", ["ce", "add_rsc"])
def test_frozen_encoder_keeps_only_method_frontend_trainable(name):
    from src.training.modeling import build_model
    from unittest.mock import patch
    args = parse_args(["--method", name, "--freeze_encoder"])
    spec = SimpleNamespace(image_shape=(16, 32), build=lambda _: ToyEncoder())
    with patch("src.training.modeling.get_encoder", return_value=spec):
        model, classifier, _, _, optimizer = build_model(args, device="cpu")
    encoder = model.encoder if name == "add_rsc" else model
    assert all(not p.requires_grad for p in encoder.parameters())
    assert all(p.requires_grad for p in classifier.parameters())
    if name == "add_rsc":
        assert all(p.requires_grad for p in model.denoise.parameters())
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert all(id(p) in optimized for p in classifier.parameters())
