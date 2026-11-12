
import pytest
import torch
from torch import nn

from auscult_trust.models.adapters import LoRALinear, apply_lora, multiscale_features


def test_lora_starts_identical_and_trains_only_adapter():
    torch.manual_seed(3)
    base = nn.Linear(8, 6)
    x = torch.randn(4, 8)
    expected = base(x).detach()
    layer = LoRALinear(base, rank=2)
    assert torch.equal(layer(x), expected)
    layer(x).square().mean().backward()
    assert base.weight.grad is None
    assert layer.lora_b.grad.abs().sum() > 0
    assert not base.bias.requires_grad
    with torch.no_grad():
        layer.lora_b.add_(0.1)
    other = LoRALinear(nn.Linear(8, 6), rank=2)
    other.load_state_dict(layer.state_dict())
    assert torch.equal(layer(x), other(x))


def test_lora_injection_targets_qv_and_rejects_no_targets():
    model = nn.ModuleDict({'q_proj': nn.Linear(8, 8), 'v_proj': nn.Linear(8, 8), 'k_proj': nn.Linear(8, 8)})
    assert apply_lora(model, 2) == 2
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert len(trainable) == 4 and all('lora_' in n for n in trainable)
    with pytest.raises(ValueError):
        apply_lora(nn.Linear(8, 8), 2)


def test_multiscale_shares_encoder_and_covers_both_halves():
    calls = []
    scale = nn.Parameter(torch.tensor(2.0))
    def encode(x):
        calls.append(x.shape)
        return x.mean((1, 2, 3)).unsqueeze(1) * scale
    x = torch.arange(2 * 1 * 65 * 16, dtype=torch.float32).reshape(2, 1, 65, 16)
    out = multiscale_features(x, encode)
    assert calls == [torch.Size([2, 1, 65, 16]), torch.Size([4, 1, 33, 16])]
    expected = (encode(x) + (encode(x[:, :, :33]) + encode(x[:, :, -33:])) / 2) / 2
    assert torch.allclose(out, expected)
    out.sum().backward()
    assert scale.grad.abs() > 0
    with pytest.raises(ValueError):
        multiscale_features(x[:, :, :16], encode)
