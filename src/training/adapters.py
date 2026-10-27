"""Optional low-rank attention updates and shared-encoder temporal views."""
import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        if not isinstance(base, nn.Linear) or not 0 < rank <= min(base.in_features, base.out_features):
            raise ValueError('LoRA requires a linear layer and a valid positive rank')
        self.base = base
        self.lora_a = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_b = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        for parameter in base.parameters():
            parameter.requires_grad = False

    def forward(self, x):
        return self.base(x) + (x @ self.lora_a.T) @ self.lora_b.T


def apply_lora(model, rank):
    targets = [(name, layer) for name, layer in model.named_modules()
               if name.rsplit('.', 1)[-1] in ('q_proj', 'v_proj') and isinstance(layer, nn.Linear)]
    if not targets:
        raise ValueError('No query/value linear layers found for LoRA')
    if any(rank > min(layer.in_features, layer.out_features) for _, layer in targets):
        raise ValueError('LoRA rank exceeds an attention projection dimension')
    for parameter in model.parameters():
        parameter.requires_grad = False
    for name, layer in targets:
        parent_name, _, child = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, LoRALinear(layer, rank))
    return len(targets)


def multiscale_features(images, encode):
    """Equal-weight full context and two overlapping half-duration views."""
    if images.ndim != 4 or images.shape[2] < 32:
        raise ValueError('Multi-scale BEATs needs at least two temporal patches')
    length = (images.shape[2] + 1) // 2
    full = encode(images)
    short = encode(torch.cat((images[:, :, :length], images[:, :, -length:]), 0))
    first, last = short.chunk(2, 0)
    return 0.5 * full + 0.25 * (first + last)
