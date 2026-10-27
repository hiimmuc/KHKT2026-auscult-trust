"""Joint cycle classification and binary recording screening with shared features."""
from copy import deepcopy
from types import SimpleNamespace

import torch
from torch import nn
from . import pafa
from .base import Batch, StepResult


class RecordingClassifier(nn.Module):
    def __init__(self, cycle_head, dimension):
        super().__init__()
        self.cycle_head = cycle_head
        self.recording_head = nn.Linear(dimension, 2)

    def forward(self, features):
        return self.cycle_head(features)

    def recording_logits(self, features, mask=None):
        window_logits = self.recording_head(features)
        abnormal = window_logits[..., 1] - window_logits[..., 0]
        if mask is not None:
            mask = mask.to(features.device).bool()
            if not torch.all(mask.any(1)):
                raise ValueError("Every recording must contain a valid window")
            abnormal = abnormal.masked_fill(~mask, -torch.inf)
        score = abnormal.max(1).values
        return torch.stack([torch.zeros_like(score), score], 1)


class Method(pafa.Method):
    supports_consistency = False

    def classifier(self, model, args):
        return RecordingClassifier(deepcopy(model.mlp_head), model.final_feat_dim)

    def step(self, context, batch):
        c = context
        if batch.images.ndim != 5 or batch.labels.shape[1] != batch.images.shape[1] + 1:
            raise ValueError("Recording training requires padded bags and recording labels")
        b, k = batch.images.shape[:2]
        mask = batch.quality.to(batch.images.device).bool()
        if mask.shape != (b, k) or not torch.all(mask[:, 0]):
            raise ValueError("Recording bags require a valid first cycle and a window mask")
        images = batch.images[mask]
        features = c.encode(images)
        cycle_batch = Batch(images, batch.labels[:, :k][mask], batch.devices,
                            batch.patients[:, None].expand(b, k)[mask.to(batch.patients.device)], batch.quality,
                            batch.soft_targets[mask] if batch.soft_targets is not None else None)
        # Reuse PAFA's exact cycle objective while encoding each valid window only once.
        proxy = SimpleNamespace(**vars(c))
        proxy.encode = lambda _: features
        cycle_result = pafa.Method.step(self, proxy, cycle_batch)
        full = features.new_zeros((b, k, features.shape[-1]))
        full[mask] = features
        head = getattr(c.classifier, "module", c.classifier)
        logits = head.recording_logits(full, mask)
        recording_loss = nn.functional.cross_entropy(logits.float(), batch.labels[:, -1])
        cycle_logits = c.classifier(full[:, 0])
        return StepResult(cycle_logits, cycle_result.loss + c.args.recording_weight * recording_loss,
                          batch.labels[:, 0])
