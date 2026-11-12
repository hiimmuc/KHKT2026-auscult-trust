"""Joint cycle classification and binary recording screening with shared features."""
from types import SimpleNamespace

import torch
from torch import nn
from auscult_trust.models.heads import RecordingHead

from . import pafa
from .base import Batch, StepResult


class Method(RecordingHead, pafa.Method):
    supports_consistency = False

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
