"""PAFA on bandwidth/noise views, with optional clean-view consistency."""
from dataclasses import replace

from . import pafa
from .base import StepResult
from ..augment import js_divergence


class Method(pafa.Method):
    supports_consistency = True

    def step(self, context, batch):
        if batch.images.ndim != 5 or batch.images.shape[1] != 2:
            raise ValueError("Device robustness training needs clean/augmented waveform views")
        augmented = replace(batch, images=batch.images[:, 1])
        result = super().step(context, augmented)
        a = context.args
        if a.consistency_weight <= 0:
            return result
        clean_logits = context.classifier(context.encode(batch.images[:, 0]))
        js = js_divergence(clean_logits, result.logits)
        weight = a.consistency_weight * min(1.0, context.epoch / max(a.consistency_ramp, 1))
        agree = (clean_logits.argmax(1) == result.logits.argmax(1)).float().mean().item() * 100
        return StepResult(result.logits, result.loss + weight * js, result.labels, (js.item(), agree))
