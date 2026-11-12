"""Cross-entropy classification and optional spectral consistency regularization."""
from ..augment import js_divergence, random_bin_gain_batch
from src.models.encoders import get_encoder
from .base import StepResult, TrainingMethod


def classify(context, batch):
    c, a = context, context.args
    features = c.encode(batch.images)
    output = c.classifier(features)
    loss = c.criteria[0](output, batch.labels)
    consistency = None
    if a.consistency_weight > 0:
        style = random_bin_gain_batch(batch.images, a.consistency_gain_db, get_encoder(a.encoder).db_scale)
        out_style = c.classifier(c.encode(style))
        js = js_divergence(output, out_style)
        weight = a.consistency_weight * min(1.0, c.epoch / max(a.consistency_ramp, 1))
        loss = loss + weight * js
        consistency = (js.item(), (output.argmax(1) == out_style.argmax(1)).float().mean().item() * 100)
    return features, StepResult(output, loss, batch.labels, consistency)


class Method(TrainingMethod):
    def step(self, context, batch):
        return classify(context, batch)[1]
