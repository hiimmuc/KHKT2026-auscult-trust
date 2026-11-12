"""Cross entropy with patient-aware feature alignment (PAFA). Config key: ce_pafa."""
from torch.nn import functional as F
from auscult_trust.models.heads import PafaHead
from auscult_trust.models.upstream import load_reference
from .base import StepResult, TrainingMethod


def distillation_loss(logits, labels, targets, hard_weight=0., temperature=1.):
    """Blend observed-label CE with temperature-scaled teacher KL."""
    if hard_weight == 0 and temperature == 1:
        return -(targets.float() * F.log_softmax(logits.float(), 1)).sum(1).mean()
    teacher = F.softmax(targets.float().clamp_min(1e-30).log() / temperature, 1)
    soft = F.kl_div(F.log_softmax(logits.float() / temperature, 1), teacher,
                    reduction='batchmean') * temperature**2
    return hard_weight * F.cross_entropy(logits.float(), labels) + (1 - hard_weight) * soft


def projection_head(dimension, *, attention=False):
    return load_reference("PAFA", "method/pafa.py").ProjectionHead(
        dimension, None, dimension, attention=attention, norm_type="ln")


def alignment_loss():
    return load_reference("PAFA", "method/pafa.py").PAFALoss()


class Method(PafaHead, TrainingMethod):
    def projector(self, model, args):
        return projection_head(model.final_feat_dim)

    def losses(self, model, args):
        return [*super().losses(model, args), alignment_loss()]

    def step(self, context, batch):
        c, a = context, context.args
        features = c.encode(batch.images)
        output = c.classifier(features)
        classification = (distillation_loss(output, batch.labels, batch.soft_targets,
                                            getattr(a, "kd_hard_weight", 0.), getattr(a, "kd_temperature", 1.))
                          if batch.soft_targets is not None else c.criteria[0](output, batch.labels))
        alignment_features = features.mean(1) if features.ndim == 3 else features
        loss = classification + c.criteria[1](
            c.projector(alignment_features).float(), batch.patients.to(batch.images.device, non_blocking=True),
            lambda_pcsl=a.lambda_pcsl, lambda_gpal=a.lambda_gpal)
        return StepResult(output, loss, batch.labels)
