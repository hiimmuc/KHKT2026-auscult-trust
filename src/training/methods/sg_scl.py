"""Stethoscope-guided supervised contrastive learning with gradient reversal."""
import numpy as np
import torch

from ..augment import SpecAugment
from src.models.upstream import load_reference
from .base import StepResult, TrainingMethod


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, gradient):
        return gradient.neg() * ctx.alpha, None


def augment_views(images):
    aug = SpecAugment()
    return torch.stack([aug(x.clone()) for x in images])


class Method(TrainingMethod):
    two_views = True

    def projector(self, model, args):
        return load_reference("patch-mix_contrastive_learning", "models/projector.py").Projector(
            model.final_feat_dim, args.proj_dim)

    def losses(self, model, args):
        return [*super().losses(model, args), load_reference("SG-SCL", "method/mcl.py").MetaCL(temperature=args.temperature)]

    def step(self, context, batch):
        c, a = context, context.args
        first, second = augment_views(batch.images), augment_views(batch.images)
        f1, f2 = c.encode(first), c.encode(second)
        output = c.classifier(f1)
        if a.grl_schedule == "reference":
            progress = float(c.index + c.epoch * c.steps) / a.epochs + 1 / c.steps
        else:
            progress = float(c.index + (c.epoch - 1) * c.steps) / (a.epochs * c.steps)
        strength = 2.0 / (1.0 + np.exp(-10 * progress)) - 1
        first = c.projector(GradientReversal.apply(f1, strength))
        second = c.projector(GradientReversal.apply(f2, strength)).detach()
        loss = c.criteria[0](output, batch.labels) + a.alpha * c.criteria[1](
            first, second, batch.devices.to(batch.images.device, non_blocking=True))
        return StepResult(output, loss, batch.labels)
