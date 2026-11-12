"""Patch-Mix augmentation with mixed-label cross entropy."""
from auscult_trust.models.upstream import load_reference
from .base import StepResult, TrainingMethod


class Method(TrainingMethod):
    def losses(self, model, args):
        losses = super().losses(model, args)
        loss = load_reference("patch-mix_contrastive_learning", "method/patchmix.py").PatchMixLoss
        return [*losses, loss(criterion=losses[0])]

    def step(self, context, batch):
        c = context
        features, labels_a, labels_b, lam, _ = c.encode(
            batch.images, y=batch.labels, patch_mix=True, time_domain=c.args.time_domain)
        output = c.classifier(features)
        return StepResult(output, c.criteria[1](output, labels_a, labels_b, lam), batch.labels)
