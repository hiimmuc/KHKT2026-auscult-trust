"""ADD-RSC adaptive differential denoising and bias-denoising objective."""
from auscult_trust.models.heads import AddRscHead
from auscult_trust.models.upstream import load_reference
from .base import StepResult, TrainingMethod


class Method(AddRscHead, TrainingMethod):
    trainable_frontend = True

    def losses(self, model, args):
        return [*super().losses(model, args),
                load_reference("ADD-RSC", "models/bias_denoise_loss.py").LabelSmoothingLoss(size=args.n_cls)]

    def step(self, context, batch):
        c, a = context, context.args
        output = c.classifier(c.encode(batch.images))
        loss = (1 - a.loss_beta) * c.criteria[0](output, batch.labels) + a.loss_beta * c.criteria[1](c.model.aux, batch.labels)
        return StepResult(output, loss, batch.labels)
