"""BEATs patient alignment with learned attention over transformer tokens."""
from .base import StepResult
from .pafa import Method as PafaMethod, projection_head


class Method(PafaMethod):
    def projector(self, model, args):
        return projection_head(model.final_feat_dim, attention=True)

    def step(self, context, batch):
        c, a = context, context.args
        tokens = c.encode(batch.images, return_tokens=True)
        output = c.classifier(tokens).mean(1)
        loss = c.criteria[0](output, batch.labels) + c.criteria[1](
            c.projector(tokens).float(), batch.patients.to(batch.images.device, non_blocking=True),
            lambda_pcsl=a.lambda_pcsl, lambda_gpal=a.lambda_gpal)
        return StepResult(output, loss, batch.labels)
