"""Patient-consistent multi-cycle learning with optional PAFA alignment."""
import torch
from torch import nn

from auscult_trust.models.heads import PcmclHead, four_class_probs

from .base import StepResult, TrainingMethod
from .pafa import alignment_loss, projection_head
from .pcmcl_data import four_class


class Method(PcmclHead, TrainingMethod):
    multilabel = True

    def projector(self, model, args):
        projector = nn.ModuleDict({"ssl": nn.Linear(model.final_feat_dim, 2)})
        if args.pcmcl_pafa:
            projector["pafa"] = projection_head(model.final_feat_dim)
        return projector

    def losses(self, model, args):
        losses = [nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss()]
        if args.pcmcl_pafa:
            losses.append(alignment_loss())
        return losses

    def step(self, context, batch):
        c, a = context, context.args
        features = c.encode(batch.images)
        output = c.classifier(features)
        loss = c.criteria[0](output.float(), batch.labels.float())
        ssl = batch.quality.to(batch.images.device, non_blocking=True)
        if (ssl >= 0).any():
            loss = loss + a.pcmcl_aux * c.criteria[1](c.projector["ssl"](features[ssl >= 0]).float(), ssl[ssl >= 0].long())
        one = batch.patients >= 0
        if a.pcmcl_pafa and one.sum() > 1:
            loss = loss + c.criteria[2](c.projector["pafa"](features[one.to(batch.images.device)]).float(),
                       batch.patients[one].to(batch.images.device, non_blocking=True),
                       lambda_pcsl=a.lambda_pcsl, lambda_gpal=a.lambda_gpal)
        return StepResult(torch.log(four_class_probs(output) + 1e-8), loss, four_class(batch.labels))
