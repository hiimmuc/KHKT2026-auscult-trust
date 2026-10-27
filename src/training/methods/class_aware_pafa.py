"""PAFA centroid alignment conditional on clinical class, with hard-label groups only."""
import torch
from torch import nn
from torch.nn import functional as F

from .base import StepResult
from .pafa import Method as PafaMethod


def centroid_pair_masks(patients, labels):
    different_patient = patients[:, None] != patients[None, :]
    same_class = labels[:, None] == labels[None, :]
    upper = torch.ones_like(same_class).triu(1)
    return upper & same_class & different_patient, upper & ~same_class


class ClassAwarePAFA(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, features, patients, labels, lambda_pcsl=0.1, lambda_gpal=0.1):
        zero = features.sum() * 0
        valid = patients >= 0
        features, patients, labels = features[valid], patients[valid], labels[valid]
        if not len(features):
            return zero
        groups = torch.unique(torch.stack((patients, labels), 1), dim=0)
        centroids, within = [], zero
        for patient, label in groups:
            group = features[(patients == patient) & (labels == label)]
            centroid = group.mean(0)
            centroids.append(centroid)
            within = within + (group - centroid).square().sum()
        centroids = torch.stack(centroids)
        positive, negative = centroid_pair_masks(groups[:, 0], groups[:, 1])
        distances = (centroids[:, None] - centroids[None, :]).square().sum(-1)
        pcsl = within / (distances[negative].sum() + self.eps) if negative.any() else zero
        alignments = []
        for label in labels.unique():
            same = groups[:, 1] == label
            if positive[same].any():
                centers = centroids[same]
                alignments.append((centers - centers.mean(0)).square().sum(1).mean())
        gpal = torch.stack(alignments).mean() if alignments else zero
        return lambda_pcsl * pcsl + lambda_gpal * gpal


class Method(PafaMethod):
    def losses(self, model, args):
        return [nn.CrossEntropyLoss(), ClassAwarePAFA()]

    def step(self, context, batch):
        c, a = context, context.args
        features = c.encode(batch.images)
        output = c.classifier(features)
        classification = (-(batch.soft_targets.float() * F.log_softmax(output.float(), 1)).sum(1).mean()
                          if batch.soft_targets is not None else c.criteria[0](output, batch.labels))
        hard = torch.ones_like(batch.labels, dtype=torch.bool)
        if batch.soft_targets is not None:
            targets = F.one_hot(batch.labels, output.shape[1]).to(batch.soft_targets)
            hard = torch.isclose(batch.soft_targets, targets, atol=1e-6, rtol=0).all(1)
        alignment_features = features.mean(1) if features.ndim == 3 else features
        loss = classification + c.criteria[1](c.projector(alignment_features)[hard].float(),
            batch.patients.to(features.device)[hard], batch.labels[hard],
            lambda_pcsl=a.lambda_pcsl, lambda_gpal=a.lambda_gpal)
        return StepResult(output, loss, batch.labels)
