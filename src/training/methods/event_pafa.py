"""Cycle PAFA with auxiliary crackle and wheeze supervision."""
from copy import deepcopy
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from . import pafa
from .base import StepResult


def event_targets(labels):
    """Four-class IDs (normal, crackle, wheeze, both) to two observed events."""
    if labels.ndim != 1 or torch.any((labels < 0) | (labels > 3)):
        raise ValueError('Event supervision requires four-class cycle labels')
    return torch.stack([labels & 1, (labels >> 1) & 1], 1).float()


class EventClassifier(nn.Module):
    def __init__(self, cycle_head, dimension):
        super().__init__()
        self.cycle_head = cycle_head
        self.event_head = nn.Linear(dimension, 2)

    def forward(self, features):
        return self.cycle_head(features)


class Method(pafa.Method):
    def classifier(self, model, args):
        if getattr(args, 'class_pool', 'mean') != 'mean' or args.n_cls != 4:
            raise ValueError('Event supervision requires mean-pooled four-class classification')
        if getattr(args, 'soft_targets', None):
            raise ValueError('Event supervision requires observed cycle labels')
        return EventClassifier(deepcopy(model.mlp_head), model.final_feat_dim)

    def step(self, context, batch):
        if batch.soft_targets is not None:
            raise ValueError('Event supervision does not accept teacher soft targets')
        features = context.encode(batch.images)
        proxy = SimpleNamespace(**vars(context))
        proxy.encode = lambda _: features
        result = pafa.Method.step(self, proxy, batch)
        head = getattr(context.classifier, 'module', context.classifier)
        event_logits = head.event_head(features)
        # Both observed events receive equal weight, independent of their prevalence.
        event_loss = F.binary_cross_entropy_with_logits(
            event_logits.float(), event_targets(batch.labels), reduction='none').mean(0).mean()
        return StepResult(result.logits, result.loss + context.args.event_weight * event_loss,
                          result.labels, result.consistency)
