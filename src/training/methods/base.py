"""Contracts shared by the trainer and method implementations."""
from abc import ABC, abstractmethod
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class Components:
    classifier: nn.Module
    projector: nn.Module
    losses: list


@dataclass
class Batch:
    images: torch.Tensor
    labels: torch.Tensor
    devices: torch.Tensor
    patients: torch.Tensor
    quality: torch.Tensor
    soft_targets: torch.Tensor | None = None


@dataclass
class StepResult:
    logits: torch.Tensor
    loss: torch.Tensor
    labels: torch.Tensor
    consistency: tuple | None = None  # (JS loss, prediction agreement percentage)


@dataclass
class StepContext:
    model: nn.Module
    classifier: nn.Module
    projector: nn.Module
    criteria: list
    args: object
    epoch: int
    index: int
    steps: int
    trainable_frontend: bool = False

    def encode(self, images, **kwargs):
        frozen = self.args.freeze_encoder and not self.trainable_frontend
        with torch.no_grad() if frozen else nullcontext():
            return self.model(images, **kwargs)


class TrainingMethod(ABC):
    trainable_frontend = False
    two_views = False
    multilabel = False
    supports_data_parallel = True
    supports_consistency = False
    supports_dual_mix = False

    def quality_scores(self, dataset, annotations, wav_dir, cache_dir):
        import numpy as np
        return np.zeros(len(dataset.labels), np.float32)

    def wrap_encoder(self, model, spec, args):
        return model

    def components(self, model, args):
        return Components(self.classifier(model, args), self.projector(model, args), self.losses(model, args))

    def classifier(self, model, args):
        return deepcopy(model.mlp_head)

    def projector(self, model, args):
        return nn.Identity()

    def losses(self, model, args):
        return [nn.CrossEntropyLoss()]

    def encoder_parameters(self, model):
        return model.parameters()

    def probabilities(self, logits):
        return torch.softmax(logits.float(), 1)

    @abstractmethod
    def step(self, context, batch):
        """Return differentiable loss, accuracy logits, and accuracy labels for one batch."""
