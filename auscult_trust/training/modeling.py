"""Construct the training model: encoder, classifier, projector, losses and optimizer."""
import torch

from auscult_trust.models.factory import build_encoder, place_encoder
from .methods import get_method
from .optim import layer_decay_parameters, set_optimizer


def build_model(args, device=None):
    """Return (encoder, classifier, projector, losses, optimizer) for training, on `device` (CUDA when available, else CPU)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    method = get_method(args.method)
    model = build_encoder(args, method)
    parts = method.components(model, args)
    classifier, projector = parts.classifier, parts.projector
    criterion = [loss.to(device) for loss in parts.losses]
    model = place_encoder(model, method, args, device)
    classifier.to(device), projector.to(device)
    parameters = (
        [p for p in model.parameters() if p.requires_grad]
        + list(classifier.parameters())
        + list(projector.parameters())
        + [p for loss in criterion[1:] for p in loss.parameters()]
    )
    if getattr(args, "layer_decay", 1.0) != 1.0:
        parameters = layer_decay_parameters(args, model, parameters)
    return model, classifier, projector, criterion, set_optimizer(args, parameters)
