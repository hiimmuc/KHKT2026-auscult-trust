"""Construct shared encoder/classifier networks and optional training components."""
import torch
from torch import nn

from .encoders import get_encoder
from .methods import get_method
from .optim import layer_decay_parameters, set_optimizer


def _build_encoder(args):
    method = get_method(args.method)
    spec = get_encoder(args.encoder)
    model = method.wrap_encoder(spec.build(args), spec, args)
    args.proj_dim = model.final_feat_dim
    return method, model


def _place_encoder(model, method, args, device):
    if str(device).startswith("cuda") and torch.cuda.device_count() > 1:
        if not method.supports_data_parallel:
            raise ValueError(f"{args.method} requires one visible GPU; set CUDA_VISIBLE_DEVICES")
        model = nn.DataParallel(model)
    model.to(device)
    if args.freeze_encoder:
        underlying = model.module if isinstance(model, nn.DataParallel) else model
        for parameter in method.encoder_parameters(underlying):
            parameter.requires_grad = False
    return model


def build_inference_model(args, device="cuda"):
    """Construct only the encoder and prediction head, without losses, projectors or optimizer."""
    method, model = _build_encoder(args)
    classifier = method.classifier(model, args)
    return _place_encoder(model, method, args, device), classifier.to(device)


def build_model(args, device="cuda"):
    """Return (encoder, classifier, projector, losses, optimizer) for training."""
    method, model = _build_encoder(args)
    parts = method.components(model, args)
    classifier, projector = parts.classifier, parts.projector
    criterion = [loss.to(device) for loss in parts.losses]
    model = _place_encoder(model, method, args, device)
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
