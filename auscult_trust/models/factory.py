"""Construct the encoder and prediction head of a configured model, without losses, projectors or an optimizer."""
import torch
from torch import nn

from auscult_trust.models.encoders import get_encoder
from auscult_trust.models.heads import get_head


def build_encoder(args, head):
    spec = get_encoder(args.encoder)
    model = head.wrap_encoder(spec.build(args), spec, args)
    args.proj_dim = model.final_feat_dim
    return model


def place_encoder(model, head, args, device):
    if str(device).startswith("cuda") and torch.cuda.device_count() > 1:
        if not head.supports_data_parallel:
            raise ValueError(f"{args.method} requires one visible GPU; set CUDA_VISIBLE_DEVICES")
        model = nn.DataParallel(model)
    model.to(device)
    if args.freeze_encoder:
        underlying = model.module if isinstance(model, nn.DataParallel) else model
        for parameter in head.encoder_parameters(underlying):
            parameter.requires_grad = False
    return model


def build_inference_model(args, device="cuda"):
    """Return the encoder and the classifier head."""
    head = get_head(args.method)
    model = build_encoder(args, head)
    classifier = head.classifier(model, args)
    return place_encoder(model, head, args, device), classifier.to(device)
