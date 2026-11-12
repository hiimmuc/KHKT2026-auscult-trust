"""Prediction heads: how a training method wraps its encoder, builds its classifier and turns logits into probabilities.

Training methods inherit the head of their name, so inference builds the same network from `get_head` without importing
any training code.
"""
from copy import deepcopy

import torch
from torch import nn

from src.models.encoders import image_shape
from src.models.encoders.beats import TokenClassifier
from src.models.upstream import load_reference


def four_class_probs(logits):
    """(n, 3) logits -> (n, 4) probabilities of normal, crackle, wheeze, both (crackle and wheeze independent)."""
    s = torch.sigmoid(logits.float())
    c, w = s[:, 1], s[:, 2]
    return torch.stack([(1 - c) * (1 - w), c * (1 - w), (1 - c) * w, c * w], 1)


class Head:
    """Cross-entropy head: a copy of the encoder's `mlp_head` and a softmax."""

    supports_data_parallel = True

    def wrap_encoder(self, model, spec, args):
        return model

    def classifier(self, model, args):
        return deepcopy(model.mlp_head)

    def encoder_parameters(self, model):
        return model.parameters()

    def probabilities(self, logits):
        return torch.softmax(logits.float(), 1)


class PafaHead(Head):
    def classifier(self, model, args):
        head = super().classifier(model, args)
        pool = getattr(args, "class_pool", "mean")
        return head if pool == "mean" else TokenClassifier(head, pool, getattr(args, "class_topk_fraction", 0.2))


class EventClassifier(nn.Module):
    def __init__(self, cycle_head, dimension):
        super().__init__()
        self.cycle_head = cycle_head
        self.event_head = nn.Linear(dimension, 2)

    def forward(self, features):
        return self.cycle_head(features)


class EventHead(PafaHead):
    def classifier(self, model, args):
        if getattr(args, "class_pool", "mean") != "mean" or args.n_cls != 4:
            raise ValueError("Event supervision requires mean-pooled four-class classification")
        if getattr(args, "soft_targets", None):
            raise ValueError("Event supervision requires observed cycle labels")
        return EventClassifier(deepcopy(model.mlp_head), model.final_feat_dim)


class RecordingClassifier(nn.Module):
    def __init__(self, cycle_head, dimension):
        super().__init__()
        self.cycle_head = cycle_head
        self.recording_head = nn.Linear(dimension, 2)

    def forward(self, features):
        return self.cycle_head(features)

    def recording_logits(self, features, mask=None):
        window_logits = self.recording_head(features)
        abnormal = window_logits[..., 1] - window_logits[..., 0]
        if mask is not None:
            mask = mask.to(features.device).bool()
            if not torch.all(mask.any(1)):
                raise ValueError("Every recording must contain a valid window")
            abnormal = abnormal.masked_fill(~mask, -torch.inf)
        score = abnormal.max(1).values
        return torch.stack([torch.zeros_like(score), score], 1)


class RecordingHead(PafaHead):
    def classifier(self, model, args):
        return RecordingClassifier(deepcopy(model.mlp_head), model.final_feat_dim)


class QLungHead(Head):
    def classifier(self, model, args):
        return load_reference("QLung", "method/loss.py").AngularClassifier(model.final_feat_dim, args.n_cls, s=37)


class PcmclHead(Head):
    def classifier(self, model, args):
        return nn.Linear(model.final_feat_dim, 3)

    def probabilities(self, logits):
        return four_class_probs(logits)


class DenoisedEncoder(nn.Module):
    def __init__(self, encoder, n_freq, args):
        super().__init__()
        self.encoder, self.final_feat_dim, self.mlp_head = encoder, encoder.final_feat_dim, encoder.mlp_head
        layer = load_reference("ADD-RSC", "models/adapt_diff_denoise.py").DiffTransformerLayer
        self.denoise = layer(d_model=n_freq, num_heads=args.denoise_heads, depth=args.denoise_depth)
        self.denoise.classifier = nn.Linear(n_freq, args.n_cls)
        self.aux = None

    def forward(self, x, **kwargs):
        z, self.aux = self.denoise(x[:, 0])
        return self.encoder(z.unsqueeze(1), **kwargs)


class AddRscHead(Head):
    supports_data_parallel = False  # reference denoiser stores auxiliary logits on its encoder

    def wrap_encoder(self, model, spec, args):
        return DenoisedEncoder(model, image_shape(spec, args)[1], args)

    def encoder_parameters(self, model):
        return model.encoder.parameters()


HEADS = {"ce": Head, "patchmix": Head, "patchmix_cl": Head, "sg_scl": Head, "ce_pafa": PafaHead,
         "ce_pafa_attn": PafaHead, "ce_pafa_class_aware": PafaHead, "ce_device_pafa": PafaHead,
         "ce_recording_pafa": RecordingHead, "ce_pafa_events": EventHead, "qlung": QLungHead, "add_rsc": AddRscHead,
         "pcmcl": PcmclHead}


def get_head(name):
    try:
        return HEADS[name]()
    except KeyError:
        raise ValueError(f"Unknown training method: {name}; choose from {', '.join(HEADS)}") from None
