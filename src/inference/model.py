"""Frozen checkpoint loading, feature extraction and batched prediction."""

import numpy as np
import torch
from src.training.encoders import get_encoder
from src.training.modeling import build_inference_model


def load_frozen(args, ckpt_path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, classifier = build_inference_model(args, device=device)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ck["model"] is not None:
        model.load_state_dict(ck["model"])
    classifier.load_state_dict(ck["classifier"])
    return model.eval(), classifier.eval()


def images_of(waves, args):
    enc = get_encoder(args.encoder)
    return np.stack([enc.preprocess(torch.from_numpy(w)[None]) for w in waves])[..., 0]  # (n, time, freq)


@torch.no_grad()
def predict_images(images, model, classifier, curves=None, *, method="ce"):
    """Softmax on (n, time, freq) images; `curves` (n, freq) are added to every frame (TTA-EQ view)."""
    from src.training.methods import get_method
    objective = get_method(method)
    parameter = next(model.parameters(), None)
    if parameter is None:
        parameter = next(classifier.parameters(), torch.empty(0))
    device = parameter.device
    probs = []
    for i in range(0, len(images), 32):
        x = torch.from_numpy(images[i : i + 32]).float()[:, None]
        if curves is not None:
            x = x + torch.from_numpy(curves[i : i + 32])[:, None, None, :]
        o = classifier(model(x.to(device)))  # fp32, as the earlier KAUH conditions
        probs.append(objective.probabilities(o).cpu().numpy())
    return np.concatenate(probs)


def predict(waves, args, ckpt_path):
    """Softmax of the frozen model on every waveform."""
    model, classifier = load_frozen(args, ckpt_path)
    return predict_images(images_of(waves, args), model, classifier, method=args.method)
