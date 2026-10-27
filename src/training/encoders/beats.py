"""BEATs (iter3+, AudioSet-2M fine-tuned backbone without its head) with Patch-Mix on the patch embeddings.

Network code: the BEATs module of microsoft/unilm as copied in `repos/PAFA/BEATs` (loaded by path). Input: the official
`BEATs.preprocess` (Kaldi fbank of wave * 2^15, 128 bins, 25 ms / 10 ms, normalised with 15.41663 and 6.55582); 8 s -> 798 x 128.
The 16 x 16 patch tokens after `patch_embedding`, layer norm and `post_extract_proj` are mixed (as the AST hook does), then the
transformer encoder; features = mean over tokens. Weights: `<data>/models/beats/BEATs_iter3_plus_AS2M.pt`.
"""

import importlib
import math
import sys
from functools import lru_cache

import torch
import torch.nn as nn
import torchaudio
from src.paths import DATA, REPOS

from src.processing.cycles import DB_PER_NAT
from .patch_mix import mix_patches
from ..adapters import apply_lora, multiscale_features

PAFA = REPOS / "PAFA"
WEIGHTS = DATA / "models" / "beats" / "BEATs_iter3_plus_AS2M.pt"
FBANK_MEAN, FBANK_STD = 15.41663, 6.55582
FEATURE_DIM = 768
DB_SCALE = 1 / (2 * FBANK_STD * DB_PER_NAT)


@lru_cache(maxsize=1)
def _beats_module():
    if str(PAFA) not in sys.path:  # the reference imports its package as `BEATs`
        sys.path.insert(0, str(PAFA))
    return importlib.import_module("BEATs.BEATs")


class BeatsEncoder(nn.Module):
    final_feat_dim = FEATURE_DIM
    class_pool = "mean"
    multiscale = False

    def __init__(self, n_cls, mix_beta, class_pool="mean", lora_rank=0, multiscale=False):
        super().__init__()
        mod = _beats_module()
        ck = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
        cfg = mod.BEATsConfig({**ck["cfg"], "finetuned_model": False, "spec_transform": None})
        self.beats = mod.BEATs(cfg)
        self.beats.load_state_dict(ck["model"])
        self.mlp_head = nn.Linear(FEATURE_DIM, n_cls)
        self.mix_beta = mix_beta
        self.class_pool = class_pool
        self.multiscale = multiscale
        if lora_rank:
            apply_lora(self.beats, lora_rank)

    def forward(self, x, y=None, patch_mix=False, time_domain=False, return_tokens=False):
        if self.multiscale:
            if patch_mix or return_tokens or self.class_pool != 'mean':
                raise ValueError('Multi-scale supports pooled BEATs features only')
            return multiscale_features(x, self._encode)
        return self._encode(x, y, patch_mix, return_tokens)

    def _encode(self, x, y=None, patch_mix=False, return_tokens=False):
        b = self.beats
        f = b.patch_embedding(x)  # (B, 1, time, freq) -> (B, C, time / 16, freq / 16)
        time, freq = f.shape[-2:]
        f = b.layer_norm(f.reshape(f.shape[0], f.shape[1], -1).transpose(1, 2))
        if b.post_extract_proj is not None:
            f = b.post_extract_proj(f)
        if patch_mix:
            f, y_a, y_b, lam, index = mix_patches(f, y, self.mix_beta)
        h, _ = b.encoder(b.dropout_input(f), padding_mask=None)
        if not return_tokens:
            h = (h.reshape(h.shape[0], time, freq, -1).mean(2)
                 if getattr(self, "class_pool", "mean") != "mean" else h.mean(1))
        return (h, y_a, y_b, lam, index) if patch_mix else h


def build_beats(args):
    return BeatsEncoder(args.n_cls, args.mix_beta, getattr(args, "class_pool", "mean"),
                        getattr(args, 'lora_rank', 0), getattr(args, 'multiscale', False))


class TokenClassifier(nn.Module):
    """Classify frequency-averaged time tokens; masks use True for padding."""

    def __init__(self, head, pool, fraction=0.2):
        super().__init__()
        if pool not in ("topk", "attention") or not 0 < fraction <= 1:
            raise ValueError("classification pooling requires topk/attention and a fraction in (0, 1]")
        self.head, self.pool, self.fraction = head, pool, fraction
        if pool == "attention":
            self.attention = nn.Linear(head.in_features, 1)

    def forward(self, tokens, padding_mask=None):
        if padding_mask is None:
            padding_mask = torch.zeros(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        if padding_mask.shape != tokens.shape[:2] or padding_mask.all(1).any():
            raise ValueError("classification pooling needs at least one valid time token per sample")
        if self.pool == "attention":
            weights = self.attention(tokens).squeeze(-1).masked_fill(padding_mask, -torch.inf).softmax(1)
            return self.head((tokens * weights.unsqueeze(-1)).sum(1))
        logits = self.head(tokens)
        pooled = []
        for scores, mask in zip(logits, padding_mask):
            valid = scores[~mask]
            pooled.append(valid.topk(math.ceil(valid.shape[0] * self.fraction), dim=0).values.mean(0))
        return torch.stack(pooled)


def preprocess_beats(wave):
    """(1, N) 16 kHz waveform -> (frames, 128, 1) image, as `BEATs.preprocess`."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fbank = torchaudio.compliance.kaldi.fbank(wave.to(device) * 2**15, num_mel_bins=128, sample_frequency=16000,
                                              frame_length=25, frame_shift=10)
    return ((fbank - FBANK_MEAN) / (2 * FBANK_STD)).unsqueeze(-1).cpu().numpy()
