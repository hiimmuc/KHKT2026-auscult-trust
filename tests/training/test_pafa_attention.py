"""Token-attention alignment must retain classification and match released PAFA gradients."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from src.training.config import parse_args
from src.models.encoders.beats import BeatsEncoder
from src.training.methods import get_method
from src.training.methods.base import Batch, StepContext
from src.models.upstream import load_reference


class TokenEncoder(nn.Module):
    final_feat_dim = 8

    def __init__(self):
        super().__init__()
        self.features = nn.Linear(8, 8)
        self.mlp_head = nn.Linear(8, 4)

    def forward(self, x, return_tokens=False):
        tokens = self.features(x)
        return tokens if return_tokens else tokens.mean(1)


class Transformer(nn.Module):
    def forward(self, x, padding_mask=None):
        return x, None


def test_beats_token_output_preserves_default_pooling_and_checkpoint_keys():
    encoder = BeatsEncoder.__new__(BeatsEncoder)
    nn.Module.__init__(encoder)
    encoder.beats = nn.Module()
    encoder.beats.patch_embedding = nn.Conv2d(1, 8, 2, stride=2)
    encoder.beats.layer_norm = nn.LayerNorm(8)
    encoder.beats.post_extract_proj = None
    encoder.beats.dropout_input = nn.Identity()
    encoder.beats.encoder = Transformer()
    images = torch.randn(4, 1, 6, 4)
    before = deepcopy(encoder.state_dict())
    tokens = encoder(images, return_tokens=True)
    assert tokens.shape == (4, 6, 8)
    torch.testing.assert_close(encoder(images), tokens.mean(1))
    assert before.keys() == encoder.state_dict().keys()
    encoder.load_state_dict(before, strict=True)


def test_token_attention_method_matches_released_loss_and_encoder_gradients():
    torch.manual_seed(17)
    args = parse_args(['--encoder', 'beats', '--method', 'ce_pafa_attn'])
    method = get_method(args.method)
    model = TokenEncoder()
    parts = method.components(model, args)
    reference_model, reference_head = deepcopy(model), deepcopy(parts.classifier)
    reference_projector = load_reference('PAFA', 'method/pafa.py').ProjectionHead(8, None, 8, attention=True, norm_type='ln')
    reference_projector.load_state_dict(parts.projector.state_dict(), strict=True)
    images, labels, patients = torch.randn(4, 5, 8), torch.arange(4), torch.tensor([0, 0, 1, 1])
    batch = Batch(images, labels, torch.zeros(4, dtype=torch.long), patients, torch.zeros(4))
    context = StepContext(model, parts.classifier, parts.projector, parts.losses, args, 1, 0, 1)
    actual = method.step(context, batch)
    tokens = reference_model(images, return_tokens=True)
    logits = reference_head(tokens).mean(1)
    reference_loss = nn.functional.cross_entropy(logits, labels) + load_reference('PAFA', 'method/pafa.py').PAFALoss()(
        reference_projector(tokens), patients, lambda_pcsl=args.lambda_pcsl, lambda_gpal=args.lambda_gpal)
    torch.testing.assert_close(actual.logits, logits)
    torch.testing.assert_close(actual.loss, reference_loss)
    actual.loss.backward()
    reference_loss.backward()
    torch.testing.assert_close(model.features.weight.grad, reference_model.features.weight.grad)
    assert parts.projector.temporal_attn[0].weight.grad.norm() > 0
    assert not get_method('ce_pafa').projector(model, args).attention


def test_token_attention_rejects_encoders_without_token_support(capsys):
    with pytest.raises(SystemExit):
        parse_args(['--encoder', 'ast', '--method', 'ce_pafa_attn'])
    assert 'requires BEATs' in capsys.readouterr().err
