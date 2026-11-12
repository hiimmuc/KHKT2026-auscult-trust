"""Observed event targets, shared gradients, and four-class checkpoint inference."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from auscult_trust.training.methods.base import Batch, StepContext
from auscult_trust.models.heads import EventClassifier
from auscult_trust.training.methods.event_pafa import Method, event_targets
from auscult_trust.training.methods.pafa import Method as PafaMethod


class Encoder(nn.Module):
    final_feat_dim = 3

    def __init__(self):
        super().__init__()
        self.features = nn.Linear(4, 3)
        self.mlp_head = nn.Linear(3, 4)
        self.calls = 0

    def forward(self, images):
        self.calls += 1
        return self.features(images)


class Alignment(nn.Module):
    def forward(self, features, patients, lambda_pcsl, lambda_gpal):
        return features.square().mean() * (lambda_pcsl + lambda_gpal)


def context(frozen=False):
    model = Encoder()
    args = SimpleNamespace(n_cls=4, class_pool='mean', soft_targets=None,
                           freeze_encoder=frozen, lambda_pcsl=.1, lambda_gpal=.01, event_weight=.25)
    classifier = Method().classifier(model, args)
    return StepContext(model, classifier, nn.Linear(3, 3), [nn.CrossEntropyLoss(), Alignment()],
                       args, 1, 0, 1)


def batch():
    return Batch(torch.randn(4, 4), torch.arange(4), torch.zeros(4),
                 torch.tensor([101, 102, 101, 102]), torch.zeros(4))


def test_crackle_wheeze_targets_do_not_confuse_wheeze_with_both():
    assert event_targets(torch.arange(4)).tolist() == [[0., 0.], [1., 0.], [0., 1.], [1., 1.]]
    for labels in [torch.tensor([-1]), torch.tensor([4]), torch.zeros(2, 2, dtype=torch.long)]:
        with pytest.raises(ValueError, match='four-class'):
            event_targets(labels)


def test_joint_loss_reuses_one_encoder_and_adds_macro_event_bce():
    torch.manual_seed(9)
    c, b = context(), batch()
    result = Method().step(c, b)
    assert c.model.calls == 1 and result.logits.shape == (4, 4)
    features = c.model.features(b.images)
    proxy = SimpleNamespace(**vars(c), encode=lambda _: features)
    baseline = PafaMethod().step(proxy, b)
    event_bce = F.binary_cross_entropy_with_logits(c.classifier.event_head(features), event_targets(b.labels), reduction='none')
    expected = baseline.loss + .25 * (event_bce[:, 0].mean() + event_bce[:, 1].mean()) / 2
    torch.testing.assert_close(result.loss, expected)
    result.loss.backward()
    for module in [c.model.features, c.projector, c.classifier.cycle_head, c.classifier.event_head]:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        assert sum(p.grad.abs().sum() for p in module.parameters()) > 0


def test_zero_event_weight_preserves_exact_pafa_objective():
    c, b = context(), batch()
    c.args.event_weight = 0.
    result = Method().step(c, b)
    proxy = SimpleNamespace(**vars(c), encode=lambda _: c.model.features(b.images))
    expected = PafaMethod().step(proxy, b)
    torch.testing.assert_close(result.loss, expected.loss)
    torch.testing.assert_close(result.logits, expected.logits)


def test_frozen_encoder_still_trains_both_prediction_heads():
    c, b = context(frozen=True), batch()
    result = Method().step(c, b)
    result.loss.backward()
    assert all(p.grad is None for p in c.model.features.parameters())
    for module in [c.classifier.cycle_head, c.classifier.event_head]:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())


def test_classifier_state_roundtrip_keeps_four_class_probabilities(tmp_path):
    c = context()
    images = torch.randn(5, 3)
    path = tmp_path / 'classifier.pth'
    torch.save(c.classifier.state_dict(), path)
    restored = EventClassifier(nn.Linear(3, 4), 3)
    restored.load_state_dict(torch.load(path, weights_only=True))
    expected = Method().probabilities(c.classifier(images))
    actual = Method().probabilities(restored(images))
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (5, 4)
    torch.testing.assert_close(actual.sum(1), torch.ones(5))
    torch.testing.assert_close(restored.event_head(images), c.classifier.event_head(images))


def test_event_method_rejects_token_pooling_and_soft_targets():
    c, b = context(), batch()
    with pytest.raises(ValueError, match='teacher'):
        Method().step(c, replace(b, soft_targets=torch.full((4, 4), .25)))
    c.args.class_pool = 'attention'
    with pytest.raises(ValueError, match='mean-pooled'):
        Method().classifier(c.model, c.args)
    c.args.class_pool, c.args.soft_targets = 'mean', 'teacher.npz'
    with pytest.raises(ValueError, match='observed'):
        Method().classifier(c.model, c.args)
