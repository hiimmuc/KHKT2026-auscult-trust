"""Classification pooling and class-conditional PAFA numerical contracts."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.models.encoders.beats import BeatsEncoder, TokenClassifier
from src.training.methods.base import Batch, StepContext
from src.training.methods.class_aware_pafa import ClassAwarePAFA, Method, centroid_pair_masks
from src.training.methods.pafa import Method as PafaMethod, alignment_loss


@pytest.mark.parametrize("pool", ["topk", "attention"])
def test_pool_padding_invariance_and_trainable_checkpoint(pool):
    torch.manual_seed(9)
    classifier = TokenClassifier(nn.Linear(3, 2), pool)
    tokens = torch.randn(2, 5, 3, requires_grad=True)
    mask = torch.tensor([[False, False, True, True, True], [False, False, False, True, True]])
    expected = classifier(tokens, mask)
    changed = tokens.detach().clone()
    changed[mask] = 10000
    torch.testing.assert_close(classifier(changed, mask), expected)
    expected.square().sum().backward()
    assert tokens.grad[mask].abs().sum() == 0
    if pool == "attention":
        assert classifier.attention.weight.grad.norm() > 0
        assert "attention.weight" in classifier.state_dict()
    restored = deepcopy(classifier)
    restored.load_state_dict(classifier.state_dict(), strict=True)
    torch.testing.assert_close(restored(tokens, mask), expected)
    with pytest.raises(ValueError, match="valid time token"):
        classifier(tokens, torch.ones_like(mask))


def test_topk_logits_formula_and_full_fraction_mean():
    tokens = torch.tensor([[[0., 10.], [4., 0.], [3., 1.], [2., 2.], [1., 3.]]])
    head = nn.Linear(2, 2, bias=False)
    head.weight.data.copy_(torch.eye(2))
    torch.testing.assert_close(TokenClassifier(head, "topk", .2)(tokens), torch.tensor([[4., 10.]]))
    torch.testing.assert_close(TokenClassifier(head, "topk", 1)(tokens), head(tokens).mean(1))


@pytest.mark.parametrize("pool", ["topk", "attention"])
def test_pafa_training_and_inference_share_checkpointed_classifier(pool):
    class Encoder(nn.Identity):
        final_feat_dim = 3
        def __init__(self):
            super().__init__()
            self.mlp_head = nn.Linear(3, 2)
    model, method = Encoder(), PafaMethod()
    args = SimpleNamespace(class_pool=pool, class_topk_fraction=.2, freeze_encoder=False,
                           lambda_pcsl=.1, lambda_gpal=.1)
    parts = method.components(model, args)
    images, labels = torch.randn(4, 5, 3), torch.tensor([0, 1, 0, 1])
    batch = Batch(images, labels, torch.zeros(4), torch.tensor([0, 0, 1, 1]), torch.zeros(4))
    context = StepContext(model, parts.classifier, parts.projector, parts.losses, args, 1, 0, 1)
    result = method.step(context, batch)
    inference_head = method.classifier(model, args)
    inference_head.load_state_dict(parts.classifier.state_dict(), strict=True)
    torch.testing.assert_close(result.logits, inference_head(model(images)))
    result.loss.backward()
    assert parts.classifier.head.weight.grad.norm() > 0
    if pool == "attention":
        assert parts.classifier.attention.weight.grad.norm() > 0


def test_beats_averages_frequency_before_classification_pooling():
    encoder = BeatsEncoder.__new__(BeatsEncoder)
    nn.Module.__init__(encoder)
    encoder.beats = nn.Module()
    encoder.beats.patch_embedding = nn.Conv2d(1, 3, 1)
    encoder.beats.layer_norm = nn.Identity()
    encoder.beats.post_extract_proj = None
    encoder.beats.dropout_input = nn.Identity()
    class Transformer(nn.Module):
        def forward(self, x, padding_mask=None):
            return x, None
    encoder.beats.encoder = Transformer()
    images = torch.randn(2, 1, 4, 2)
    tokens = encoder(images, return_tokens=True)
    torch.testing.assert_close(encoder(images), tokens.mean(1))
    encoder.class_pool = "topk"
    torch.testing.assert_close(encoder(images), tokens.reshape(2, 4, 2, 3).mean(2))


def test_class_aware_pair_masks_and_centroid_formula():
    patients = torch.tensor([0, 0, 1, 1])
    labels = torch.tensor([0, 1, 0, 1])
    positive, negative = centroid_pair_masks(patients, labels)
    assert positive.nonzero().tolist() == [[0, 2], [1, 3]]
    assert negative.nonzero().tolist() == [[0, 1], [0, 3], [1, 2], [2, 3]]
    features = torch.tensor([[0.], [10.], [2.], [14.]], requires_grad=True)
    actual = ClassAwarePAFA()(features, patients, labels, lambda_pcsl=0, lambda_gpal=1)
    # GPAL retains the released centroid formula, independently within each class.
    reference = torch.stack([alignment_loss()(features[labels == label], patients[labels == label],
        lambda_pcsl=0, lambda_gpal=1) for label in labels.unique()]).mean()
    torch.testing.assert_close(actual, reference)
    assert actual.item() == 2.5
    shifted = features.detach() + (labels * 100).unsqueeze(1)
    torch.testing.assert_close(ClassAwarePAFA()(shifted, patients, labels, 0, 1), actual)
    actual.backward()
    assert torch.isfinite(features.grad).all() and features.grad.norm() > 0


def test_class_aware_pcsl_keeps_different_class_separation():
    features = torch.tensor([[0.], [2.], [10.], [12.]], requires_grad=True)
    patients, labels = torch.tensor([0, 0, 1, 1]), torch.tensor([0, 0, 1, 1])
    actual = ClassAwarePAFA()(features, patients, labels, 1, 0)
    torch.testing.assert_close(actual, torch.tensor(4 / (100 + 1e-6)))
    actual.backward()
    assert torch.isfinite(features.grad).all()


@pytest.mark.parametrize("patients,labels", [([], []), ([0], [0]), ([0, 0], [0, 0]), ([-1, -1], [0, 1])])
def test_class_aware_undefined_groups_are_differentiable_zero(patients, labels):
    features = torch.randn(len(labels), 3, requires_grad=True)
    loss = ClassAwarePAFA()(features, torch.tensor(patients).long(), torch.tensor(labels).long())
    assert loss.item() == 0
    loss.backward()
    assert torch.isfinite(features.grad).all()


def test_class_aware_excludes_mixed_targets_from_alignment():
    model = nn.Identity()
    head, projector = nn.Linear(3, 2), nn.Linear(3, 3)
    args = SimpleNamespace(freeze_encoder=False, lambda_pcsl=1, lambda_gpal=1)
    images, labels = torch.randn(3, 3), torch.tensor([0, 0, 1])
    soft = torch.tensor([[1., 0.], [.5, .5], [0., 1.]])
    class Capture(ClassAwarePAFA):
        def forward(self, features, patients, labels, **kwargs):
            assert labels.tolist() == [0, 1]
            assert patients.tolist() == [0, 2]
            return super().forward(features, patients, labels, **kwargs)
    batch = Batch(images, labels, torch.zeros(3), torch.arange(3), torch.zeros(3), soft)
    context = StepContext(model, head, projector, [nn.CrossEntropyLoss(), Capture()], args, 1, 0, 1)
    result = Method().step(context, batch)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert head.weight.grad.norm() > 0
