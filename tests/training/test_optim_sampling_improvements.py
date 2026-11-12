import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from auscult_trust.training.dataset import PatientCappedSampler, _training_sampling
from auscult_trust.training.optim import adjust_learning_rate, layer_decay_parameters, set_optimizer, warmup_learning_rate


def _model():
    model = nn.Module()
    model.beats = nn.Module()
    model.beats.patch_embedding = nn.Linear(2, 2)
    model.beats.encoder = nn.Module()
    model.beats.encoder.layers = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
    model.mlp_head = nn.Linear(2, 4)
    return model


def _args():
    return SimpleNamespace(layer_decay=0.8, learning_rate=0.01, optimizer="adam", weight_decay=0,
                           cosine=True, epochs=100, lr_decay_rate=0.1, warm=True, warm_epochs=5,
                           warmup_from=0.001, warmup_to=0.01, method="ce_pafa", patient_cap=True)


def test_llrd_covers_each_trainable_parameter_once_and_preserves_head_rate():
    model, head, args = _model(), nn.Linear(2, 4), _args()
    model.beats.patch_embedding.bias.requires_grad = False
    # The trainer passes model parameters among the extras; shared parameters must not repeat.
    groups = layer_decay_parameters(args, model, list(model.parameters()) + list(head.parameters()))
    parameters = [p for group in groups for p in group["params"]]
    expected = [p for p in list(model.parameters()) + list(head.parameters()) if p.requires_grad]
    assert len(parameters) == len({id(p) for p in parameters}) == len(expected)
    assert {id(p) for p in parameters} == {id(p) for p in expected}
    scales = {id(p): g["lr_scale"] for g in groups for p in g["params"]}
    assert scales[id(model.beats.patch_embedding.weight)] == pytest.approx(0.8 ** 3)
    assert scales[id(model.beats.encoder.layers[0].weight)] == pytest.approx(0.8 ** 2)
    assert scales[id(model.beats.encoder.layers[1].weight)] == pytest.approx(0.8)
    assert scales[id(head.weight)] == scales[id(model.mlp_head.weight)] == 1


def test_llrd_ratios_survive_schedules_and_optimizer_resume():
    model, args = _model(), _args()
    optimizer = set_optimizer(args, layer_decay_parameters(args, model, []))
    scales = [group["lr_scale"] for group in optimizer.param_groups]

    def check():
        base_lr = optimizer.param_groups[-1]["lr"]
        assert [g["lr"] / base_lr for g in optimizer.param_groups] == pytest.approx(scales)

    check()
    adjust_learning_rate(args, optimizer, 1)
    warmup_learning_rate(args, 1, 0, 10, optimizer)
    check()
    adjust_learning_rate(args, optimizer, 5)
    warmup_learning_rate(args, 5, 9, 10, optimizer)
    check()
    state = copy.deepcopy(optimizer.state_dict())
    optimizer = set_optimizer(args, layer_decay_parameters(args, _model(), []))
    optimizer.load_state_dict(state)
    check()
    adjust_learning_rate(args, optimizer, 60)
    warmup_learning_rate(args, 60, 0, 10, optimizer)
    check()
    adjust_learning_rate(args, optimizer, 100)
    check()
    args.cosine, args.lr_decay_epochs = False, [10, 20]
    adjust_learning_rate(args, optimizer, 30)
    check()


def test_llrd_rejects_unsupported_model_and_invalid_decay():
    args = _args()
    with pytest.raises(ValueError, match="BEATs"):
        layer_decay_parameters(args, nn.Linear(2, 2), [])
    args.layer_decay = 0
    with pytest.raises(ValueError, match="layer_decay"):
        layer_decay_parameters(args, _model(), [])


def test_patient_sampling_fixed_count_ceiling_class_support_and_resume():
    labels = [0] * 20 + [1, 2, 3] + [0] * 3
    patients = ["dominant"] * 20 + ["rare"] * 3 + ["other"] * 3
    sampler = PatientCappedSampler(labels, patients, [0] * 20 + [1] * 6)
    saved_rng = torch.random.get_rng_state()
    try:
        torch.manual_seed(12)
        checkpoint_rng = torch.random.get_rng_state()
        first = list(sampler)
        assert len(first) == len(labels)
        _, counts = np.unique(np.asarray(patients)[first], return_counts=True)
        assert counts.max() <= sampler.cap == 9
        assert counts.max() - counts.min() <= 1
        assert set(np.asarray(labels)[first]) == {0, 1, 2, 3}
        assert sum(sampler.last_counts["class"].values()) == len(labels)
        second = list(sampler)
        assert first != second
        torch.random.set_rng_state(checkpoint_rng)
        resumed = PatientCappedSampler(labels, patients, [0] * 20 + [1] * 6)
        assert list(resumed) == first
        assert list(resumed) == second
    finally:
        torch.random.set_rng_state(saved_rng)


def test_patient_sampling_rejects_infeasible_rare_class_cap():
    with pytest.raises(ValueError, match="preserve all source"):
        PatientCappedSampler([1, 2, 3, 0, 0, 0, 0, 0], ["a", "a", "a", "b", "c", "d", "e", "f"], [0] * 8)


def test_sampler_receives_only_source_training_rows_and_keeps_last_batch():
    ds = SimpleNamespace(labels=[0, 3, 1, 2, 3], patients=np.array(["source", "test", "source", "source", "cal"]),
                         devices=[0, 9, 0, 0, 9])
    kwargs = _training_sampling(ds, np.array([0, 2, 3]), _args())
    sampler = kwargs["sampler"]
    assert set(sampler.patients) == {"source"}
    assert set(sampler.labels) == {0, 1, 2}
    assert set(sampler.devices) == {0}
    assert kwargs["drop_last"] is False
    assert set(list(sampler)).issubset({0, 1, 2})
    args = _args()
    args.patient_cap = False
    assert _training_sampling(ds, np.array([0, 2, 3]), args) == {"shuffle": True, "drop_last": True}


def test_patient_sampling_rejects_synthetic_multicycle_mode():
    args = _args()
    args.method = "pcmcl"
    with pytest.raises(ValueError, match="multi-cycle"):
        _training_sampling(None, None, args)


def test_validation_metadata_preserves_nested_subset_order(monkeypatch):
    from torch.utils.data import Subset
    from auscult_trust.training.dataset import validation_metadata
    from auscult_trust.training import recording_bags

    class Cycles:
        patients = ["a", "a", "b", "c"]

        def __len__(self):
            return len(self.patients)

    monkeypatch.setattr(recording_bags, "cycle_recordings", lambda ds: np.array(["r0", "r0", "r1", "r2"]))
    subset = Subset(Subset(Cycles(), [3, 0, 2]), [2, 0])
    patients, stems = validation_metadata(subset)
    assert patients.tolist() == ["b", "c"]
    assert stems.tolist() == ["r1", "r2"]
