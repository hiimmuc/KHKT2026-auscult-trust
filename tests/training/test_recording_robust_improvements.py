"""Recording labels, split isolation, trainable heads, and physical noise controls."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset, Subset

from src.training.device_augment import band_noise, DeviceAugmentedCycles
from src.training.methods.base import Batch, StepContext
from src.training.methods.recording_pafa import Method, RecordingClassifier
from src.training.methods.device_pafa import Method as DeviceMethod
from src.training.recording_bags import RecordingBags, predict_recording, evaluate_recordings


class Cycles(Dataset):
    labels = [0, 0, 0, 2, 0, 1]
    patients = np.array([101, 101, 102, 102, 103, 103])
    devices = [0] * 6
    train_flag = True
    filenames = ['a', 'b', 'c']
    sc = None
    transform = staticmethod(lambda x: torch.from_numpy(x).permute(2, 0, 1))
    encoder = SimpleNamespace(preprocess=lambda w: np.full((2, 2, 1), w.mean().item(), np.float32))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return torch.full((1, 2, 2), float(i)), self.labels[i], self.devices[i], int(self.patients[i]), 0.


IDS = np.array(['a', 'a', 'b', 'b', 'c', 'c'])


def test_recording_labels_padding_and_patient_pool():
    ds = Cycles()
    bags = RecordingBags(Subset(Subset(ds, [0, 1, 2, 3]), [0, 1]), 4, False, IDS)
    assert bags.recording_ids.tolist() == ['a']
    image, label, _, patient, mask = bags[0]
    assert patient == 101 and label.tolist() == [0, 0, 0, 0, 0]
    assert mask.tolist() == [True, True, False, False]
    assert torch.all(image[~mask] == 0)
    abnormal = RecordingBags(Subset(ds, [2, 3]), 1, False, IDS)
    assert abnormal[0][1].tolist() == [0, 1]  # label sees abnormal cycle beyond the selected window
    with pytest.raises(ValueError, match='Partial recording'):
        RecordingBags(Subset(ds, [2]), 2, False, IDS)


def test_recordings_cannot_merge_patients():
    with pytest.raises(ValueError, match='multiple patients'):
        RecordingBags(Cycles(), 2, False, ['x'] * 6)


def test_recording_mil_mask_and_streaming_inference():
    head = RecordingClassifier(nn.Linear(3, 4), 3)
    x = torch.randn(2, 4, 3)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
    before = head.recording_logits(x, mask)
    x[~mask] = 1e6
    torch.testing.assert_close(head.recording_logits(x, mask), before)
    with pytest.raises(ValueError, match='valid window'):
        head.recording_logits(x, torch.zeros_like(mask))
    windows = torch.randn(11, 3)
    expected = head.recording_logits(windows[None]).softmax(1)[0]
    for size in [1, 4, 20]:
        torch.testing.assert_close(predict_recording(nn.Identity(), head, windows, size), expected)


class Alignment(nn.Module):
    def forward(self, features, patients, **kwargs):
        return features.square().mean() * 0.01


def context(classifier):
    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 3))
    args = SimpleNamespace(freeze_encoder=False, lambda_pcsl=1., lambda_gpal=.01,
                           recording_weight=.5, consistency_weight=.5, consistency_ramp=2)
    return StepContext(model, classifier, nn.Identity(), [nn.CrossEntropyLoss(), Alignment()],
                       args, 1, 0, 1)


def test_joint_objective_trains_shared_encoder_and_both_heads():
    torch.manual_seed(4)
    head = RecordingClassifier(nn.Linear(3, 4), 3)
    c = context(head)
    batch = Batch(torch.randn(2, 3, 1, 2, 2), torch.tensor([[0, 0, 0, 0], [0, 1, 2, 1]]),
                  torch.zeros(2), torch.tensor([101, 102]),
                  torch.tensor([[True, True, False], [True, True, True]]))
    result = Method().step(c, batch)
    assert result.logits.shape == (2, 4) and result.labels.tolist() == [0, 0]
    result.loss.backward()
    for module in [c.model, head.cycle_head, head.recording_head]:
        grads = [p.grad for p in module.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads) > 0


def test_recording_validation_rejects_training_patient_overlap(monkeypatch):
    monkeypatch.setattr('src.training.recording_bags.cycle_recordings', lambda _: IDS)
    head = RecordingClassifier(nn.Linear(3, 4), 3)
    with pytest.raises(ValueError, match='overlap'):
        evaluate_recordings(Subset(Cycles(), [0, 1]), context(head).model, head,
                            SimpleNamespace(recording_eval_cycles=2), excluded_patients=[101])


def test_waveform_noise_exact_snr_duration_and_silence():
    wave = torch.sin(torch.arange(16000)[None] * (2 * torch.pi * 300 / 16000))
    augmented = band_noise(wave, 16000, 16000, 25, torch.Generator().manual_seed(7))
    assert augmented.shape == wave.shape and torch.isfinite(augmented).all()
    snr = 10 * torch.log10(wave.square().mean() / (augmented - wave).square().mean())
    assert snr.item() == pytest.approx(25, abs=.01)
    torch.testing.assert_close(band_noise(torch.zeros_like(wave), 16000, 4000, 20), torch.zeros_like(wave))
    with pytest.raises(ValueError):
        band_noise(wave, 16000, 32000, 20)


def test_resampling_removes_out_of_band_content_without_pitch_shift():
    time = torch.arange(16000)[None] / 16000
    low, high = (torch.sin(2 * torch.pi * hz * time) for hz in [300, 6000])
    aug = band_noise(low + high, 16000, 4000, 100, torch.Generator().manual_seed(0))
    spectrum = torch.fft.rfft(aug)[0].abs()
    assert spectrum.argmax().item() == 300
    assert spectrum[6000] < spectrum[300] * .01
    assert aug.shape == low.shape


def test_device_consistency_has_finite_shared_encoder_gradient():
    c = context(nn.Linear(3, 4))
    batch = Batch(torch.randn(4, 2, 1, 2, 2), torch.tensor([0, 1, 2, 3]),
                  torch.zeros(4), torch.tensor([101, 102, 103, 104]), torch.zeros(4))
    result = DeviceMethod().step(c, batch)
    assert result.consistency is not None and result.consistency[0] >= 0
    result.loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in c.model.parameters())
    c.args.consistency_weight = 0
    assert DeviceMethod().step(c, batch).consistency is None


def test_device_wrapper_preserves_subset_and_raw_cycle_labels(monkeypatch):
    monkeypatch.setattr('src.processing.cycles.read_annotations', lambda _: {key: [0, 1] for key in ['a', 'b', 'c']})
    args = SimpleNamespace(data_folder='unused', sample_rate=16000, device_augment_prob=1.,
                           device_rate_low=4000, device_rate_high=8000,
                           device_snr_low=20., device_snr_high=35., random_gain_db=0.)
    ds = DeviceAugmentedCycles(Subset(Cycles(), [2, 3]), args)
    ds._waves = lambda _: [(torch.ones(1, 1600), 0), (torch.ones(1, 1600), 2)]
    image, label, _, patient, _ = ds[1]
    assert ds.indices.tolist() == [2, 3]
    assert image.shape == (2, 1, 2, 2) and label == 2 and patient == 102
    assert torch.all(image[0] == 3)


def test_recording_validation_cycle_baseline_is_explicit(monkeypatch):
    monkeypatch.setattr('src.training.recording_bags.cycle_recordings', lambda _: IDS)
    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 3))
    head = nn.Linear(3, 4)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
    result = evaluate_recordings(Subset(Cycles(), [0, 1]), model, head,
                                SimpleNamespace(recording_eval_cycles=2, method='ce_pafa'),
                                excluded_patients=[102])
    np.testing.assert_allclose(result[2], [[.25, .75]])
    assert result[5].tolist() == ['a'] and result[6].tolist() == [101]


def test_recording_bags_preserve_teacher_targets():
    class TeacherCycles(Cycles):
        def __getitem__(self, index):
            return (*super().__getitem__(index), np.eye(4, dtype=np.float32)[self.labels[index]])
    bags = RecordingBags(Subset(TeacherCycles(), [2, 3]), 3, False, IDS)
    values = bags[0]
    assert values[5].shape == (3, 4)
    torch.testing.assert_close(values[5][:2], torch.tensor([[1., 0., 0., 0.], [0., 0., 1., 0.]]))
    assert values[5][2].sum() == 0
