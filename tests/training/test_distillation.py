from types import SimpleNamespace

import numpy as np
import pytest
import torch

from auscult_trust.provenance import record_prediction
from auscult_trust.training.distillation import aligned_targets
from auscult_trust.training.methods.base import Batch, StepContext
from auscult_trust.training.methods.pafa import Method


def test_targets_are_exactly_the_training_subset_and_verified(tmp_path):
    path = tmp_path / 'targets.npz'
    ids = np.array(['a:0', 'a:1', 'b:0'])
    labels, patients, devices = np.array([0, 1, 2]), np.array(['1', '1', '2']), np.array([0, 0, 1])
    rows = np.array([0, 1])
    np.savez(path, cycle_id=ids[rows], labels=labels[rows], patient=patients[rows], device=devices[rows],
             probs=np.full((2, 4), .25))
    record_prediction(path, {'files': {}})
    out = aligned_targets(path, ids, labels, patients, devices, rows)
    np.testing.assert_array_equal(out[2], 0)
    with pytest.raises(ValueError, match='alignment'):
        aligned_targets(path, ids, labels, patients, devices, np.array([0, 2]))
    path.write_bytes(b'tampered')
    with pytest.raises(ValueError, match='integrity'):
        aligned_targets(path, ids, labels, patients, devices, rows)


def test_pafa_distillation_matches_soft_cross_entropy_and_backpropagates():
    features = torch.tensor([[1., 2.], [2., 3.]])
    head = torch.nn.Linear(2, 4)
    targets = torch.tensor([[.4, .3, .2, .1], [.1, .2, .3, .4]])
    args = SimpleNamespace(freeze_encoder=False, lambda_pcsl=50., lambda_gpal=.0005)
    context = StepContext(torch.nn.Identity(), head, torch.nn.Identity(),
                          [torch.nn.CrossEntropyLoss(), lambda *a, **kw: torch.tensor(0.)], args, 1, 0, 1)
    batch = Batch(features, torch.tensor([0, 3]), torch.zeros(2), torch.arange(2), torch.zeros(2), targets)
    result = Method().step(context, batch)
    expected = -(targets * torch.log_softmax(head(features), 1)).sum(1).mean()
    torch.testing.assert_close(result.loss, expected)
    result.loss.backward()
    assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all()


def test_refined_objective_mixes_hard_labels_and_temperature_kl():
    from auscult_trust.training.methods.pafa import distillation_loss
    logits = torch.tensor([[2., -.5, .3, 1.]], requires_grad=True)
    teacher = torch.tensor([[.6, .2, .15, .05]])
    labels = torch.tensor([3])
    temperature = 2.
    softened = teacher.pow(1 / temperature)
    softened = softened / softened.sum(1, keepdim=True)
    expected = (.5 * torch.nn.functional.cross_entropy(logits, labels)
                + .5 * temperature**2 * torch.nn.functional.kl_div(
                    torch.log_softmax(logits / temperature, 1), softened, reduction='batchmean'))
    actual = distillation_loss(logits, labels, teacher, .5, temperature)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(logits.grad).all()


def test_teacher_rows_exclude_outer_validation_patients(monkeypatch):
    from auscult_trust.training import dataset
    from auscult_trust.training.distillation import training_rows
    ds = SimpleNamespace(args=SimpleNamespace(selection='cv', cv_folds=3), lodo=None,
                         labels=[0, 0, 1, 2], patients=np.array(['a', 'a', 'b', 'c']))
    monkeypatch.setattr(dataset, '_validation_indices', lambda ds: (np.array([0, 1, 3]), np.array([2])))
    np.testing.assert_array_equal(training_rows(ds), [0, 1, 3])
    assert not set(ds.patients[training_rows(ds)]) & {'b'}
    ds.args.selection = 'fixed'
    np.testing.assert_array_equal(training_rows(ds), [0, 1, 2, 3])
    ds.lodo = (np.array([0, 1]), np.array([2]), np.array([3]))
    np.testing.assert_array_equal(training_rows(ds), [0, 1])


def test_teacher_checkpoint_must_match_epoch_and_source_fold(tmp_path):
    import json
    from auscult_trust.training.distillation import validate_teacher
    args = SimpleNamespace(seed=0, selection='cv', cv_folds=3, cv_fold=1,
                           holdout_device=None, spectrum_correction=False, sc_mode='dynamic', method='ce_pafa',
                           soft_targets=None)
    (tmp_path / 'report.json').write_text(json.dumps({'history': [{'epoch': 66}]}))
    checkpoint = tmp_path / 'snapshot_epoch_66.pth'
    torch.save({'epoch': 66, 'args': vars(args)}, checkpoint)
    validate_teacher(tmp_path, checkpoint, args, 66)
    with pytest.raises(ValueError, match='epoch'):
        validate_teacher(tmp_path, checkpoint, args, 65)
    args.cv_fold = 2
    with pytest.raises(ValueError, match='cohort'):
        validate_teacher(tmp_path, checkpoint, args, 66)


def test_refined_kd_retains_observed_labels_and_patient_alignment():
    from auscult_trust.training.methods.pafa import distillation_loss
    features = torch.tensor([[1., 2.], [2., 3.]])
    labels, patients = torch.tensor([0, 3]), torch.tensor([12, 15])
    targets = torch.tensor([[.1, .7, .1, .1], [.7, .1, .1, .1]])
    captured = []
    def alignment(embedding, patient_ids, **kwargs):
        captured.append(patient_ids.clone())
        return embedding.sum() * 0
    head = torch.nn.Linear(2, 4)
    args = SimpleNamespace(freeze_encoder=False, lambda_pcsl=50., lambda_gpal=.0005,
                           kd_hard_weight=.5, kd_temperature=2.)
    context = StepContext(torch.nn.Identity(), head, torch.nn.Identity(),
                          [torch.nn.CrossEntropyLoss(), alignment], args, 1, 0, 1)
    batch = Batch(features, labels, torch.zeros(2), patients, torch.zeros(2), targets)
    result = Method().step(context, batch)
    torch.testing.assert_close(result.loss, distillation_loss(head(features), labels, targets, .5, 2.))
    torch.testing.assert_close(result.labels, labels)
    torch.testing.assert_close(captured[0], patients)


def test_refined_config_and_snapshot_validation():
    from auscult_trust.training.config import parse_args
    args = parse_args(['--method', 'ce_pafa', '--soft_targets', 'teacher.npz',
                       '--kd_hard_weight', '.5', '--kd_temperature', '2', '--snapshot_epochs', '66'])
    assert args.snapshot_epoch_list == [66]
    assert args.kd_hard_weight == .5 and args.kd_temperature == 2.
    for flags in (['--kd_temperature', 'nan'], ['--kd_hard_weight', '1.1'],
                  ['--kd_temperature', '2'], ['--snapshot_epochs', '0'], ['--snapshot_epochs', 'invalid']):
        with pytest.raises(SystemExit):
            parse_args(flags)


def test_single_teacher_export_certifies_only_outer_training_rows(tmp_path, monkeypatch):
    import json
    from auscult_trust.training import distillation, dataset
    from auscult_trust.processing import cycles
    folder = tmp_path / 'data'
    audio = folder / 'audio_test_data'
    audio.mkdir(parents=True)
    (folder / 'official_split.txt').write_text('source split')
    pretrained = tmp_path / 'models/beats/BEATs_iter3_plus_AS2M.pt'
    pretrained.parent.mkdir(parents=True)
    pretrained.write_bytes(b'pretrained')
    unit = tmp_path / 'teacher'
    unit.mkdir()
    args = SimpleNamespace(seed=0, selection='cv', cv_folds=3, cv_fold=1,
                           holdout_device=None, spectrum_correction=False, sc_mode='dynamic',
                           method='ce_pafa', soft_targets=None, data_folder=str(folder))
    (unit / 'train_args.json').write_text(json.dumps(vars(args)))
    (unit / 'report.json').write_text(json.dumps({'history': [{'epoch': 66}]}))
    checkpoint = unit / 'snapshot_epoch_66.pth'
    torch.save({'epoch': 66, 'args': vars(args)}, checkpoint)
    ds = SimpleNamespace(args=args, lodo=None, filenames=['a', 'b', 'c'],
                         labels=[0, 1, 2, 3], devices=[0, 0, 1, 2],
                         patients=np.array(['a', 'a', 'b', 'c']),
                         images=[np.ones((2, 2, 1), np.float32) * i for i in range(4)])
    monkeypatch.setattr(dataset, 'ICBHIDataset', lambda *a, **kw: ds)
    monkeypatch.setattr(dataset, '_validation_indices', lambda ds: (np.array([0, 1, 3]), np.array([2])))
    monkeypatch.setattr(cycles, 'read_annotations', lambda *a: {'a': [0, 1], 'b': [0], 'c': [0]})
    monkeypatch.setattr(distillation, 'load_args', lambda *a: args)
    monkeypatch.setattr(distillation, 'load_frozen', lambda *a: (torch.nn.Flatten(), torch.nn.Linear(4, 4)))
    monkeypatch.setattr(distillation, 'DATA', tmp_path)
    monkeypatch.setattr(distillation, 'THIRD_PARTY', tmp_path / 'refs')
    out = tmp_path / 'targets.npz'
    distillation.export(SimpleNamespace(units=[unit], checkpoints=[checkpoint], out=out, expected_epoch=66))
    with np.load(out) as targets:
        assert targets['cycle_id'].tolist() == ['a:0', 'a:1', 'c:0']
        assert targets['patient'].tolist() == ['a', 'a', 'c']
    certificate = json.loads((tmp_path / 'targets.npz.provenance.json').read_text())
    teacher = certificate['inputs']['teachers'][0]
    assert teacher['in_sample_targets'] is True
    assert teacher['excluded_patients'] == ['b']
    assert teacher['training_patients'] == ['a', 'c']
    ids = np.array(['a:0', 'a:1', 'b:0', 'c:0'])
    aligned_targets(out, ids, np.array(ds.labels), ds.patients, np.array(ds.devices), np.array([0, 1, 3]))
    with pytest.raises(ValueError, match='alignment'):
        aligned_targets(out, ids, np.array(ds.labels), ds.patients, np.array(ds.devices), np.arange(4))
