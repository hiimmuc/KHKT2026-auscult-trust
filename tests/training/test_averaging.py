
import pytest
import torch

from auscult_trust.training import dataset as dataset_module
from auscult_trust.training.averaging import EpochAverager


def modules(value):
    result = [torch.nn.Linear(1, 1, bias=False) for _ in range(3)]
    for i, module in enumerate(result):
        module.weight.data.fill_(value + i)
        module.register_buffer('steps', torch.tensor(value, dtype=torch.long))
    return result


def test_average_schedule_resume_and_restores_training_weights(tmp_path):
    current = modules(2)
    average = EpochAverager(start=2, interval=2)
    assert not average.update(1, current)
    assert average.update(2, current)
    saved = tmp_path / 'last.pth'
    torch.save(average.state_dict(), saved)
    resumed = EpochAverager(start=2, interval=2)
    resumed.load_state_dict(torch.load(saved, weights_only=False))
    current = modules(6)
    assert not resumed.update(3, current)
    assert resumed.update(4, current)
    with resumed.applied(current):
        assert [m.weight.item() for m in current] == [4, 5, 6]
        assert [m.steps.item() for m in current] == [6, 6, 6]
        export = tmp_path / 'report_epoch_4.pth'
        torch.save({'model': current[0].state_dict(), 'classifier': current[1].state_dict(),
                    'projector': current[2].state_dict(), 'swa': resumed.metadata()}, export)
    assert [m.weight.item() for m in current] == [6, 7, 8]
    checkpoint = torch.load(export, weights_only=False)
    assert checkpoint['swa']['epochs'] == [2, 4]
    assert 'optimizer' not in checkpoint
    for i, name in enumerate(('model', 'classifier', 'projector')):
        assert checkpoint[name]['weight'].item() == 4 + i


def test_average_disabled_and_exception_restore():
    current = modules(1)
    disabled = EpochAverager()
    assert not disabled.update(1, current)
    with disabled.applied(current):
        assert current[0].weight.item() == 1
    average = EpochAverager(start=1)
    average.update(1, current)
    current[0].weight.data.fill_(5)
    with pytest.raises(RuntimeError), average.applied(current):
        assert current[0].weight.item() == 1
        raise RuntimeError('evaluation interrupted')
    assert current[0].weight.item() == 5


def test_average_rejects_missing_resume_state_and_batchnorm():
    with pytest.raises(ValueError, match='SWA state'):
        EpochAverager(start=1).load_state_dict(None)
    with pytest.raises(ValueError, match='BatchNorm'):
        EpochAverager(start=1).update(1, [torch.nn.BatchNorm1d(2)])


def test_training_resume_exports_average_but_restarts_from_live_weights(tmp_path, monkeypatch):
    import json
    import numpy as np
    from types import SimpleNamespace
    from auscult_trust.training import main

    args = main.build_parser().parse_args([])
    for key, value in dict(exp='train', cell='swa', unit='seed0', selection='fixed',
                           epochs=3, report_epoch_list=[3], cv_folds=0, method='ce',
                           freeze_encoder=False, swa_start=1, swa_interval=1,
                           config_files=[], stop_epoch=0, snapshot_epoch_list=[2]).items():
        setattr(args, key, value)
    out, checkpoints = tmp_path / 'output', tmp_path / 'checkpoints'
    out.mkdir()
    checkpoints.mkdir()
    dataset = SimpleNamespace(labels=[0], devices=[0], patients=[1], sc=None)
    loader = SimpleNamespace(dataset=dataset)
    monkeypatch.setattr(main, 'parse_args', lambda: args)
    monkeypatch.setattr(main.runs, 'unit_dir', lambda exp, cell, unit, kind: out if kind == 'outputs' else checkpoints)
    monkeypatch.setattr(main.runs, 'write_meta', lambda *a: None)
    monkeypatch.setattr(main, 'prediction_inputs', lambda *a: {})
    monkeypatch.setattr(main, 'build_loaders', lambda a: dataset_module.Loaders(loader, None, loader, dataset, dataset))
    def build(a):
        networks = modules(0)
        optimizer = torch.optim.SGD([p for m in networks for p in m.parameters()], lr=0.1, momentum=0.9)
        return *networks, [], optimizer
    monkeypatch.setattr(main, 'build_model', build)
    monkeypatch.setattr(main, 'adjust_learning_rate', lambda *a: None)
    interrupt = [True]
    observed = []
    def train(loader, model, classifier, projector, criterion, optimizer, epoch, args, scaler):
        observed.append((epoch, model.weight.item()))
        if epoch == 3 and interrupt[0]:
            raise RuntimeError('interrupt')
        for i, module in enumerate((model, classifier, projector)):
            module.weight.data.fill_(epoch + i)
        return 0., 100.
    monkeypatch.setattr(main, 'train_epoch', train)
    def evaluate(loader, model, classifier, args):
        score = model.weight.item()
        report = {'all': dict(sp=score, se=score, score=score, hs=score, macro_f1=0.)}
        return report, np.array([0]), np.array([[1., 0., 0., 0.]]), np.array([0]), np.array([0])
    monkeypatch.setattr(main, 'evaluate', evaluate)
    with pytest.raises(RuntimeError, match='interrupt'):
        main.main()
    restart = torch.load(checkpoints / 'last.pth', weights_only=False)
    assert restart['model']['weight'].item() == 2
    assert restart['swa']['states'][0]['weight'].item() == 1.5
    assert restart['swa']['epochs'] == [1, 2]
    snapshot = torch.load(checkpoints / 'snapshot_epoch_2.pth', weights_only=False)
    assert snapshot['epoch'] == 2
    assert snapshot['model']['weight'].item() == 1.5
    interrupt[0] = False
    del args.class_nums  # a fresh CLI parse does not contain derived training counts
    main.main()
    assert observed[-1] == (3, 2.)
    exported = torch.load(checkpoints / 'report_epoch_3.pth', weights_only=False)
    assert exported['model']['weight'].item() == 2
    assert exported['classifier']['weight'].item() == 3
    assert exported['projector']['weight'].item() == 4
    assert exported['swa']['epochs'] == [1, 2, 3]
    assert 'optimizer' not in exported
    report = json.loads((out / 'report.json').read_text())
    assert report['history'][0]['test_score'] is None
    assert report['history'][-1]['test_score'] == 2.
    assert main.completed(out, checkpoints)
