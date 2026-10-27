from types import SimpleNamespace
import numpy as np
import pytest
import torch

from src.training.config import parse_args

CONFIG = 'experiments/train/base.yaml,experiments/train/encoder_beats.yaml,experiments/train/grid/pafa_beats.yaml,experiments/train/sc.yaml,experiments/train/sc_static.yaml,experiments/train/sc_gain.yaml'


def test_cv_never_constructs_official_test(monkeypatch):
    from src.training import dataset
    seen = []
    class Source(torch.utils.data.Dataset):
        def __init__(self, train_flag, transform, args, **kwargs):
            seen.append(train_flag)
            assert train_flag, 'official test must not be constructed for CV'
            self.labels = [0, 1, 2, 3]
            self.devices = [0] * 4
            self.patients = np.array(['1', '2', '3', '4'])
            self.sc = None
        def __len__(self): return 4
        def __getitem__(self, i): return torch.zeros(1, 4, 4), self.labels[i], 0, i, 0
    monkeypatch.setattr(dataset, 'ICBHIDataset', Source)
    monkeypatch.setattr(dataset, '_validation_indices', lambda ds: (np.array([0, 1]), np.array([2, 3])))
    args = parse_args(['--config', CONFIG, '--cv_folds', '3', '--num_workers', '0'])
    loaders = dataset.build_loaders(args)
    assert seen == [True]
    assert loaders[2] is None and loaders[4] is None
    assert len(loaders[0].dataset) == len(loaders[1].dataset) == 2


@pytest.mark.parametrize('recipe', ['llrd', 'patient_cap', 'class_topk', 'class_attention', 'class_aware', 'recording_head', 'band_noise', 'band_noise_consistency'])
def test_every_recipe_parses(recipe):
    args = parse_args(['--config', CONFIG + ',experiments/train/' + recipe + '.yaml', '--cv_folds', '3'])
    assert args.encoder == 'beats' and args.epochs == 100 and args.cv_folds == 3


def test_invalid_combinations_rejected():
    for extra in (['--layer_decay', '0'], ['--method', 'ce_recording_pafa', '--patient_cap'],
                  ['--method', 'ce_device_pafa', '--num_workers', '2']):
        with pytest.raises(SystemExit):
            parse_args(['--config', CONFIG, *extra])


def test_export_pools_embeddings_without_changing_token_classification(monkeypatch):
    from src.training.export_probs import run
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, **kwargs: self)
    class Encoder(torch.nn.Module):
        def forward(self, images):
            return images.flatten(2).transpose(1, 2).repeat(1, 1, 3)
    class Classifier(torch.nn.Module):
        def forward(self, tokens):
            return torch.cat([tokens.mean(1), tokens.mean(1)[:, :1]], dim=1)
    images = torch.arange(8, dtype=torch.float32).reshape(2, 1, 2, 2)
    p, embeddings, _, _ = run([(images, torch.tensor([0, 1]), torch.tensor([0, 0]))], Encoder(), Classifier())
    assert embeddings.shape == (2, 3)
    np.testing.assert_allclose(embeddings[:, 0], images.mean((1, 2, 3)))
    assert p.shape == (2, 4)
