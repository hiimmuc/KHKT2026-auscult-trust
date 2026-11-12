from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.processing import cycle_correction
from src.training import dataset


def make_dataset(selection='cv', profile='target'):
    ds = object.__new__(dataset.ICBHIDataset)
    ds.train_flag, ds.lodo = True, None
    ds.devices = [0, 0, 1, 1]
    ds.patients = np.array(['1', '2', '3', '4'])
    ds.args = SimpleNamespace(selection=selection, cv_folds=3, cv_fold=0,
        sc_reference='arithmetic', sc_mode='static', sc_limit_freq_low=50,
        sc_limit_freq_high=2000, sc_limit_freq_diff=20, sc_test_profile=profile,
        sc_only_device=None)
    return ds


@pytest.mark.parametrize('profile', ['target', 'source'])
def test_cv_sc_excludes_validation_from_training_reference_and_coefficients(monkeypatch, profile):
    monkeypatch.setattr(dataset, 'cv_split', lambda *a: (np.array([0, 2]), np.array([1, 3])))
    monkeypatch.setattr(cycle_correction, 'mean_spectrum', lambda waves: np.full(513, np.mean(waves)))
    monkeypatch.setattr(cycle_correction, 'apply_spectrum_correction', lambda wave, coef: wave * coef[10])
    ds = make_dataset(profile=profile)
    def correct(scale):
        return ds._correct([torch.full((1, 4), x) for x in (2., 20.*scale, 4., 40.*scale)], None)
    first, changed = correct(1), correct(10)
    np.testing.assert_allclose(first['reference'], 3.)
    np.testing.assert_array_equal(first['reference'], changed['reference'])
    for d in (0, 1):
        np.testing.assert_array_equal(first['coefficients'][d], changed['coefficients'][d])
        np.testing.assert_allclose(first['device_spectra'][d], (2., 4.)[d])
    assert [first['audio'][i].mean().item() for i in (0, 2)] == [3., 3.]
    expected = [3., 3.] if profile == 'target' else [30., 30.]
    np.testing.assert_allclose([first['audio'][i].mean().item() for i in (1, 3)], expected)


def test_fixed_sc_retains_full_training_pool(monkeypatch):
    monkeypatch.setattr(cycle_correction, 'mean_spectrum', lambda waves: np.full(513, np.mean(waves)))
    monkeypatch.setattr(cycle_correction, 'apply_spectrum_correction', lambda wave, coef: wave)
    state = make_dataset(selection='fixed')._correct(
        [torch.full((1, 4), x) for x in (2., 20., 4., 40.)], None)
    np.testing.assert_allclose(state['reference'], 16.5)


def test_cv_sc_persists_validation_state_and_patient_boundary(tmp_path):
    from src.training.main import save_sc_state
    train = SimpleNamespace(sc={'reference': np.ones(513), 'device_spectra': {0: np.ones(513)},
        'coefficients': {0: np.ones(513)}, 'validation_spectra': {0: np.full(513, 2.)},
        'validation_coefficients': {0: np.full(513, .5)},
        'reference_patients': np.array(['1']), 'validation_patients': np.array(['2'])})
    test = SimpleNamespace(sc={'device_spectra': {0: np.ones(513)}, 'coefficients': {0: np.ones(513)}})
    save_sc_state(tmp_path, train, test, SimpleNamespace(sc_mode='static'))
    with np.load(tmp_path / 'sc_state.npz') as state:
        np.testing.assert_array_equal(state['reference_patients'], ['1'])
        np.testing.assert_array_equal(state['validation_patients'], ['2'])
        np.testing.assert_array_equal(state['validation_coefficients_0'], np.full(513, .5))


def test_lodo_reference_and_training_coefficients_exclude_calibration(monkeypatch):
    monkeypatch.setattr(cycle_correction, 'mean_spectrum', lambda waves: np.full(513, np.mean(waves)))
    monkeypatch.setattr(cycle_correction, 'apply_spectrum_correction', lambda wave, coef: wave * coef[10])
    ds = make_dataset(selection='fixed')
    ds.devices = [0, 0, 1, 1, 2, 0]
    ds.patients = np.array(['1', '2', '3', '4', '5', '6'])
    ds.args.holdout_device = dataset.DEVICES[2]
    ds.lodo = (np.array([0, 2]), np.array([1, 3]), np.array([4]))
    def correct(scale):
        return ds._correct([torch.full((1, 4), x) for x in
            (2., 200.*scale, 4., 400.*scale, 10., 10000.)], None)
    first, changed = correct(1), correct(10)
    np.testing.assert_allclose(first['reference'], 3.)
    for d in (0, 1, 2):
        np.testing.assert_array_equal(first['coefficients'][d], changed['coefficients'][d])
    np.testing.assert_allclose([first['audio'][i].mean().item() for i in (0, 1, 2, 3, 4)], 3.)
    np.testing.assert_array_equal(first['reference_patients'], ['1', '3'])
    np.testing.assert_array_equal(first['calibration_patients'], ['2', '4'])
