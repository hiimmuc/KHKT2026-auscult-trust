"""Learned recording inference preserves metadata and isolates filter renderings."""
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from src.evaluation.recording_head import recording_predictions
from src.models.heads import RecordingClassifier


def classifier():
    head = RecordingClassifier(nn.Linear(1, 4), 1)
    with torch.no_grad():
        head.recording_head.weight.copy_(torch.tensor([[0.], [1.]]))
        head.recording_head.bias.zero_()
        head.cycle_head.weight.zero_()
        head.cycle_head.bias.zero_()  # equal cycle probabilities cannot produce the learned head outputs
    return head


def test_true_recording_head_groups_all_windows_and_keeps_filters_separate():
    table = pd.DataFrame({'stem': ['x', 'x', 'y', 'x', 'x'],
                          'filter': ['Bell', 'Diaphragm', 'Bell', 'Bell', 'Diaphragm'],
                          'patient': ['101', '101', '102', '101', '101'],
                          'label': [1, 1, 0, 1, 1]})
    images = np.array([0., -2., -1., 4., -3.], dtype=np.float32)[:, None, None]
    result = recording_predictions(images, table, nn.Flatten(), classifier())
    assert result['recording_stem'].tolist() == ['x', 'x', 'y']
    assert result['recording_filter'].tolist() == ['Bell', 'Diaphragm', 'Bell']
    assert result['recording_patient'].tolist() == ['101', '101', '102']
    np.testing.assert_allclose(result['recording_probs'][:, 1], torch.sigmoid(torch.tensor([4., -2., -1.])).numpy())
    assert result['recording_label'].tolist() == [1, 1, 0]


def test_recording_spectral_views_share_cycle_inference_units():
    table = pd.DataFrame({'stem': ['x', 'x'], 'patient': ['101', '101'], 'device': [0, 0], 'label': [1, 1]})
    result = recording_predictions(np.zeros((2, 1, 1), np.float32), table, nn.Flatten(), classifier(),
                                   np.array([[1.], [3.]], np.float32))
    assert result['recording_probs'][0, 1] == pytest.approx(torch.sigmoid(torch.tensor(3.)).item())
    assert result['recording_device'].tolist() == [0]


def test_missing_head_or_inconsistent_recording_metadata_is_rejected():
    table = pd.DataFrame({'stem': ['x', 'x'], 'patient': ['101', '102'], 'label': [0, 0]})
    images = np.zeros((2, 1, 1), np.float32)
    with pytest.raises(ValueError, match='trained recording head'):
        recording_predictions(images, table, nn.Flatten(), nn.Linear(1, 4))
    with pytest.raises(ValueError, match='patient'):
        recording_predictions(images, table, nn.Flatten(), classifier())
