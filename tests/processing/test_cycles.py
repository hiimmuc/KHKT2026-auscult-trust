import numpy as np
import pytest
import torch
from torchaudio import transforms as T

from auscult_trust.processing.cycles import FADE_RATIO, prepare_recording


@pytest.mark.parametrize("sample_rate", [4000, 16000, 44100])
def test_shared_preparation_preserves_resample_then_fade_order(sample_rate):
    data = torch.from_numpy(np.random.default_rng(4).normal(size=(1, sample_rate)).astype(np.float32))
    expected = T.Resample(sample_rate, 16000)(data) if sample_rate != 16000 else data
    fade = int(16000 / FADE_RATIO)
    expected = T.Fade(fade_in_len=fade, fade_out_len=fade, fade_shape="linear")(expected)
    assert torch.equal(prepare_recording(data, sample_rate), expected)
