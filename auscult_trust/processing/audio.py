"""Audio loading and cycle cropping. Mono float32 at a common sample rate."""

from math import gcd

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

SR = 16000  # OPERA input rate


def load_wav(path, sr=SR):
    """Load a wav file as mono float32 resampled to `sr`.

    Args:
        path: Path to a PCM wav file.
        sr: Target sample rate in Hz.

    Returns:
        1-D float32 array scaled to [-1, 1].
    """
    rate, x = wavfile.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        x = x / float(np.iinfo(x.dtype).max)
    x = x.astype(np.float32)
    if rate != sr:
        g = gcd(rate, sr)
        x = resample_poly(x, sr // g, rate // g).astype(np.float32)
    return x


def crop_pad(x, n):
    """Crop or zero-pad a waveform to exactly `n` samples.

    Args:
        x: 1-D waveform.
        n: Target length in samples.

    Returns:
        1-D array of length `n`.
    """
    return x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))
