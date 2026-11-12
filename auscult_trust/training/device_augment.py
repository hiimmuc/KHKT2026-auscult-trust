"""Waveform bandwidth/noise augmentation before SC and encoder preprocessing."""
from functools import lru_cache

import numpy as np
import torch
import torchaudio.functional as AF
from torch.utils.data import Dataset

from .recording_bags import source_indices


def band_noise(wave, sample_rate, target_rate, snr_db, generator=None):
    """Simulate finite acquisition bandwidth and additive noise at an exact RMS SNR.

    Down/up sampling includes anti-alias filtering. Duration and signal gain stay fixed;
    noise is bandwidth-limited with the same acquisition rate. Silence stays silence.
    """
    if (wave.ndim != 2 or not wave.is_floating_point() or wave.shape[-1] == 0 or
            not torch.isfinite(wave).all()):
        raise ValueError("Expected a finite channel-first floating waveform")
    if not 0 < target_rate <= sample_rate or not np.isfinite(snr_db):
        raise ValueError("Invalid acquisition rate or SNR")
    n = wave.shape[-1]
    def bandwidth(x):
        if target_rate == sample_rate:
            return x.clone()
        return AF.resample(AF.resample(x, sample_rate, target_rate), target_rate, sample_rate)[..., :n]
    signal = bandwidth(wave)
    noise = bandwidth(torch.randn(wave.shape, device=wave.device, dtype=wave.dtype, generator=generator))
    rms = signal.square().mean().sqrt()
    noise_rms = noise.square().mean().sqrt().clamp_min(torch.finfo(wave.dtype).eps)
    return signal + noise * (rms / noise_rms) * 10 ** (-snr_db / 20)


class DeviceAugmentedCycles(Dataset):
    """Training-only clean/augmented views; preserves the selected source-cycle pool.

    Use num_workers=0 because encoder preprocessing may run on the GPU. Raw audio is
    loaded lazily, retaining only two recordings, rather than bypassing fbank caches.
    """
    def __init__(self, source, args):
        from auscult_trust.processing.cycles import read_annotations
        self.dataset, self.indices = source_indices(source)
        self.args = args
        if not self.dataset.train_flag:
            raise ValueError("Device augmentation is training-only")
        self.wav_dir = f"{args.data_folder}/audio_test_data"
        self.annotations = read_annotations(self.wav_dir)
        self.locations = [(stem, j) for stem in self.dataset.filenames
                          for j in range(len(self.annotations[stem]))]
        if len(self.locations) != len(self.dataset):
            raise ValueError("Raw cycle annotations do not align with cached images")
        if not 0 <= args.device_augment_prob <= 1:
            raise ValueError("device_augment_prob must be in [0, 1]")
        if not 0 < args.device_rate_low <= args.device_rate_high <= args.sample_rate:
            raise ValueError("Device acquisition rates must be within the source sample rate")
        if not np.isfinite([args.device_snr_low, args.device_snr_high]).all() or args.device_snr_low > args.device_snr_high:
            raise ValueError("Invalid noise SNR interval")

    def __len__(self):
        return len(self.indices)

    @lru_cache(maxsize=2)
    def _waves(self, stem):
        from auscult_trust.processing.cycles import individual_cycles
        return individual_cycles(self.annotations[stem], self.wav_dir, stem, self.args)

    def __getitem__(self, index):
        from auscult_trust.processing.correction import apply_spectrum_correction
        from .augment import random_bin_gain_image
        i = int(self.indices[index])
        clean, label, device, patient, quality, *extra = self.dataset[i]
        if np.random.random() >= self.args.device_augment_prob:
            augmented = clean.clone() if torch.is_tensor(clean) else np.array(clean, copy=True)
        else:
            stem, cycle = self.locations[i]
            wave, raw_label = self._waves(stem)[cycle]
            if raw_label != label:
                raise ValueError("Raw cycle and cached cycle labels differ")
            # Discrete rates keep rational resampling filters small and reproducible.
            rate = int(np.random.choice([self.args.device_rate_low,
                                         (self.args.device_rate_low + self.args.device_rate_high) // 2,
                                         self.args.device_rate_high]))
            snr = np.random.uniform(self.args.device_snr_low, self.args.device_snr_high)
            wave = band_noise(wave, self.args.sample_rate, rate, snr)
            if self.dataset.sc is not None:
                if getattr(self.args, "sc_only_device", None) is not None:
                    raise ValueError("Device augmentation requires SC on all source devices")
                coef = self.dataset.sc["coefficients"][device]
                wave = torch.from_numpy(apply_spectrum_correction(wave.numpy()[0], coef).astype(np.float32))[None]
            image = self.dataset.encoder.preprocess(wave)
            if self.args.random_gain_db > 0:
                image = random_bin_gain_image(image, self.args.random_gain_db, self.dataset.encoder.db_scale,
                                              np.random.default_rng(np.random.randint(2**32)))
            augmented = self.dataset.transform(image) if self.dataset.transform is not None else image
        return (torch.stack([torch.as_tensor(clean), torch.as_tensor(augmented)]),
                label, device, patient, quality, *extra)


def wrap_training_dataset(source, args):
    return DeviceAugmentedCycles(source, args)
