"""Spectrum correction of a set of ICBHI cycles: which cycles define each device spectrum, the reference and the target."""
from collections import defaultdict

import numpy as np
import torch

from src.processing.correction import (
    apply_spectrum_correction,
    mean_spectrum,
    reference_spectrum,
    spectrum_coefficients,
)
from src.processing.icbhi import DEVICES


def correct_cycles(audio, devices, patients, args, *, train_flag, sc_reference, lodo=None, cv_rows=None):
    """Spectrum-correct every cycle.

    Args:
        audio: List of (1, n) cycle tensors.
        devices: Device id per cycle.
        patients: Patient id per cycle.
        args: Run arguments (`sc_*`, `holdout_device`).
        train_flag: True for the training dataset, False for the test dataset.
        sc_reference: Device spectra of the training dataset (test dataset only).
        lodo: (train, calibration, test) cycle indices of a leave-one-device-out run, else None.
        cv_rows: (train, validation) cycle indices of a cross-validation fold, else None.

    Returns:
        Dict `device_spectra`, `reference` (s_ref), `coefficients` and the corrected `audio`; with held-out cycles also
        their spectra, coefficients and patients.
    """
    used = None
    validation = set()
    reference_rows = None
    partition = "validation"
    if lodo is not None:
        train_rows, cal_rows, test_rows = lodo
        used = set(np.concatenate((train_rows, test_rows)).tolist())
        validation = set(cal_rows.tolist())
        reference_rows = train_rows
        partition = "calibration"
    elif cv_rows is not None:
        train_rows, val_rows = cv_rows
        used, validation = set(train_rows.tolist()), set(val_rows.tolist())
        reference_rows = train_rows
    by_dev = defaultdict(list)
    val_by_dev = defaultdict(list)
    for i, (a, d) in enumerate(zip(audio, devices)):
        if used is None or i in used:
            by_dev[d].append(a.numpy()[0])
        if i in validation:
            val_by_dev[d].append(a.numpy()[0])
    spectra = {d: mean_spectrum(w) for d, w in by_dev.items()}
    if lodo is not None:  # reference = mean of the source devices; the held-out device is corrected towards it
        held = DEVICES.index(args.holdout_device)
        ref_spectra = {d: s for d, s in spectra.items() if d != held}
    else:
        ref_spectra = spectra if train_flag else sc_reference
    a = args
    coef = spectrum_coefficients(
        spectra,
        reference=a.sc_reference,
        sc_mode=a.sc_mode,
        limit_freq_low=a.sc_limit_freq_low,
        limit_freq_high=a.sc_limit_freq_high,
        limit_freq_diff=a.sc_limit_freq_diff,
        reference_spectra=ref_spectra,
    )
    if not train_flag and lodo is None and getattr(a, "sc_test_profile", "target") == "source":
        # inductive: each test device uses the coefficient estimated on the training clips of the same device
        coef = spectrum_coefficients(
            sc_reference,
            reference=a.sc_reference,
            sc_mode=a.sc_mode,
            limit_freq_low=a.sc_limit_freq_low,
            limit_freq_high=a.sc_limit_freq_high,
            limit_freq_diff=a.sc_limit_freq_diff,
            reference_spectra=sc_reference,
        )
    only = getattr(a, "sc_only_device", None)
    val_spectra = {d: mean_spectrum(w) for d, w in val_by_dev.items()}
    val_coef = coef
    if validation and getattr(a, "sc_test_profile", "target") == "target":
        val_coef = spectrum_coefficients(
            val_spectra, reference=a.sc_reference, sc_mode=a.sc_mode,
            limit_freq_low=a.sc_limit_freq_low, limit_freq_high=a.sc_limit_freq_high,
            limit_freq_diff=a.sc_limit_freq_diff, reference_spectra=ref_spectra,
        )
    apply = {d for d in coef if only is None or d == DEVICES.index(only)}
    val_apply = {d for d in val_coef if only is None or d == DEVICES.index(only)}
    corrected = [
        torch.from_numpy(apply_spectrum_correction(
            w.numpy()[0], (val_coef if i in validation else coef)[d]).astype(np.float32))[None]
        if d in (val_apply if i in validation else apply)
        else w  # devices outside --sc_only_device stay raw (not even an STFT round trip)
        for i, (w, d) in enumerate(zip(audio, devices))
    ]
    return {
        "device_spectra": spectra,
        "reference": reference_spectrum(ref_spectra, a.sc_reference),
        "coefficients": coef,
        "audio": corrected,
        **({partition + "_spectra": val_spectra, partition + "_coefficients": val_coef,
            "reference_patients": np.unique(patients[reference_rows]),
            partition + "_patients": np.unique(patients[list(validation)])} if validation else {}),
    }
