"""ICBHI cycle dataset (official 60/40 split, or one 80/20 patient fold) and the training data loaders."""

import copy
import os
import random
from typing import NamedTuple

import numpy as np
import torch
from src.processing.cycle_correction import correct_cycles
from src.processing.recordings import select_recordings
from src.processing.icbhi import DEVICES
from src.processing.splits import cv_split, lodo_split, patient_val_split
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from torchvision import transforms

from .augment import SpecAugment, random_bin_gain_image
from .cache import CACHE, ImageCache
from src.processing.cycles import individual_cycles, read_annotations
from .methods.pcmcl_data import PCMCLTrain
from .methods import get_method
from src.models.encoders import get_encoder, image_shape


def validation_metadata(source_subset):
    """Patient and recording stem arrays in the exact cycle subset order."""
    from .recording_bags import cycle_recordings, source_indices
    dataset, indices = source_indices(source_subset)
    return np.asarray(dataset.patients)[indices], cycle_recordings(dataset)[indices]


def _validation_indices(ds):
    a = ds.args
    if a.cv_folds:
        return cv_split(ds.patients, np.asarray(ds.devices), a.cv_folds, a.cv_fold)
    return patient_val_split(ds.patients, a.val_frac, a.seed)


class ICBHIDataset(Dataset):
    """Cycles of the train or test split. Items: (fbank image, label, device id, patient id as int, QLung quality score).

    The split is the official 60/40 one, or with `args.test_fold` k in 0-4 the patient fold k of `patient_list_foldwise.txt`
    as test set (80/20). The quality score (spectral entropy and RMS energy of the raw cycle) is computed for `qlung` only, else 0.

    With `args.spectrum_correction` the waveform of every cycle goes through spectrum correction (SC) before the STFT: train
    clips use their own (source) device spectra, test clips their own (target) spectra, both against the train
    reference (`sc_reference` = `device_spectra` of the train dataset). The SC state is kept in `self.sc`.
    """

    def __init__(self, train_flag, transform, args, sc_reference=None, print_flag=True, audio_only=False):
        self.train_flag, self.transform, self.args = train_flag, transform, args
        self.filenames, self.split, self.dropped = select_recordings(args, train_flag)
        wav_dir = os.path.join(args.data_folder, "audio_test_data")
        annotations = read_annotations(wav_dir)

        self.encoder = get_encoder(args.encoder)
        cache = ImageCache(self.encoder, args, self.split)
        cached = None if args.spectrum_correction or audio_only else cache.load()
        if cached:
            self.images, self.labels, self.devices, self.patients = cached
            self.sc = None
            self._lodo_indices()
        else:
            audio, labels, devices, patients = [], [], [], []
            for f in self.filenames:
                for wave, label in individual_cycles(annotations[f], wav_dir, f, args):
                    audio.append(wave)
                    labels.append(label)
                    devices.append(DEVICES.index(f.split("_")[-1]))
                    patients.append(f.split("_")[0])
            self.labels, self.devices, self.patients = labels, devices, np.array(patients)
            if audio_only:  # raw cycles for inference-time pipelines; no spectrograms, no SC
                self.audio, self.sc, self.images = audio, None, None
                self._lodo_indices()
                return
            self._lodo_indices()
            self.sc = self._correct(audio, sc_reference) if args.spectrum_correction else None
            if self.sc:
                audio = self.sc.pop("audio")
            if args.spectrum_correction:
                self.images = [self.encoder.preprocess(a) for a in audio]
            else:
                self.images = cache.build(audio, self.labels, self.devices, self.patients)
        assert self.images[0].shape == (*image_shape(self.encoder, self.args), 1), self.images[0].shape
        self.quality = get_method(args.method).quality_scores(self, annotations, wav_dir, CACHE)
        self.soft_targets = None
        if train_flag and getattr(args, 'soft_targets', None):
            from .distillation import aligned_targets, training_rows
            ids = np.array([f'{stem}:{i}' for stem in self.filenames for i in range(len(annotations[stem]))])
            rows = training_rows(self)
            self.soft_targets = aligned_targets(args.soft_targets, ids, np.asarray(self.labels), self.patients,
                                                np.asarray(self.devices), rows)
        if print_flag:
            counts = np.bincount(self.labels, minlength=args.n_cls)
            print(
                "[{} dataset] {} cycles, {}".format(
                    self.split,
                    len(self.labels),
                    ", ".join(
                        "{} {} ({:.1f}%)".format(c, n, 100 * n / len(self.labels))
                        for c, n in zip(args.cls_list, counts)
                    ),
                )
            )

    def _lodo_indices(self):
        """In leave-one-device-out mode, the (train, calibration, test) cycle indices of this run's seed."""
        self.lodo = None
        if self.args.holdout_device:
            self.lodo = lodo_split(
                self.patients,
                np.asarray(self.devices),
                DEVICES.index(self.args.holdout_device),
                self.args.lodo_cal_frac,
                self.args.seed,
            )

    def _correct(self, audio, sc_reference):
        """Spectrum-correct every cycle; the held-out cycles of a LODO or CV run are corrected from their own spectra."""
        cv_rows = _validation_indices(self) if self.lodo is None and self.train_flag and self.args.selection == "cv" else None
        return correct_cycles(audio, self.devices, self.patients, self.args, train_flag=self.train_flag,
                              sc_reference=sc_reference, lodo=self.lodo, cv_rows=cv_rows)

    def __getitem__(self, index):
        image = np.array(self.images[index])  # a copy: the cached array is read-only
        if self.train_flag and self.args.random_gain_db > 0:
            image = random_bin_gain_image(
                image,
                self.args.random_gain_db,
                self.encoder.db_scale,
                np.random.default_rng(random.getrandbits(32)),
            )
        if self.transform is not None:
            image = self.transform(image)
        values = image, self.labels[index], self.devices[index], int(self.patients[index]), self.quality[index]
        return (*values, self.soft_targets[index]) if self.train_flag and self.soft_targets is not None else values

    def __len__(self):
        return len(self.labels)


class PatientCappedSampler(Sampler):
    """Draw N source cycles with replacement, balancing patients under ceil(N/P)."""

    def __init__(self, labels, patients, devices):
        self.labels, self.patients, self.devices = map(np.asarray, (labels, patients, devices))
        if not len(self.labels) or not (len(self.labels) == len(self.patients) == len(self.devices)):
            raise ValueError("patient cap requires nonempty aligned training metadata")
        self.patient_ids, self.patient_index = np.unique(self.patients, return_inverse=True)
        self.rows = [np.flatnonzero(self.patient_index == i) for i in range(len(self.patient_ids))]
        self.cap = (len(self.labels) + len(self.rows) - 1) // len(self.rows)
        self.classes = sorted(np.unique(self.labels), key=lambda c: len(np.unique(self.patients[self.labels == c])))
        self.class_patients = [[i for i, rows in enumerate(self.rows) if np.any(self.labels[rows] == c)] for c in self.classes]
        # Reserve every source class before filling quotas; a strict ceiling can be infeasible.
        self.reservations = []
        counts = np.zeros(len(self.rows), dtype=int)

        def reserve(k):
            if k == len(self.classes):
                return True
            for i in sorted(self.class_patients[k], key=lambda i: counts[i]):
                if counts[i] < self.cap:
                    counts[i] += 1
                    self.reservations.append(i)
                    if reserve(k + 1):
                        return True
                    self.reservations.pop()
                    counts[i] -= 1
            return False

        if not reserve(0):
            raise ValueError("patient ceiling cannot preserve all source training classes")

    def __len__(self):
        return len(self.labels)

    def __iter__(self):
        # Use the checkpointed torch RNG so epoch-boundary resume repeats the next draw.
        counts = np.bincount(self.reservations, minlength=len(self.rows))
        selected = []
        for c, i in zip(self.classes, self.reservations):
            rows = self.rows[i][self.labels[self.rows[i]] == c]
            selected.append(int(rows[torch.randint(len(rows), ()).item()]))
        order = torch.randperm(len(self.rows)).tolist()
        for _ in range(len(self) - len(selected)):
            i = min(order, key=lambda i: counts[i])
            counts[i] += 1
        for i, rows in enumerate(self.rows):
            remaining = counts[i] - self.reservations.count(i)
            selected.extend(rows[torch.randint(len(rows), (remaining,)).numpy()].tolist())
        selected = np.asarray(selected)[torch.randperm(len(selected)).numpy()]
        self.last_counts = {
            "patient": dict(zip(self.patient_ids.tolist(), counts.tolist())),
            "class": dict(zip(*[v.tolist() for v in np.unique(self.labels[selected], return_counts=True)])),
            "device": dict(zip(*[v.tolist() for v in np.unique(self.devices[selected], return_counts=True)])),
        }
        print(f"Patient sampling: draws={len(selected)}, ceiling={self.cap}, counts={self.last_counts}")
        return iter(selected.tolist())


def _training_sampling(ds, indices, args):
    if not getattr(args, "patient_cap", False):
        return {"shuffle": True, "drop_last": True}
    if get_method(args.method).multilabel:
        raise ValueError("patient_cap does not support synthetic multi-cycle training")
    return {"sampler": PatientCappedSampler(np.asarray(ds.labels)[indices], ds.patients[indices],
                                           np.asarray(ds.devices)[indices]), "drop_last": False}


class _Cycles:
    """Labels, devices, patients and SC state of a subset of cycles (what the report and export code reads)."""

    def __init__(self, ds, idx):
        self.labels = [ds.labels[i] for i in idx]
        self.devices = [ds.devices[i] for i in idx]
        self.patients = ds.patients[idx]
        self.sc = ds.sc


class Loaders(NamedTuple):
    """Training loaders; `val` is None for a fixed refit, `test_dataset` is None under cross-validation, and the
    calibration fields are set for leave-one-device-out runs only."""

    train: DataLoader
    val: DataLoader | None
    test: DataLoader | None
    train_dataset: Dataset
    test_dataset: object
    calibration: DataLoader | None = None
    calibration_cycles: object = None


def _lodo_loaders(args):
    """Leave-one-device-out loaders: training patients, calibration patients (evaluated at the report epochs) and the
    held-out device as test set.

    Returns:
        `Loaders` with the held-out device as test set and its cycles as `test_dataset`.
    """
    train_tf = transforms.Compose([transforms.ToTensor()] + ([SpecAugment()] if args.spec_augment and args.method != "sg_scl" else []))
    eval_tf = transforms.ToTensor()
    ds = ICBHIDataset(True, train_tf, args)
    tr_idx, ca_idx, te_idx = ds.lodo
    eval_ds = copy.copy(ds)
    eval_ds.transform, eval_ds.train_flag = eval_tf, False
    print(
        "LODO {}: patients train/cal/test {}/{}/{}, cycles {}/{}/{}, dropped (both official sides) {}".format(
            args.holdout_device,
            *(len(set(ds.patients[i])) for i in (tr_idx, ca_idx, te_idx)),
            len(tr_idx),
            len(ca_idx),
            len(te_idx),
            sorted(ds.dropped),
        )
    )

    def make(d, **kw):
        return DataLoader(
            d, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, **kw
        )

    train = make(Subset(ds, tr_idx), **_training_sampling(ds, tr_idx, args))
    if get_method(args.method).multilabel:  # on-the-fly multi-cycle images use the main process
        train = DataLoader(PCMCLTrain(ds, tr_idx, args), batch_size=args.batch_size, shuffle=True, drop_last=True)
    return Loaders(
        train,
        None,
        make(Subset(eval_ds, te_idx), shuffle=False),
        ds,
        _Cycles(ds, te_idx),
        make(Subset(eval_ds, ca_idx), shuffle=False),
        _Cycles(ds, ca_idx),
    )


def build_loaders(args):
    """Train, validation (None for a fixed refit), test and, for leave-one-device-out, calibration loaders as `Loaders`."""
    if args.holdout_device:
        return _lodo_loaders(args)
    train_tf = transforms.Compose([transforms.ToTensor()] + ([SpecAugment()] if args.spec_augment and not get_method(args.method).two_views else []))
    eval_tf = transforms.ToTensor()
    train_dataset = ICBHIDataset(True, train_tf, args)
    test_dataset = None if args.cv_folds else ICBHIDataset(
        False, eval_tf, args, sc_reference=train_dataset.sc and train_dataset.sc["device_spectra"]
    )

    # Patient-level validation split of the training set. Shallow copy with the eval transform: shares the cached
    # spectrograms, no SpecAugment.
    tr_idx, va_idx = _validation_indices(train_dataset)
    val_dataset = copy.copy(train_dataset)
    val_dataset.transform, val_dataset.train_flag = eval_tf, False
    val_set = Subset(val_dataset, va_idx)
    train_set = Subset(train_dataset, tr_idx) if args.selection == "cv" else train_dataset
    print(
        "train/val patients: {}/{}  cycles {}/{}".format(
            len(set(train_dataset.patients[tr_idx])),
            len(set(train_dataset.patients[va_idx])),
            len(tr_idx),
            len(va_idx),
        )
    )

    def make(ds, **kw):
        return DataLoader(
            ds, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, **kw
        )

    rows = tr_idx if args.selection == "cv" else np.arange(len(train_dataset.labels))
    train_loader = make(train_set, **_training_sampling(train_dataset, rows, args))
    if get_method(args.method).multilabel:  # on-the-fly multi-cycle images use the main process
        assert args.selection != "cv", "pcmcl: CV screening not implemented"
        train_loader = DataLoader(PCMCLTrain(train_dataset, np.arange(len(train_dataset.labels)), args),
                                  batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = (
        make(val_set, shuffle=False) if args.selection == "cv" else None
    )  # fixed refit: val patients are in the training set
    return Loaders(
        train_loader,
        val_loader,
        None if test_dataset is None else make(test_dataset, shuffle=False),
        train_dataset,
        test_dataset,
    )
