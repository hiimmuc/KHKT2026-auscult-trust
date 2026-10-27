"""ICBHI cycle dataset (official 60/40 split, or one 80/20 patient fold) and the training data loaders."""

import copy
import os
import random
from collections import defaultdict

import numpy as np
import torch
from src.paths import DATA
from src.processing.correction import (
    apply_spectrum_correction,
    mean_spectrum,
    reference_spectrum,
    spectrum_coefficients,
)
from src.processing.icbhi import DEVICES
from src.processing.noise import TRAIN_FOLDS, noise_bank
from src.processing.splits import cv_split, lodo_split, patient_val_split
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from torchvision import transforms

from .augment import SpecAugment, noise_mix_image, random_bin_gain_image
from src.processing.cycles import individual_cycles, read_annotations
from .methods.pcmcl_data import PCMCLTrain
from .methods import get_method
from .encoders import get_encoder, image_shape

CACHE = (
    DATA / "cache" / "train_images"
)  # one array per encoder and split, shared by all jobs and seeds


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
        lodo = bool(args.holdout_device)  # leave-one-device-out: one dataset over all cycles, split by index later
        self.split = "all" if lodo else "train" if train_flag else "test"
        if args.test_fold != "official":
            self.split = "fold{}_{}".format(args.test_fold, self.split)
        wav_dir = os.path.join(args.data_folder, "audio_test_data")
        official = dict(
            line.strip().split("\t")
            for line in open(os.path.join(args.data_folder, "official_split.txt"))
            .read()
            .splitlines()
        )
        stems = sorted(
            {f.split(".")[0] for f in os.listdir(wav_dir) if f.endswith((".wav", ".txt"))}
        )
        if lodo:  # drop the patients that the official split puts on both sides
            sides = defaultdict(set)
            for f in stems:
                sides[f.split("_")[0]].add(official.get(f))
            self.dropped = {p for p, s in sides.items() if len(s) > 1 or None in s}
            self.filenames = [f for f in stems if f.split("_")[0] not in self.dropped]
        elif args.test_fold == "official":
            self.filenames = [f for f in stems if official.get(f) == self.split]
        else:
            fold = dict(line.split() for line in open(os.path.join(args.data_folder, "patient_list_foldwise.txt")).read().splitlines())
            self.filenames = [f for f in stems if (fold[f.split("_")[0]] == args.test_fold) != train_flag]
        annotations = read_annotations(wav_dir)

        self.encoder = get_encoder(args.encoder)
        cached = None if args.spectrum_correction or audio_only else self._load_cache()
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
            self.sc = (
                self._spectrum_correct(audio, sc_reference) if args.spectrum_correction else None
            )
            if self.sc:
                audio = self.sc.pop("audio")
            if args.spectrum_correction:
                self.images = [self.encoder.preprocess(a) for a in audio]
            else:
                self.images = self._build_cache(audio)
        assert self.images[0].shape == (*image_shape(self.encoder, self.args), 1), self.images[0].shape
        self.quality = get_method(args.method).quality_scores(self, annotations, wav_dir, CACHE)
        self.soft_targets = None
        if train_flag and getattr(args, 'soft_targets', None):
            from .distillation import aligned_targets, training_rows
            ids = np.array([f'{stem}:{i}' for stem in self.filenames for i in range(len(annotations[stem]))])
            rows = training_rows(self)
            self.soft_targets = aligned_targets(args.soft_targets, ids, np.asarray(self.labels), self.patients,
                                                np.asarray(self.devices), rows)
        if train_flag and args.noise_train_prob > 0:  # training noise: ESC-50 folds disjoint from the noise-stress test
            self.noise_images = np.stack([self.encoder.preprocess(torch.from_numpy(w)[None]) for w in noise_bank(TRAIN_FOLDS)])
            assert self.noise_images.shape[1:] == self.images[0].shape, "training noise needs 8 s cycles"
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

    def _cache_paths(self):
        a = self.args
        stem = "{}_{}_{}cls_{}s_{}_{}hz_{}mel".format(
            a.encoder, self.split, a.n_cls, a.desired_length, a.pad_types, a.sample_rate, a.n_mels
        )
        return CACHE / (stem + ".npy"), CACHE / (stem + ".meta.npz")

    def _load_cache(self):
        """Spectrograms of this encoder and split memory-mapped from disk (shared by every job), or None."""
        images, meta = self._cache_paths()
        if not (images.exists() and meta.exists()):
            return None
        m = np.load(meta)
        return (
            np.load(images, mmap_mode="r"),
            m["labels"].tolist(),
            m["devices"].tolist(),
            m["patients"],
        )

    def _build_cache(self, audio):
        """Write the spectrogram of every cycle straight into a .npy file (no second copy in RAM), return it memory-mapped."""
        images, meta = self._cache_paths()
        images.parent.mkdir(parents=True, exist_ok=True)
        tmp, tmp_meta = (
            p.with_name("{}.{}.tmp{}".format(p.stem, os.getpid(), p.suffix))
            for p in (images, meta)
        )
        out = np.lib.format.open_memmap(
            tmp, mode="w+", dtype=np.float32, shape=(len(audio), *image_shape(self.encoder, self.args), 1)
        )
        for i, a in enumerate(audio):
            out[i] = self.encoder.preprocess(a)
        out.flush()
        del out
        np.savez(tmp_meta, labels=self.labels, devices=self.devices, patients=self.patients)
        tmp_meta.replace(meta)
        tmp.replace(images)  # the image file last: its presence means both files are complete
        return np.load(images, mmap_mode="r")


    def _spectrum_correct(self, audio, sc_reference):
        """SC of every cycle. Returns dict `device_spectra`, `reference` (s_ref), `coefficients`, corrected `audio`."""
        used = None
        validation = set()
        reference_rows = None
        partition = "validation"
        if self.lodo is not None:
            train_rows, cal_rows, test_rows = self.lodo
            used = set(np.concatenate((train_rows, test_rows)).tolist())
            validation = set(cal_rows.tolist())
            reference_rows = train_rows
            partition = "calibration"
        elif self.train_flag and self.args.selection == "cv":
            train_rows, val_rows = _validation_indices(self)
            used, validation = set(train_rows.tolist()), set(val_rows.tolist())
            reference_rows = train_rows
        by_dev = defaultdict(list)
        val_by_dev = defaultdict(list)
        for i, (a, d) in enumerate(zip(audio, self.devices)):
            if used is None or i in used:
                by_dev[d].append(a.numpy()[0])
            if i in validation:
                val_by_dev[d].append(a.numpy()[0])
        spectra = {d: mean_spectrum(w) for d, w in by_dev.items()}
        if self.lodo is not None:  # reference = mean of the source devices; the held-out device is corrected towards it
            held = DEVICES.index(self.args.holdout_device)
            ref_spectra = {d: s for d, s in spectra.items() if d != held}
        else:
            ref_spectra = spectra if self.train_flag else sc_reference
        a = self.args
        coef = spectrum_coefficients(
            spectra,
            reference=a.sc_reference,
            sc_mode=a.sc_mode,
            limit_freq_low=a.sc_limit_freq_low,
            limit_freq_high=a.sc_limit_freq_high,
            limit_freq_diff=a.sc_limit_freq_diff,
            reference_spectra=ref_spectra,
        )
        if not self.train_flag and self.lodo is None and getattr(a, "sc_test_profile", "target") == "source":
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
            for i, (w, d) in enumerate(zip(audio, self.devices))
        ]
        return {
            "device_spectra": spectra,
            "reference": reference_spectrum(ref_spectra, a.sc_reference),
            "coefficients": coef,
            "audio": corrected,
            **({partition + "_spectra": val_spectra, partition + "_coefficients": val_coef,
                "reference_patients": np.unique(self.patients[reference_rows]),
                partition + "_patients": np.unique(self.patients[list(validation)])} if validation else {}),
        }

    def __getitem__(self, index):
        image = np.array(self.images[index])  # a copy: the cached array is read-only
        if self.train_flag and self.args.random_gain_db > 0:  # P3
            image = random_bin_gain_image(
                image,
                self.args.random_gain_db,
                self.encoder.db_scale,
                np.random.default_rng(random.getrandbits(32)),
            )
        if self.train_flag and self.args.noise_train_prob > 0 and random.random() < self.args.noise_train_prob:
            rng = np.random.default_rng(random.getrandbits(32))
            image = noise_mix_image(image, self.noise_images[rng.integers(len(self.noise_images))],
                                    rng.uniform(self.args.noise_snr_low, self.args.noise_snr_high), self.encoder.db_scale, rng)
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


def _lodo_loaders(args):
    """Leave-one-device-out loaders: training patients, calibration patients (evaluated at the report epochs) and the
    held-out device as test set.

    Returns:
        Tuple (train_loader, None, test_loader, dataset, test cycles, calibration loader, calibration cycles).
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
    return (
        train,
        None,
        make(Subset(eval_ds, te_idx), shuffle=False),
        ds,
        _Cycles(ds, te_idx),
        make(Subset(eval_ds, ca_idx), shuffle=False),
        _Cycles(ds, ca_idx),
    )


def build_loaders(args):
    """Train, validation (None for a fixed refit) and test loaders, plus calibration loader and cycles (leave-one-device-out only).

    Returns:
        Tuple (train_loader, val_loader, test_loader, train_dataset, test_dataset, calibration_loader, calibration_cycles);
        the last two are None unless `args.holdout_device` is set.
    """
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
    return (
        train_loader,
        val_loader,
        None if test_dataset is None else make(test_dataset, shuffle=False),
        train_dataset,
        test_dataset,
        None,
        None,
    )
