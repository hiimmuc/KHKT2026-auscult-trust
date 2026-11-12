"""Spectrogram cache of one encoder and split, memory-mapped and shared by every job and seed."""
import os

import numpy as np

from auscult_trust.models.encoders import image_shape
from auscult_trust.paths import DATA

CACHE = DATA / "cache" / "train_images"


class ImageCache:
    def __init__(self, encoder, args, split):
        stem = "{}_{}_{}cls_{}s_{}_{}hz_{}mel".format(
            args.encoder, split, args.n_cls, args.desired_length, args.pad_types, args.sample_rate, args.n_mels
        )
        self.encoder, self.args = encoder, args
        self.images, self.meta = CACHE / (stem + ".npy"), CACHE / (stem + ".meta.npz")

    def load(self):
        """(images memory-mapped, labels, devices, patients), or None when the cache is missing or incomplete."""
        if not (self.images.exists() and self.meta.exists()):
            return None
        m = np.load(self.meta)
        return np.load(self.images, mmap_mode="r"), m["labels"].tolist(), m["devices"].tolist(), m["patients"]

    def build(self, audio, labels, devices, patients):
        """Write the spectrogram of every cycle straight into a .npy file (no second copy in RAM), return it memory-mapped."""
        self.images.parent.mkdir(parents=True, exist_ok=True)
        tmp, tmp_meta = (
            p.with_name("{}.{}.tmp{}".format(p.stem, os.getpid(), p.suffix)) for p in (self.images, self.meta)
        )
        out = np.lib.format.open_memmap(
            tmp, mode="w+", dtype=np.float32, shape=(len(audio), *image_shape(self.encoder, self.args), 1)
        )
        for i, a in enumerate(audio):
            out[i] = self.encoder.preprocess(a)
        out.flush()
        del out
        np.savez(tmp_meta, labels=labels, devices=devices, patients=patients)
        tmp_meta.replace(self.meta)
        tmp.replace(self.images)  # the image file last: its presence means both files are complete
        return np.load(self.images, mmap_mode="r")
