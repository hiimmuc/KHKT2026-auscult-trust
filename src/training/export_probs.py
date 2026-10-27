"""Freeze a trained run and export softmax + embeddings for the conformal evaluation.

Writes `<unit outputs dir>/export/<split>.npz` for the official test set and, when the run had one, the held-out
validation patients (`probs` (n, K), `emb` (n, 768), `labels`, `device` (ids of `src.processing.icbhi.DEVICES`),
`patient`), and `export/model.sha256` with the hash of the weight file, so every downstream number is tied to one frozen
checkpoint. The weights are not modified.

    python -m src.training.export_probs outputs/train/<run_id>/<cell>/<unit> --ckpt report_epoch_<E>.pth
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from src import runs, artifacts
from src.artifacts import atomic_npz
from src.provenance import prediction_inputs, record_prediction

from .config import load_args
from .dataset import build_loaders
from src.inference.model import load_frozen
from .reporting import per_device_report
from .methods import get_method


@torch.no_grad()
def run(loader, model, classifier, method="ce"):
    model.eval(), classifier.eval()
    objective = get_method(method)
    probs, emb, ys, devs = [], [], [], []
    for images, labels, devices, *_ in loader:
        feat = model(images.cuda(non_blocking=True))
        probs.append(objective.probabilities(classifier(feat)).cpu().numpy())
        emb.append((feat.mean(1) if feat.ndim == 3 else feat).float().cpu().numpy())
        ys.append(labels.numpy())
        devs.append(devices.numpy())
    return tuple(np.concatenate(x) for x in (probs, emb, ys, devs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "unit_dir", help="outputs/<exp>/<run_id>/<cell>/<unit> of the run (holds train_args.json)"
    )
    ap.add_argument(
        "--ckpt",
        required=True,
        help="weight file in the unit checkpoints dir (fixed refit: report_epoch_<E>.pth)",
    )
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument(
        "--sc_test_profile",
        choices=["target", "source"],
        default=None,
        help="override for SC runs: source = each test device uses its training coefficient (inductive); output in export_source/",
    )
    a = ap.parse_args()
    out_unit, ckpt_unit = Path(a.unit_dir), runs.twin_dir(a.unit_dir, "checkpoints")
    args = load_args(out_unit)
    args.num_workers = a.num_workers
    if a.sc_test_profile:
        args.sc_test_profile = a.sc_test_profile
    context = dict(unit=out_unit, checkpoint=ckpt_unit / a.ckpt, dataset="cycles",
                   sc=args.spectrum_correction, options={"sc_test_profile": a.sc_test_profile})
    inputs = prediction_inputs(**context)

    _, val_loader, test_loader, train_ds, test_ds, _, _ = build_loaders(args)
    ckpt = ckpt_unit / a.ckpt
    model, classifier = load_frozen(args, ckpt)

    out = out_unit / ("export_source" if a.sc_test_profile == "source" else "export")
    out.mkdir(exist_ok=True)
    loaders = [("test", test_loader, np.asarray(test_loader.dataset.patients))]
    if val_loader is not None:  # a fixed refit trains on every training patient: no held-out val
        loaders.append(
            (
                "val",
                val_loader,
                np.asarray(val_loader.dataset.dataset.patients)[val_loader.dataset.indices],
            )
        )
    for name, loader, patients in loaders:
        P, E, Y, D = run(loader, model, classifier, args.method)
        atomic_npz(
            out / (name + ".npz"), probs=P, emb=E, labels=Y, device=D, patient=patients
        )
        print(
            name,
            P.shape,
            "pooled Score {:.2f}".format(per_device_report(Y, P.argmax(1), D)["all"]["score"]),
        )
        record_prediction(out / (name + ".npz"), inputs, **context)
    if test_ds.sc is not None:  # coefficients actually applied to the test clips
        np.savez_compressed(out / "test_coefficients.npz", **{"coefficients_{}".format(d): c for d, c in test_ds.sc["coefficients"].items()},
                            **{"train_coefficients_{}".format(d): c for d, c in train_ds.sc["coefficients"].items()})
    (out / "model.sha256").write_text("{}  {}\n".format(artifacts.sha256_file(ckpt), ckpt.name))


if __name__ == "__main__":
    main()
