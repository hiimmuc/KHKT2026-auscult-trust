"""Train an ICBHI lung-sound classifier with a selected encoder and method.

    python -m auscult_trust.training.main --config experiments/train/base.yaml,experiments/train/sc.yaml \
        --cv_folds 3 --cv_fold 0 --seed 0

Writes `outputs/<exp>/<run_id>/<cell>/<unit>/` (train_args.json, report.json, test_preds.npz, sc_state.npz) and
`checkpoints/<exp>/<run_id>/<cell>/<unit>/` (last.pth while running; report_epoch_<E>.pth for a fixed refit, best.pth for a
cv/test-selected full run, nothing for a screening fold).
A finished unit is skipped; an interrupted one resumes from `last.pth` when the same RUN_ID is set.
"""

import hashlib
import json
import random
import time
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from auscult_trust import runs
from auscult_trust.artifacts import atomic_json, atomic_path, atomic_npz
from auscult_trust.provenance import prediction_inputs
from auscult_trust.training.state import capture_rng, restore_rng, completed, publish_completion

from .averaging import EpochAverager
from .config import build_parser, parse_args
from .dataset import build_loaders
from .engine import evaluate, train_epoch
from .modeling import build_model
from .optim import adjust_learning_rate
from .reporting import pick_epochs
from .methods import get_method

warnings.filterwarnings("ignore")
_VOLATILE = {"num_workers", "print_freq"}  # legacy recipe audit ignores scheduling settings


def atomic_save(value, path):
    with atomic_path(path) as temporary:
        torch.save(value, temporary)


def save_model(model, args, epoch, path, classifier, projector=None, swa=None):
    """Weights for inference: classifier always, encoder only when it was trained (a frozen one is the pretrained file)."""
    atomic_save(
        {
            "args": vars(args),
            # ADD-RSC's denoising layer is trained even in front of a frozen encoder
            "model": None if args.freeze_encoder and not get_method(args.method).trainable_frontend else model.state_dict(),
            "epoch": epoch,
            "classifier": classifier.state_dict(),
            **({"projector": projector.state_dict(), "swa": swa} if swa is not None else {}),
        },
        path,
    )


def check_same_config(path, args):
    """Refuse to continue a unit whose saved config differs from this run's (no silent mixing of runs)."""
    new = json.loads(json.dumps(vars(args), default=str))
    if path.exists():
        old = json.loads(path.read_text())
        defaults = json.loads(json.dumps(vars(build_parser().parse_args([])), default=str))
        # a key added to the parser after the unit started is fine while it keeps its default
        # Worker count affects stochastic augmentation streams and must match on resume.
        diff = {k for k in new if k not in (_VOLATILE - {"num_workers"}) and old.get(k, defaults.get(k)) != new[k]}
        if diff:
            raise ValueError("{} exists with a different config ({}); use a new RUN_ID".format(path.parent, sorted(diff)))
        return
    atomic_json(path, new)


def save_sc_state(out_dir, train_dataset, test_dataset, args):
    """Save s_ref, the per-device spectra and SC coefficients (train and test) and the SC hash; returns the hash."""
    state = {"sc_mode": args.sc_mode, "reference": train_dataset.sc["reference"]}
    for name, ds in (("train", train_dataset), ("test", test_dataset)):
        if ds is None:
            continue
        for d, v in ds.sc["device_spectra"].items():
            state["{}_spectrum_{}".format(name, d)] = v
        for d, v in ds.sc["coefficients"].items():
            state["{}_coefficients_{}".format(name, d)] = v
    for key in ("validation_spectra", "validation_coefficients", "calibration_spectra", "calibration_coefficients"):
        for d, v in train_dataset.sc.get(key, {}).items():
            state["{}_{}".format(key, d)] = v
    for key in ("reference_patients", "validation_patients", "calibration_patients"):
        if key in train_dataset.sc:
            state[key] = train_dataset.sc[key]
    atomic_npz(out_dir / "sc_state.npz", **state)
    return hashlib.sha256(np.ascontiguousarray(state["reference"]).tobytes()).hexdigest()


def history_json(history):
    score = lambda r: None if r is None else r["all"]["score"]
    sse = lambda r: None if r is None else {k: r["all"][k] for k in ("sp", "se", "score", "hs")}
    return [
        {
            "epoch": h["epoch"],
            "train_loss": h.get("train_loss"),  # absent in units resumed from an older last.pth
            "train_acc": h.get("train_acc"),
            "val_score": score(h["val"]),
            "test_score": score(h["test"]),
            "val": sse(h["val"]),
            "test": sse(h["test"]),
            "val_devices": (
                None
                if h["val"] is None
                else {k: {m: v[m] for m in ("sp", "se", "score", "hs")} for k, v in h["val"].items() if k != "all"}
            ),
            "recording_val": h.get("recording_val"),
            **({"swa": h["swa"]} if "swa" in h else {}),
        }
        for h in history
    ]


def main():
    args = parse_args()
    out_dir, ckpt_dir = (
        runs.unit_dir(args.exp, args.cell, args.unit, kind) for kind in ("outputs", "checkpoints")
    )
    check_same_config(out_dir / "train_args.json", args)
    if completed(out_dir, ckpt_dir):
        print("done: {}".format(out_dir))
        return
    inputs = prediction_inputs(out_dir, None, "cycles")
    input_path = out_dir / "input_manifest.json"
    if input_path.exists() and json.loads(input_path.read_text()) != inputs:
        raise ValueError("Training inputs or environment changed; use a new RUN_ID")
    atomic_json(input_path, inputs)
    runs.write_meta(args.exp, {"config_files": args.config_files})

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    loaders = build_loaders(args)
    train_loader, val_loader, test_loader = loaders.train, loaders.val, loaders.test
    train_dataset, test_dataset = loaders.train_dataset, loaders.test_dataset
    cal_loader, cal_cycles = loaders.calibration, loaders.calibration_cycles
    if args.method in ('ce_recording_pafa', 'ce_device_pafa'):
        from torch.utils.data import DataLoader
        source_sampler = train_loader.sampler
        if args.method == 'ce_recording_pafa':
            from .recording_bags import wrap_training_dataset
        else:
            from .device_augment import wrap_training_dataset
        train_loader = DataLoader(wrap_training_dataset(train_loader.dataset, args),
                                  batch_size=args.batch_size,
                                  **({'sampler': source_sampler} if args.patient_cap else {'shuffle': True}),
                                  num_workers=args.num_workers, pin_memory=True, drop_last=False)
    labels, idx = np.asarray(train_dataset.labels), getattr(train_loader.dataset, "indices", None)  # Subset or full set
    args.class_nums = np.bincount(labels if idx is None else labels[idx], minlength=args.n_cls)  # QLung's class-prior margin
    model, classifier, projector, criterion, optimizer = build_model(args)
    scaler = torch.cuda.amp.GradScaler()
    average = EpochAverager(getattr(args, "swa_start", 0), getattr(args, "swa_interval", 1))
    modules = (model, classifier, projector)
    best_swa = None
    s_ref_sha = (
        save_sc_state(out_dir, train_dataset, test_dataset, args) if train_dataset.sc else None
    )

    start_epoch, best_key, best_model, history, test_preds = 1, -1.0, None, [], {}
    val_predictions = {}
    recording_predictions = {}
    cal_preds = {}  # leave-one-device-out: calibration-patient predictions at the report epochs
    last = ckpt_dir / "last.pth"
    if last.is_file():  # restore RNG after model and dataset construction, before the next iterator
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]), classifier.load_state_dict(ck["classifier"])
        projector.load_state_dict(ck["projector"]), optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        for c, state in zip(criterion, ck.get("criterion", [])):
            c.load_state_dict(state)
        start_epoch, best_model, best_key = ck["epoch"] + 1, ck["best_model"], ck["best_key"]
        history, test_preds = ck["history"], ck["test_preds"]
        val_predictions = ck.get("val_predictions", {})
        recording_predictions = ck.get("recording_predictions", {})
        cal_preds = ck.get("cal_preds", {})
        average.load_state_dict(ck.get("swa"))
        best_swa = ck.get("best_swa")
        restore_rng(ck.get("rng"))
        print("=> resumed '{}' after epoch {}".format(last, ck["epoch"]))

    screening = args.cv_folds > 0  # selection never sees test data while screening

    def select_key(h):
        if (
            args.selection == "fixed"
        ):  # registered epoch from screening; later report epochs do not replace it
            return 1.0 if h["epoch"] == args.report_epoch_list[0] else -1.0
        return (h["val"] if args.selection == "cv" else h["test"])["all"]["score"]

    print(
        "Training {} epochs (cell={}, unit={}, selection={})".format(
            args.epochs, args.cell, args.unit, args.selection
        )
    )
    for epoch in range(start_epoch, args.epochs + 1):
        adjust_learning_rate(args, optimizer, epoch)
        t0 = time.time()
        loss, acc = train_epoch(
            train_loader, model, classifier, projector, criterion, optimizer, epoch, args, scaler
        )
        print(
            "Train epoch {}, total time {:.2f}, accuracy:{:.2f}".format(
                epoch, time.time() - t0, acc
            )
        )

        average.update(epoch, modules)
        ready = not average.start or len(average.epochs) >= 2
        with average.applied(modules):
            val_rep = None
            recording_val = None
            if ready and val_loader is not None:
                val_rep, _, val_probs, val_labels, val_devices = evaluate(val_loader, model, classifier, args)
                val_predictions[epoch] = val_probs.astype(np.float32)
                if args.method == 'ce_recording_pafa':
                    from .recording_bags import evaluate_recordings, source_indices
                    ds, rows = source_indices(train_loader.dataset)
                    rec = evaluate_recordings(val_loader.dataset, model, classifier, args,
                                              excluded_patients=np.asarray(ds.patients)[rows])
                else:
                    from .validation import recording_from_cycles
                    rec = recording_from_cycles(val_loader.dataset, val_probs, args.recording_eval_cycles)
                recording_val = rec[0]
                recording_predictions[epoch] = (rec[2].astype(np.float32), *rec[3:])
            test_rep = None
            if ready and not screening:
                test_rep, preds, probs, _, _ = evaluate(test_loader, model, classifier, args)
                test_preds[epoch] = (preds.astype(np.int8), probs.astype(np.float16))
            history.append(
                {"epoch": epoch, "train_loss": float(loss), "train_acc": float(acc), "val": val_rep,
                 "test": test_rep, "recording_val": recording_val,
                 **({"swa": average.metadata()} if average.start else {})}
            )
            t = (
                test_rep["all"]
                if test_rep
                else {"sp": float("nan"), "se": float("nan"), "score": float("nan")}
            )
            print(
                " * test Sp {:.2f}, Se {:.2f}, Score {:.2f} | val Score {}".format(
                    t["sp"],
                    t["se"],
                    t["score"],
                    "{:.2f}".format(val_rep["all"]["score"]) if val_rep else "-",
                )
            )

            key = select_key(history[-1]) if ready else -1.0
            if key > best_key and (args.selection in ("cv", "fixed") or t["se"] > 5):
                best_key = key
                best_model = [
                    {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
                    for m in (modules if average.start else (model, classifier))
                ]
                best_swa = average.metadata() if average.start else None
                print("Best ckpt is modified with key = {:.2f} when Epoch = {}".format(key, epoch))
                if screening:
                    save_model(model, args, epoch, ckpt_dir / "best.pth", classifier, projector, best_swa)
            if ready and cal_loader is not None and epoch in args.report_epoch_list:
                _, c_preds, c_probs, _, _ = evaluate(cal_loader, model, classifier, args)
                cal_preds[epoch] = (c_preds.astype(np.int8), c_probs.astype(np.float16))
            if ready and epoch in getattr(args, 'snapshot_epoch_list', []):
                save_model(model, args, epoch, ckpt_dir / f'snapshot_epoch_{epoch}.pth', classifier,
                           projector, average.metadata() if average.start else None)
            if ready and args.selection == "fixed" and epoch in args.report_epoch_list:
                save_model(
                    model, args, epoch, ckpt_dir / "report_epoch_{}.pth".format(epoch), classifier,
                    projector, average.metadata() if average.start else None
                )

        torch.save(
            {
                "model": model.state_dict(),
                "classifier": classifier.state_dict(),
                "projector": projector.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "criterion": [c.state_dict() for c in criterion],
                "best_model": best_model,
                **({"swa": average.state_dict(), "best_swa": best_swa} if average.start else {}),
                "best_key": best_key,
                "history": history,
                "test_preds": test_preds,
                "cal_preds": cal_preds,
                "val_predictions": val_predictions,
                "recording_predictions": recording_predictions,
                "epoch": epoch,
                "rng": capture_rng(),
            },
            str(last) + ".tmp",
        )
        (ckpt_dir / "last.pth.tmp").replace(
            last
        )  # atomic: a crash mid-save keeps the previous epoch
        if cal_loader is not None and epoch >= max(args.report_epoch_list):
            break  # same weights as the full schedule at this epoch; later epochs are not needed
        if args.stop_epoch and epoch >= args.stop_epoch:
            break

    picked = pick_epochs(history)
    if args.selection == "fixed":
        by_epoch = {h["epoch"]: h for h in history}
        picked["fixed"] = {str(e): by_epoch[e] for e in args.report_epoch_list}
    report = {
        "cell": args.cell,
        "unit": args.unit,
        "selection": args.selection,
        "sc_mode": args.sc_mode if args.spectrum_correction else None,
        "s_ref_sha256": s_ref_sha,
        **({"swa": average.metadata()} if average.start else {}),
        **picked,
        "history": history_json(history),
    }
    if val_predictions:
        from .dataset import validation_metadata
        from .recording_bags import source_indices
        patients, stems = validation_metadata(val_loader.dataset)
        source, indices = source_indices(val_loader.dataset)
        epochs = sorted(val_predictions)
        atomic_npz(out_dir / "val_preds.npz", epochs=np.array(epochs),
                   probs=np.stack([val_predictions[e] for e in epochs]),
                   labels=np.asarray(source.labels)[indices],
                   device=np.asarray(source.devices)[indices], patient=patients, stem=stems)
    if recording_predictions:
        epochs = sorted(recording_predictions)
        _, labels, devices, stems, patients = recording_predictions[epochs[0]]
        atomic_npz(out_dir / "recording_val_preds.npz", epochs=np.array(epochs),
                   probs=np.stack([recording_predictions[e][0] for e in epochs]),
                   labels=labels, device=devices, stem=stems, patient=patients)
    if test_preds:  # per-epoch test predictions for paired, patient-level bootstrap CIs offline
        ep = sorted(test_preds)
        atomic_npz(
            out_dir / "test_preds.npz",
            epochs=np.array(ep),
            preds=np.stack([test_preds[e][0] for e in ep]),
            probs=np.stack([test_preds[e][1] for e in ep]),
            labels=np.asarray(test_dataset.labels),
            device=np.asarray(test_dataset.devices),
            patient=np.asarray(test_dataset.patients),
        )
    if cal_preds:  # leave-one-device-out: calibration set of the conformal analysis
        ep = sorted(cal_preds)
        atomic_npz(
            out_dir / "cal_preds.npz",
            epochs=np.array(ep),
            probs=np.stack([cal_preds[e][1] for e in ep]),
            labels=np.asarray(cal_cycles.labels),
            device=np.asarray(cal_cycles.devices),
            patient=np.asarray(cal_cycles.patients),
        )
    for name in ("last", "cv_selected", "test_best_optimistic"):
        if name in picked and picked[name]["test"] is not None:
            r = picked[name]["test"]["all"]
            print(
                "{:>22} (epoch {:>3}): Sp {:.2f} Se {:.2f} Score {:.2f} HS {:.2f} macroF1 {:.3f}{}".format(
                    name,
                    picked[name]["epoch"],
                    r["sp"],
                    r["se"],
                    r["score"],
                    r["hs"],
                    r["macro_f1"],
                    (
                        "  [OPTIMISTIC: epoch chosen on test]"
                        if name == "test_best_optimistic"
                        else ""
                    ),
                )
            )
    if (
        best_model is not None and args.selection != "fixed" and not screening
    ):  # frozen weights of the selected epoch
        model.load_state_dict(best_model[0]), classifier.load_state_dict(best_model[1])
        if best_swa is not None:
            projector.load_state_dict(best_model[2])
        save_model(model, args, args.epochs, ckpt_dir / "best.pth", classifier, projector, best_swa)
    atomic_json(out_dir / "report.json", report)
    publish_completion(out_dir, ckpt_dir)
    last.unlink()  # resume state, after the report: a crash in between keeps the unit done, not half-written


if __name__ == "__main__":
    main()
