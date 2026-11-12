"""Frozen evaluation of the distilled student `kd_g4_k1` against the 23/10 baseline, the previous AuscultTrust and the gain6 anchor.

Parametrised copy of `final_evaluation.py` (which stays untouched for the 23/10 run). Run from the repository root with the training Python:

    AT_EVAL_RUN=final-kd-g4-k1 OMP_NUM_THREADS=4 PYTHONPATH=. .venv-train/bin/python -u scripts/final_evaluation_student.py

Conditions (five paired seeds 0-4 each): baseline = BEATs + PAFA without SC (epoch 100), previous = BEATs + PAFA + static SC (epoch 100, the 23/10
AuscultTrust), anchor = gain6-cv66 (epoch 66), student = kd_g4_k1 (epoch 66). Verified exports of baseline and previous are copied from the 23/10 run when
their provenance still verifies. No training or model selection; no bundle is built or activated; no isolation contrast (it would pair the baseline
weights with the student's reference, which is not meaningful).
"""
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from scripts.eval_log import logger
from src.artifacts import immutable_json
from src.artifacts import sha256_file as sha
from src.provenance import environment, prediction_inputs, verify_prediction

RUN = os.environ.get("AT_EVAL_RUN", "final-kd-g4-k1")
OLD_RUN = "final-pafa-static"
LOG = Path("logs") / RUN
SERVING_SEED = 0  # fixed before any per-seed result is read, as on 23/10
# name, unit parent, static SC, epoch, name of the same condition in the 23/10 run (exports reusable) or None
CONDITIONS = (
    ("baseline", Path("outputs/train/grid-v1/beats+ft+pafa"), False, 100, "baseline"),
    ("previous", Path("outputs/train/lodo-v1/beats+ft+pafa+sc+sc_static"), True, 100, "auscult"),
    ("anchor", Path("outputs/train/followup-20261026-final/gain6-cv66"), True, 66, None),
    ("student", Path("outputs/train/distill-20261028-final/kd_g4_k1"), True, 66, None),
)


event, run = logger(LOG)


def export_ok(path, checkpoint_sha, sc, *, unit, checkpoint, dataset):
    if not path.exists():
        return False
    options = ({"a2": False, "tta_eq": 0, "sc_state": None, "sc_args": None} if dataset == "kauh" else {})
    try:
        verify_prediction(path, prediction_inputs(unit, checkpoint, dataset, sc, sc_unit=None, options=options))
        with np.load(path, allow_pickle=True) as z:
            if str(z["ckpt_sha256"]) != checkpoint_sha or bool(z["sc"]) != sc:
                raise ValueError(f"Checkpoint or correction mismatch: {path}")
            p = z["probs"]
            if p.ndim != 2 or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-5):
                raise ValueError(f"Invalid probabilities: {path}")
    except Exception as error:  # a reused file that fails verification is recomputed, never trusted
        event(status="rejected", artifact=str(path), reason=str(error)[:200])
        path.unlink()
        return False
    event(status="verified", artifact=str(path), sha256=sha(path))
    return True


def certified(unit, checkpoint, checkpoint_sha):
    """The checkpoint hash must match the campaign receipts, not just whatever file was copied."""
    receipt = unit / "collected-checkpoint.json"
    if receipt.exists() and json.loads(receipt.read_text())["sha256"] != checkpoint_sha:
        raise ValueError(f"Checkpoint differs from its collected receipt: {unit}")
    completion = unit / "completion.json"
    if completion.exists():
        bound = [d for name, d in json.loads(completion.read_text())["files"].items() if name.endswith(checkpoint.name)]
        if bound and bound[0] != checkpoint_sha:
            raise ValueError(f"Checkpoint differs from its completion receipt: {unit}")
        return {"collected": receipt.exists(), "completion": bool(bound)}
    return {"collected": receipt.exists(), "completion": False}


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    os.environ["PYTHONPATH"] = "."
    event(status="campaign_started", run=RUN, serving_seed=SERVING_SEED,
          git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), script_sha256=sha(__file__))
    from src.paths import DATA
    from src.training.config import load_args
    data_files = sorted((DATA / "raw/kauh").rglob("*"))
    args = load_args(CONDITIONS[0][1] / "seed0")
    folder = Path(args.data_folder)
    data_files += sorted((folder / "audio_test_data").glob("*"))
    data_files.append(folder / "official_split.txt")
    input_path = LOG / "input_hashes.json"
    immutable_json(input_path, {str(p): sha(p) for p in data_files if p.is_file()})
    immutable_json(LOG / "environment.json", environment())
    data_sha = sha(input_path)
    for seed in range(5):
        for name, parent, sc, epoch, old in CONDITIONS:
            unit = parent / f"seed{seed}"
            checkpoint = Path(str(unit).replace("outputs/", "checkpoints/", 1)) / f"report_epoch_{epoch}.pth"
            checkpoint_sha = sha(checkpoint)
            receipts = certified(unit, checkpoint, checkpoint_sha)
            inputs = [checkpoint, unit / "train_args.json", unit / "test_preds.npz", unit / "report.json"]
            if sc:
                inputs.append(unit / "sc_state.npz")
            if (unit / "teacher.npz").exists():
                inputs.append(unit / "teacher.npz")
            frozen = Path("outputs/train") / RUN / name / f"seed{seed}"
            frozen.mkdir(parents=True, exist_ok=True)
            manifest = {"condition": name, "seed": seed, "epoch": epoch, "serving_seed": SERVING_SEED, "unit": str(unit),
                        "checkpoint_sha256": checkpoint_sha, "receipts": receipts, "data_manifest_sha256": data_sha,
                        "inputs": {str(p): sha(p) for p in inputs}, "sc": sc}
            if sc:
                with np.load(unit / "sc_state.npz") as z:
                    if str(z["sc_mode"]) != "static":
                        raise ValueError(f"Expected frozen static correction: {unit}")
                    manifest["s_ref_sha256"] = hashlib.sha256(z["reference"].tobytes()).hexdigest()
            immutable_json(frozen / "freeze.json", manifest)
            event(status="frozen", condition=name, seed=seed, epoch=epoch, checkpoint_sha256=checkpoint_sha)
            if old is None:  # new units are private copies here; the 23/10 units keep their own exports untouched
                export = unit / "export"
                inputs_now = lambda: prediction_inputs(unit, checkpoint, "cycles", sc, options={"sc_test_profile": None})
                try:
                    verify_prediction(export / "test.npz", inputs_now())
                except (ValueError, FileNotFoundError, OSError):  # stale or missing under the current code: rerun the exporter
                    shutil.rmtree(export, ignore_errors=True)
                    run(["-m", "src.training.export_probs", unit, "--ckpt", checkpoint.name, "--num_workers", "0"])
                    verify_prediction(export / "test.npz", inputs_now())
                if (export / "model.sha256").read_text().split()[0] != checkpoint_sha:
                    raise ValueError(f"Cycle export checkpoint mismatch: {unit}")
            for dataset, module in (("screening", "screening_infer"), ("kauh", "kauh_infer")):
                out = Path("outputs") / dataset / RUN / name / f"seed{seed}.npz"
                out.parent.mkdir(parents=True, exist_ok=True)
                source = Path("outputs") / dataset / OLD_RUN / (old or "-") / f"seed{seed}.npz"
                if old and not out.exists() and source.exists():
                    shutil.copy2(source, out)
                    for extra in source.parent.glob(f"seed{seed}.npz.*"):
                        shutil.copy2(extra, out.parent / extra.name)
                if not export_ok(out, checkpoint_sha, sc, unit=unit, checkpoint=checkpoint, dataset=dataset):
                    run(["-m", "src.pipelines." + module,
                         unit, "--ckpt", checkpoint.name, "--out", out, *(["--sc"] if sc else [])])
                    if not export_ok(out, checkpoint_sha, sc, unit=unit, checkpoint=checkpoint, dataset=dataset):
                        raise ValueError(f"Fresh export failed verification: {out}")
    run(["-m", "src.evaluation.kauh", "--dir", f"outputs/kauh/{RUN}", "--base", "baseline",
         "--arms", "previous", "anchor", "student", "--boot", "1000", "--null_splits", "200"])
    run(["-m", "src.evaluation.screening", "--dir", f"outputs/screening/{RUN}", "--kauh_dir", f"outputs/kauh/{RUN}",
         "--pairs", "baseline:baseline", "previous:previous", "anchor:anchor", "student:student"])
    event(status="campaign_complete", run=RUN)


if __name__ == "__main__":
    main()
