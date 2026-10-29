"""Frozen evaluation of the distilled student against its no-SC twin (KAUH A/B2 paired by patient and seed, screening).

Run after `final_evaluation_student.py` and after the twin units (`student-final-20261029-final/kd_g4_k1_nosc`, five seeds) are on this machine:

    OMP_NUM_THREADS=4 PYTHONPATH=. .venv-train/bin/python -u scripts/final_evaluation_twin.py

Conditions: twin = same recipe and soft targets with SC off (epoch 66); student = `kd_g4_k1` (exports reused from `final-kd-g4-k1` when their
provenance verifies); twin_ttsc = twin weights with the student's static SC reference applied at test time only (isolation contrast).
"""
import os

os.environ["AT_EVAL_RUN"] = "final-kd-g4-k1-twin"
import hashlib
import shutil
from pathlib import Path

import numpy as np

from scripts.final_evaluation_student import (RUN, certified, event, export_ok, run, sha)
from src.artifacts import immutable_json
from src.provenance import environment, prediction_inputs, verify_prediction

PREV_RUN = "final-kd-g4-k1"
TWIN = Path("outputs/train/student-final-20261029-final/kd_g4_k1_nosc")
STUDENT = Path("outputs/train/distill-20261028-final/kd_g4_k1")
EPOCH = 66


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ["PYTHONPATH"] = "."
    event(status="campaign_started", run=RUN, serving_seed=0, script_sha256=sha(__file__))
    immutable_json(Path("logs") / RUN / "environment.json", environment())
    for seed in range(5):
        twin, student = TWIN / f"seed{seed}", STUDENT / f"seed{seed}"
        t_ckpt = Path(str(twin).replace("outputs/", "checkpoints/", 1)) / f"report_epoch_{EPOCH}.pth"
        s_ckpt = Path(str(student).replace("outputs/", "checkpoints/", 1)) / f"report_epoch_{EPOCH}.pth"
        t_sha, s_sha = sha(t_ckpt), sha(s_ckpt)
        certified(twin, t_ckpt, t_sha)
        certified(student, s_ckpt, s_sha)
        for name, unit, ckpt, ck_sha, sc in (("twin", twin, t_ckpt, t_sha, False), ("student", student, s_ckpt, s_sha, True)):
            frozen = Path("outputs/train") / RUN / name / f"seed{seed}"
            frozen.mkdir(parents=True, exist_ok=True)
            manifest = {"condition": name, "seed": seed, "epoch": EPOCH, "unit": str(unit), "checkpoint_sha256": ck_sha, "sc": sc,
                        "inputs": {str(p): sha(p) for p in (ckpt, unit / "train_args.json", unit / "test_preds.npz", unit / "report.json")}}
            if sc:
                with np.load(unit / "sc_state.npz") as z:
                    manifest["s_ref_sha256"] = hashlib.sha256(z["reference"].tobytes()).hexdigest()
            immutable_json(frozen / "freeze.json", manifest)
            for dataset, module in (("screening", "screening"), ("kauh", "kauh_infer")):
                out = Path("outputs") / dataset / RUN / name / f"seed{seed}.npz"
                out.parent.mkdir(parents=True, exist_ok=True)
                source = Path("outputs") / dataset / PREV_RUN / name / f"seed{seed}.npz"
                if name == "student" and not out.exists() and source.exists():
                    shutil.copy2(source, out)
                    for extra in source.parent.glob(f"seed{seed}.npz.*"):
                        shutil.copy2(extra, out.parent / extra.name)
                if not export_ok(out, ck_sha, sc, unit=unit, checkpoint=ckpt, dataset=dataset):
                    run(["-m", "src.evaluation." + module, *(["infer"] if dataset == "screening" else []),
                         unit, "--ckpt", ckpt.name, "--out", out, *(["--sc"] if sc else [])])
                    if not export_ok(out, ck_sha, sc, unit=unit, checkpoint=ckpt, dataset=dataset):
                        raise ValueError(f"Fresh export failed verification: {out}")
        # isolation contrast: twin weights, the student's static SC reference at test time only
        out = Path("outputs/kauh") / RUN / "twin_ttsc" / f"seed{seed}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        options = {"a2": False, "tta_eq": 0, "sc_state": str(student / "sc_state.npz"), "sc_args": str(student / "train_args.json")}
        try:
            verify_prediction(out, prediction_inputs(twin, t_ckpt, "kauh", True, sc_unit=student, options=options))
            ok = True
        except (ValueError, FileNotFoundError, OSError):
            ok = False
        if not ok:
            run(["-m", "src.evaluation.kauh_infer", twin, "--ckpt", t_ckpt.name, "--out", out, "--sc",
                 "--sc_state", student / "sc_state.npz", "--sc_args", student / "train_args.json"])
            verify_prediction(out, prediction_inputs(twin, t_ckpt, "kauh", True, sc_unit=student, options=options))
        event(status="verified", artifact=str(out), sha256=sha(out))
    run(["-m", "src.evaluation.kauh", "--dir", f"outputs/kauh/{RUN}", "--base", "twin",
         "--arms", "student", "twin_ttsc", "--boot", "1000", "--null_splits", "200"])
    run(["-m", "src.evaluation.screening", "analyse", "--dir", f"outputs/screening/{RUN}", "--kauh_dir", f"outputs/kauh/{RUN}",
         "--pairs", "twin:twin", "student:student", "twin:twin_ttsc"])
    event(status="campaign_complete", run=RUN)


if __name__ == "__main__":
    main()
