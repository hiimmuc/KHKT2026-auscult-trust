"""Run the frozen PAFA baseline/static-SC evaluation with an append-only command log.

Run from the repository root with the training Python environment. Existing prediction exports are
reused only when their full provenance and output hash match. No training or model selection is done.
"""

import hashlib
import os
from pathlib import Path
import subprocess

import numpy as np
from scripts.eval_log import logger
from src.artifacts import sha256_file as sha
from src.artifacts import immutable_json
from src.provenance import prediction_inputs, verify_prediction, environment

RUN = os.environ.get("AT_EVAL_RUN", "final-pafa-static")
LOG = Path("logs") / RUN
BASE = Path("outputs/train/grid-v1/beats+ft+pafa")
SC = Path("outputs/train/lodo-v1/beats+ft+pafa+sc+sc_static")


event, run = logger(LOG)


def export_ok(path, checkpoint_sha, sc, *, unit, checkpoint, dataset, sc_unit=None):
    if not path.exists():
        return False
    options = ({"a2": False, "tta_eq": 0, "sc_state": str(Path(sc_unit) / "sc_state.npz") if sc_unit else None,
                "sc_args": str(Path(sc_unit) / "train_args.json") if sc_unit else None} if dataset == "kauh" else {})
    inputs = prediction_inputs(unit, checkpoint, dataset, sc,
                               sc_unit=sc_unit, options=options)
    verify_prediction(path, inputs)
    with np.load(path, allow_pickle=True) as z:
        if str(z["ckpt_sha256"]) != checkpoint_sha or bool(z["sc"]) != sc:
            raise ValueError(f"Checkpoint or correction mismatch: {path}")
        p = z["probs"]
        if p.ndim != 2 or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-5):
            raise ValueError(f"Invalid probabilities: {path}")
    event(status="verified", artifact=str(path), sha256=sha(path))
    return True


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    os.environ["PYTHONPATH"] = "."
    event(status="campaign_started", run=RUN,
          git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
          script_sha256=sha(__file__))
    # Record the exact evaluation inputs, including all audio and labels used downstream.
    from src.paths import DATA
    from src.training.config import load_args
    data_files = sorted((DATA / "raw/kauh").rglob("*"))
    args = load_args(BASE / "seed0")
    folder = Path(args.data_folder)
    data_files += sorted((folder / "audio_test_data").glob("*"))
    data_files.append(folder / "official_split.txt")
    hashes = {str(p): sha(p) for p in data_files if p.is_file()}
    input_path = LOG / "input_hashes.json"
    immutable_json(input_path, hashes)
    immutable_json(LOG / "environment.json", environment())
    data_sha = sha(input_path)
    for seed in range(5):
        for name, parent, sc in (("baseline", BASE, False), ("auscult", SC, True)):
            unit = parent / f"seed{seed}"
            checkpoint = Path(str(unit).replace("outputs/", "checkpoints/", 1)) / "report_epoch_100.pth"
            checkpoint_sha = sha(checkpoint)
            inputs = [checkpoint, unit / "train_args.json", unit / "test_preds.npz", unit / "report.json"]
            if sc:
                inputs.append(unit / "sc_state.npz")
            frozen = Path("outputs/train") / RUN / name / f"seed{seed}"
            frozen.mkdir(parents=True, exist_ok=True)
            manifest = {"condition": name, "seed": seed, "epoch": 100, "unit": str(unit),
                        "checkpoint_sha256": checkpoint_sha, "data_manifest_sha256": data_sha,
                        "inputs": {str(p): sha(p) for p in inputs}, "sc": sc}
            if sc:
                with np.load(unit / "sc_state.npz") as z:
                    if str(z["sc_mode"]) != "static":
                        raise ValueError(f"Expected frozen static correction: {unit}")
                    manifest["s_ref_sha256"] = hashlib.sha256(z["reference"].tobytes()).hexdigest()
            immutable_json(frozen / "freeze.json", manifest)
            event(status="frozen", condition=name, seed=seed, checkpoint_sha256=checkpoint_sha)
            export = unit / "export"
            if not (export / "test.npz").exists():
                run(["-m", "src.training.export_probs", unit, "--ckpt", checkpoint.name, "--num_workers", "0"])
            verify_prediction(export / "test.npz", prediction_inputs(unit, checkpoint, "cycles", sc,
                              options={"sc_test_profile": None}))
            if (export / "model.sha256").read_text().split()[0] != checkpoint_sha:
                raise ValueError(f"Cycle export checkpoint mismatch: {unit}")
            for dataset, module in (("screening", "screening"), ("kauh", "kauh_infer")):
                out = Path("outputs") / dataset / RUN / name / f"seed{seed}.npz"
                if not export_ok(out, checkpoint_sha, sc, unit=unit, checkpoint=checkpoint, dataset=dataset):
                    run(["-m", "src.evaluation." + module, *(["infer"] if dataset == "screening" else []),
                         unit, "--ckpt", checkpoint.name, "--out", out, *(["--sc"] if sc else [])])
                    export_ok(out, checkpoint_sha, sc, unit=unit, checkpoint=checkpoint, dataset=dataset)
        # Isolation contrast: baseline weights with static SC from the corresponding SC run.
        unit, sc_unit = BASE / f"seed{seed}", SC / f"seed{seed}"
        checkpoint = Path(str(unit).replace("outputs/", "checkpoints/", 1)) / "report_epoch_100.pth"
        out = Path("outputs/kauh") / RUN / "baseline_ttsc" / f"seed{seed}.npz"
        if not export_ok(out, sha(checkpoint), True, unit=unit, checkpoint=checkpoint, dataset="kauh", sc_unit=sc_unit):
            run(["-m", "src.evaluation.kauh_infer", unit, "--ckpt", checkpoint.name, "--out", out,
                 "--sc", "--sc_state", sc_unit / "sc_state.npz", "--sc_args", sc_unit / "train_args.json"])
            export_ok(out, sha(checkpoint), True, unit=unit, checkpoint=checkpoint, dataset="kauh", sc_unit=sc_unit)
    run(["-m", "src.evaluation.kauh", "--dir", f"outputs/kauh/{RUN}", "--base", "baseline",
         "--arms", "auscult", "baseline_ttsc", "--boot", "1000", "--null_splits", "200"])
    run(["-m", "src.evaluation.screening", "analyse", "--dir", f"outputs/screening/{RUN}",
         "--kauh_dir", f"outputs/kauh/{RUN}", "--pairs", "baseline:baseline", "baseline:baseline_ttsc", "auscult:auscult"])
    run(["-m", "api.bundle", "--out", f"app/bundle-{RUN}"])
    event(status="campaign_complete", run=RUN)


if __name__ == "__main__":
    main()
