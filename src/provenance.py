"""Research input fingerprints, runtime environments, and prediction certificates."""

from importlib.metadata import distributions
import json
from pathlib import Path
import platform
import subprocess

from src.artifacts import atomic_json, file_hashes, sha256_file, verify_file
from src.paths import REPO, REPOS, DATA


def environment():
    """Resolved packages and local reference revisions; no environment variables or tokens."""
    refs = {}
    if REPOS.exists():
        for root in sorted(REPOS.iterdir()):
            if (root / ".git").exists():
                def git(*args):
                    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
                changed = [line for line in git("status", "--porcelain").splitlines()
                           if "__pycache__/" not in line and not line.endswith(".pyc")]
                refs[root.name] = {"commit": git("rev-parse", "HEAD"), "dirty": bool(changed)}
    hardware = {"machine": platform.machine()}
    try:
        import torch
        hardware.update(cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                        gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    except ImportError:
        pass
    return {"python": platform.python_version(), "platform": platform.platform(), "hardware": hardware,
            "packages": {d.metadata["Name"]: d.version for d in distributions() if d.metadata["Name"]},
            "references": refs}


def prediction_inputs(unit, checkpoint, dataset, sc=False, sc_unit=None, options=None):
    """Fingerprint data, arguments, correction state, code, and model inputs before inference."""
    unit = Path(unit)
    args_path = unit / "train_args.json"
    args = json.loads(args_path.read_text())
    folder = Path(args["data_folder"])
    files = [args_path] + ([checkpoint] if checkpoint is not None else [])
    files += [value for value in args.values() if isinstance(value, str) and Path(value).is_file()]
    files += args.get("config_files", [])
    if sc:
        files += [Path(sc_unit or unit) / "sc_state.npz", Path(sc_unit or unit) / "train_args.json"]
        files += [options[k] for k in ("sc_state", "sc_args") if options and options.get(k)]
    roots = [DATA / "raw/kauh"] if dataset == "kauh" else [folder / "audio_test_data"]
    if dataset != "kauh":
        files.append(folder / "official_split.txt")
        if (folder / "patient_list_foldwise.txt").exists():
            files.append(folder / "patient_list_foldwise.txt")
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
        files += [p for p in root.rglob("*") if p.is_file()]
    # Constructors can read pretrained files even when checkpoint weights replace them.
    model_root = DATA / "models" / {"beats": "beats", "ast": "ast", "htsat": "htsat",
                                    "opera_ct": "opera"}.get(args.get("encoder"), args.get("encoder", ""))
    files += [p for p in model_root.rglob("*") if p.is_file()]
    files += list((REPO / "src").rglob("*.py"))
    files += [p for p in REPOS.rglob("*.py") if ".git" not in p.parts]
    return {"schema": 1, "dataset": dataset, "sc": bool(sc), "options": options or {},
            "files": file_hashes(files), "environment": environment()}


def record_prediction(path, inputs, **context):
    if context and prediction_inputs(**context) != inputs:
        raise ValueError("Prediction inputs changed during inference; output has not been certified")
    atomic_json(str(path) + ".provenance.json", {"inputs": inputs, "sha256": sha256_file(path)})


def verify_prediction(path, inputs):
    record = Path(str(path) + ".provenance.json")
    if not record.exists():
        raise ValueError(f"Missing provenance for {path}; rerun its exporter explicitly")
    saved = json.loads(record.read_text())
    if saved["inputs"] != inputs:
        raise ValueError(f"Stale prediction inputs: {path}; use a new output or rerun its exporter")
    verify_file(path, saved["sha256"])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Record the resolved research environment")
    parser.add_argument("--environment", type=Path, required=True, help="output JSON path")
    atomic_json(parser.parse_args().environment, environment())
