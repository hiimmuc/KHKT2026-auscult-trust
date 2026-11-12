"""Index every training unit under outputs/train: one row per run/cell/unit with completeness, last-epoch Score and checkpoint.

    python scripts/index_results.py            # writes outputs/index.csv
"""
import csv
import json

from auscult_trust.paths import CHECKPOINTS, OUTPUTS


def row(unit):
    cell, run = unit.parent, unit.parent.parent
    out = {"run": run.name, "cell": cell.name, "unit": unit.name,
           "completion": (unit / "completion.json").exists(), "preds": (unit / "test_preds.npz").exists(),
           "epoch": "", "score": "", "ckpt": any((CHECKPOINTS / "train" / run.name / cell.name / unit.name).glob("*.pth"))}
    try:
        last = json.loads((unit / "report.json").read_text())["last"]
        out.update(epoch=last["epoch"], score=round(last["test"]["all"]["score"], 2))
    except (OSError, KeyError, TypeError, ValueError):
        pass
    return out


def main():
    units = sorted(p.parent for p in (OUTPUTS / "train").glob("*/*/*/report.json") if not p.parts[-4] == "latest")
    rows = [row(u) for u in units]
    path = OUTPUTS / "index.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} units -> {path}; with preds {sum(r['preds'] for r in rows)}, completion {sum(r['completion'] for r in rows)}, ckpt {sum(r['ckpt'] for r in rows)}")


if __name__ == "__main__":
    main()
