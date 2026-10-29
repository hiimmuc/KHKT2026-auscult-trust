"""Official, unseen-patient, best-on-test and LODO tables for the distilled student, its no-SC twin and the references (markdown on stdout).

    PYTHONPATH=. .venv-train/bin/python scripts/student_final_report.py [--h2]

Conditions (5 paired seeds 0-4): baseline = BEATs + PAFA no SC (epoch 100), previous = BEATs + PAFA + static SC (epoch 100), anchor = gain6-cv66
(epoch 66), student = kd_g4_k1 (epoch 66), twin = kd_g4_k1_nosc (epoch 66). Official numbers are the last recorded epoch of each unit; best-on-test
is the best epoch on the test split and is optimistic. With `--h2` a merged run directory of symlinks is built and `src.evaluation.lodo` is run
(null gate and H2 statistic per fold) with the twin and the baseline as bases.
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from src.processing.icbhi import DEVICES
from src.processing.splits import both_sides_patients
from src.training.reporting import per_device_report

T = Path("outputs/train")
NEW = T / "student-final-20261029-final"
OLD = T / "distill-20261028-final"
SEEDS = range(5)
OFFICIAL = {
    "baseline": lambda s: T / "grid-v1/beats+ft+pafa" / f"seed{s}",
    "previous": lambda s: T / "lodo-v1/beats+ft+pafa+sc+sc_static" / f"seed{s}",
    "anchor": lambda s: T / "followup-20261026-final/gain6-cv66" / f"seed{s}",
    "student": lambda s: OLD / "kd_g4_k1" / f"seed{s}",
    "twin": lambda s: NEW / "kd_g4_k1_nosc" / f"seed{s}",
}
LODO = {
    "baseline": lambda d, s: T / "lodo-v1" / f"beats+ft+pafa+lodo_{d.lower()}" / f"seed{s}",
    "previous": lambda d, s: T / "lodo-v1" / f"beats+ft+pafa+sc+sc_static+lodo_{d.lower()}" / f"seed{s}",
    "anchor": lambda d, s: T / "followup-20261026-final" / f"gain6-cv66+lodo_{d.lower()}" / f"seed{s}",
    "student": lambda d, s: (OLD if s < 3 else NEW) / f"kd_g4_k1+lodo_{d.lower()}" / f"seed{s}",
    "twin": lambda d, s: NEW / f"kd_g4_k1_nosc+lodo_{d.lower()}" / f"seed{s}",
}


def ms(v):
    v = np.asarray(v, float)
    return f"{v.mean():.1f} ± {v.std(ddof=1):.1f}" if len(v) > 1 else f"{v.mean():.1f}"


def official(both):
    rows = {}
    for name, path in OFFICIAL.items():
        rows[name] = []
        for s in SEEDS:
            z = np.load(path(s) / "test_preds.npz", allow_pickle=True)
            y, dev, pat = z["labels"], z["device"], z["patient"]
            last = z["preds"][-1]
            rep = per_device_report(y, last, dev)
            keep = ~np.isin(pat, list(both))
            un = per_device_report(y[keep], last[keep], dev[keep])["all"]["score"]
            best = max(per_device_report(y, p, dev)["all"]["score"] for p in z["preds"])
            rows[name].append({**{k: rep["all"][k] for k in ("sp", "se", "score", "hs")},
                               "worst": min(v["score"] for n, v in rep.items() if n != "all"), "unseen": un, "best": best})
    return rows


def lodo_scores():
    out = {}
    for name, path in LODO.items():
        out[name] = {d: [] for d in DEVICES}
        for d in DEVICES:
            for s in SEEDS:
                unit = path(d, s)
                if not (unit / "report.json").exists():
                    continue
                fixed = json.loads((unit / "report.json").read_text())["fixed"]
                assert len(fixed) == 1, unit
                out[name][d].append((s, next(iter(fixed.values()))["test"]["all"]["score"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h2", action="store_true")
    a = ap.parse_args()
    both = both_sides_patients()
    rows = official(both)
    print("## Official split (5 paired seeds; last recorded epoch; unseen-patient = without patients %s; best-on-test is optimistic)\n" % ", ".join(sorted(both)))
    print("| Condition | Sp | Se | Score | HS | Worst-dev Score | Unseen-patient Score | Best on test [optimistic] |\n|" + "---|" * 8)
    for name, r in rows.items():
        print(f"| {name} | " + " | ".join(ms([x[k] for x in r]) for k in ("sp", "se", "score", "hs", "worst", "unseen", "best")) + " |")
    print("\nPaired differences (mean ± SD over seeds; positive = first condition higher):\n")
    print("| Contrast | Δ Score | Δ Unseen-patient | Δ Sp | Δ Se | seeds above 0 |\n|" + "---|" * 6)
    for x, y in (("student", "twin"), ("student", "anchor"), ("student", "previous"), ("student", "baseline"), ("twin", "baseline"), ("previous", "baseline")):
        d = lambda k: [p[k] - q[k] for p, q in zip(rows[x], rows[y])]
        print(f"| {x} - {y} | {ms(d('score'))} | {ms(d('unseen'))} | {ms(d('sp'))} | {ms(d('se'))} | {sum(v > 0 for v in d('score'))}/5 |")
    sc = lodo_scores()
    print("\n## LODO (Score on the held-out device; seeds available per cell in brackets)\n")
    print("| Condition | " + " | ".join(DEVICES) + " | Macro (seeds with all 4 devices) |\n|" + "---|" * (len(DEVICES) + 2))
    macro = {}
    for name, per in sc.items():
        cells, seeds_all = [], None
        for d in DEVICES:
            cells.append(f"{ms([v for _, v in per[d]])} [{len(per[d])}]")
            ss = {s for s, _ in per[d]}
            seeds_all = ss if seeds_all is None else seeds_all & ss
        macro[name] = {s: float(np.mean([dict(per[d])[s] for d in DEVICES])) for s in sorted(seeds_all or [])}
        print(f"| {name} | " + " | ".join(cells) + f" | {ms(list(macro[name].values())) if macro[name] else 'n/a'} [{len(macro[name])}] |")
    print("\nPaired macro differences over the common seeds:\n")
    print("| Contrast | Δ macro Score | per-seed Δ |\n|---|---|---|")
    for x, y in (("student", "twin"), ("student", "anchor"), ("student", "previous"), ("student", "baseline"), ("twin", "baseline"), ("previous", "baseline")):
        common = sorted(set(macro[x]) & set(macro[y]))
        if common:
            d = [macro[x][s] - macro[y][s] for s in common]
            print(f"| {x} - {y} | {ms(d)} [{len(common)} seeds] | " + " / ".join(f"{v:+.2f}" for v in d) + " |")
    if a.h2:
        merged = T / "student-lodo-merged"
        for name, path in LODO.items():
            for d in DEVICES:
                for s in SEEDS:
                    unit = path(d, s)
                    link = merged / f"{name}+lodo_{d.lower()}" / f"seed{s}"
                    if unit.exists() and not link.exists():
                        link.parent.mkdir(parents=True, exist_ok=True)
                        link.symlink_to(unit.resolve())
        for base, arms, stem in (("twin", ["student"], "h2_student_vs_twin"), ("baseline", ["previous", "anchor", "student", "twin"], "h2_vs_baseline")):
            print(f"\n## H2 / null gate, base = {base}\n")
            subprocess.run([".venv-train/bin/python", "-m", "src.evaluation.lodo", "--run", "student-lodo-merged", "--base", base,
                            "--arms", *arms, "--out", stem], check=True, env={**__import__("os").environ, "PYTHONPATH": ".", "CUDA_VISIBLE_DEVICES": ""})


if __name__ == "__main__":
    main()
