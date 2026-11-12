"""Recording-level screening view: ICBHI 8 s windows, recording p(abnormal), 2-class split conformal and triage, and the
ICBHI-calibrated threshold applied to every KAUH filter (deployment check).

    python -m auscult_trust.evaluation.screening --pairs p0:p0 p0:p0_sc auscult:auscult

Input: window softmax exports of `auscult_trust.pipelines.screening_infer` (ICBHI) and `auscult_trust.pipelines.kauh_infer` (KAUH).

Recording p(abnormal) = mean over windows of 1 - p(normal) (`max` as a sensitivity view). E0: the ICBHI test patients are
split 20 times into calibration and evaluation halves stratified by device. LAC split conformal (alpha 0.1) on the
calibration recordings. Applied to (a) the ICBHI evaluation half and (b) every KAUH filter (`outputs/kauh/final/<cond>`).
Triage: {normal} no referral, {abnormal} refer, {both} uncertain, {} re-record. Reported: coverage, prior-matched Delta,
set size, referral / uncertain / re-record rates, screening Se (abnormal in the set) and Sp (set = {normal}), and argmax
Se / Sp at p(abnormal) > 0.5.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from auscult_trust.evaluation.conformal import lac_scores, predict_sets, split_threshold
from auscult_trust.evaluation.kauh import load as load_kauh
from auscult_trust.evaluation.recordings import icbhi_recordings, kauh_recordings
from auscult_trust.evaluation.shift import prior_matched_coverage
from auscult_trust.processing.splits import calibration_eval_splits

ALPHA = 0.1


def triage(thr, p, y, prior):
    sets = predict_sets(p, thr)
    n = sets.sum(1)
    ab, no = y == 1, y == 0
    return {
        "coverage": float(sets[np.arange(len(y)), y].mean()),
        "delta": float((1 - ALPHA) - prior_matched_coverage(sets, y, prior)),
        "size": float(n.mean()),
        "refer": float((sets[:, 1] & ~sets[:, 0]).mean()),
        "no_referral": float((sets[:, 0] & ~sets[:, 1]).mean()),
        "uncertain": float((n == 2).mean()),
        "rerecord": float((n == 0).mean()),
        "screen_se": float(sets[ab, 1].mean()) if ab.any() else float("nan"),
        "screen_sp": float((sets[no, 0] & ~sets[no, 1]).mean()) if no.any() else float("nan"),
        "argmax_se": float((p[ab, 1] > 0.5).mean()) if ab.any() else float("nan"),
        "argmax_sp": float((p[no, 1] <= 0.5).mean()) if no.any() else float("nan"),
        "abnormal_share": float(ab.mean()),
    }


def analyse(a):
    root, kroot = Path(a.dir), Path(a.kauh_dir)
    keys = (
        "coverage",
        "delta",
        "size",
        "refer",
        "uncertain",
        "rerecord",
        "screen_se",
        "screen_sp",
        "argmax_se",
        "argmax_sp",
        "abnormal_share",
    )
    L = [
        "| ICBHI calibration | Applied to | Aggregation | Seeds | " + " | ".join(keys) + " |",
        "|" + "---|" * (len(keys) + 4),
    ]
    res = {}
    for pair in a.pairs:
        cal_cond, kauh_cond = pair.split(":")
        for agg in ("mean", "max"):
            out = {}
            seeds = sorted(int(p.stem[4:]) for p in (root / cal_cond).glob("seed*.npz"))
            for s in seeds:
                ic = icbhi_recordings(
                    dict(np.load(root / cal_cond / "seed{}.npz".format(s), allow_pickle=True)), agg
                )
                kf = kroot / kauh_cond / "seed{}.npz".format(s)
                kr = kauh_recordings(load_kauh(kf), agg) if kf.exists() else None
                df = pd.DataFrame({"patient": ic["patient"], "device": ic["device"]})
                for sp in calibration_eval_splits(df, n_splits=20, seed=0):
                    ci, ei = np.isin(ic["patient"], sp["calibration"]), np.isin(
                        ic["patient"], sp["evaluation"]
                    )
                    thr = split_threshold(lac_scores(ic["p"][ci], ic["y"][ci]), ALPHA)
                    prior = np.bincount(ic["y"][ci], minlength=2) / ci.sum()
                    out.setdefault("ICBHI evaluation half", []).append(
                        triage(thr, ic["p"][ei], ic["y"][ei], prior)
                    )
                    if kr is not None:
                        for f in sorted(set(kr["filter"])):
                            m = kr["filter"] == f
                            out.setdefault("KAUH {} ({})".format(f, kauh_cond), []).append(
                                triage(thr, kr["p"][m], kr["y"][m], prior)
                            )
            for target, rows in out.items():
                mean = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
                res["{}|{}|{}".format(cal_cond, target, agg)] = mean
                L.append(
                    "| {} | {} | {} | {} | ".format(cal_cond, target, agg, len(seeds))
                    + " | ".join("{:.3f}".format(mean[k]) for k in keys)
                    + " |"
                )
    md = "\n".join(L) + "\n"
    (root / "screening_summary.md").write_text(md)
    (root / "screening_summary.json").write_text(json.dumps(res, indent=1))
    print(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/screening")
    ap.add_argument("--kauh_dir", default="outputs/kauh/final")
    ap.add_argument(
        "--pairs",
        nargs="+",
        default=["p0:p0", "p0:p0_sc", "auscult:auscult"],
        help="<ICBHI condition>:<KAUH condition>; the ICBHI calibration of the weights, applied to that KAUH condition",
    )
    analyse(ap.parse_args())


if __name__ == "__main__":
    main()
