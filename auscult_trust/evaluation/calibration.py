"""Split-conformal coverage reports and patient-level calibration nulls."""

import numpy as np
from auscult_trust.evaluation.conformal import lac_scores, predict_sets, split_threshold
from auscult_trust.evaluation.shift import prior_matched_coverage, set_stats

ALPHA = 0.1


def coverage(cal_p, cal_y, test_p, test_y, alpha=ALPHA):
    """Split-conformal result of one calibration/test pair.

    Returns:
        Dict `coverage`, `delta` (nominal minus prior-matched coverage; positive = under-coverage), and set statistics.
    """
    thr = split_threshold(lac_scores(cal_p, cal_y), alpha)
    sets = predict_sets(test_p, thr)
    prior = np.bincount(cal_y, minlength=cal_p.shape[1]) / len(cal_y)
    hit = sets[np.arange(len(test_y)), test_y]
    return {
        "coverage": float(hit.mean()),
        "delta": float((1 - alpha) - prior_matched_coverage(sets, test_y, prior)),
        **set_stats(sets),
    }


def robust_coverage(cal_p, cal_y, cal_dev, test_p, test_y, alpha=ALPHA, min_n=30):
    """Device-robust split conformal: threshold = max over source devices (>= `min_n` calibration cycles) of the per-device
    LAC quantile. Returns the same keys as `coverage`."""
    scores = lac_scores(cal_p, cal_y)
    devs = [d for d in np.unique(cal_dev) if (cal_dev == d).sum() >= min_n]
    thr = (
        max(split_threshold(scores[cal_dev == d], alpha) for d in devs)
        if devs
        else split_threshold(scores, alpha)
    )
    sets = predict_sets(test_p, thr)
    prior = np.bincount(cal_y, minlength=cal_p.shape[1]) / len(cal_y)
    return {
        "coverage": float(sets[np.arange(len(test_y)), test_y].mean()),
        "delta": float((1 - alpha) - prior_matched_coverage(sets, test_y, prior)),
        **set_stats(sets),
    }


def patient_rows(patients):
    ids, inv = np.unique(patients, return_inverse=True)
    return [np.flatnonzero(inv == k) for k in range(len(ids))]


def resample_rows(groups, rng):
    pick = rng.integers(0, len(groups), len(groups))
    return np.concatenate([groups[k] for k in pick])


def null_delta_p95(u, n_splits=200, alpha=ALPHA, seed=0):
    """95th percentile of |Delta| when the calibration patients are split at random (no device shift)."""
    groups = patient_rows(u["cal_pat"])
    n_test = max(1, min(len(set(u["test_pat"])), len(groups) // 2))
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_splits):
        order = rng.permutation(len(groups))
        te = np.concatenate([groups[k] for k in order[:n_test]])
        ca = np.concatenate([groups[k] for k in order[n_test:]])
        out.append(
            abs(
                coverage(u["cal_p"][ca], u["cal_y"][ca], u["cal_p"][te], u["cal_y"][te], alpha)[
                    "delta"
                ]
            )
        )
    return float(np.percentile(out, 95))
