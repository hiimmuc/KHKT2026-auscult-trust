"""Coverage and set-size metrics on a new device: Delta = (1 - alpha) - coverage."""

import numpy as np


def set_stats(sets):
    """Mean size, singleton rate and empty-set rate of prediction sets.

    Args:
        sets: (n, K) boolean membership matrix.

    Returns:
        Dict `size`, `singleton`, `empty`.
    """
    n = sets.sum(1)
    return {
        "size": float(n.mean()),
        "singleton": float((n == 1).mean()),
        "empty": float((n == 0).mean()),
    }


def prior_matched_coverage(sets, y, target_prior):
    """Coverage re-weighted so the class mix equals `target_prior`. Evaluation only, never used to calibrate.

    Separates the device effect from the class-proportion effect: compare with plain coverage on the same rows.

    Args:
        sets: (n, K) boolean membership matrix.
        y: (n,) true labels.
        target_prior: (K,) class proportions to match, e.g. those of the calibration data.

    Returns:
        Float; classes absent from `y` are dropped and the rest renormalised.
    """
    K = sets.shape[1]
    per = np.array([sets[y == k, k].mean() if (y == k).any() else np.nan for k in range(K)])
    w = np.where(np.isnan(per), 0.0, np.asarray(target_prior, float))
    return float((np.nan_to_num(per) * w).sum() / w.sum())
