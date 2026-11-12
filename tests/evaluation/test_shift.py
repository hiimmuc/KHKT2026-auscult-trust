import numpy as np
from auscult_trust.evaluation.shift import (
    prior_matched_coverage,
    set_stats,
)


def test_set_stats():
    sets = np.array([[1, 0], [1, 1], [0, 0], [0, 1]], bool)
    s = set_stats(sets)
    assert (s["size"], s["singleton"], s["empty"]) == (1.0, 0.5, 0.25)


def test_prior_matched_coverage_reweights_classes():
    y = np.array([0, 0, 0, 0, 1, 1])
    sets = np.zeros((6, 2), bool)
    sets[:4, 0] = True  # class 0 fully covered, class 1 never
    assert np.isclose(prior_matched_coverage(sets, y, [0.5, 0.5]), 0.5)
    assert np.isclose(prior_matched_coverage(sets, y, [0.9, 0.1]), 0.9)
