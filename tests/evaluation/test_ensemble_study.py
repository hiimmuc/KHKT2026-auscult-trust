import numpy as np
import pytest

from src.evaluation.ensemble_study import adaptation_split, average, k_for_rate, predict


def test_adaptation_is_patient_disjoint_and_seeded():
    patients = np.repeat(np.arange(10), 5)
    a, e = adaptation_split(patients, np.random.default_rng(5))
    assert not set(patients[a]) & set(patients[e])
    assert len(a) + len(e) == len(patients)
    np.testing.assert_array_equal(a, adaptation_split(patients, np.random.default_rng(5))[0])


def test_average_rejects_shuffled_cycle_ids():
    unit = {key: np.array([0, 1]) for key in ('cal_y', 'cal_pat', 'cal_dev', 'test_y', 'test_pat', 'cal_ids', 'test_ids')}
    unit.update(cal_p=np.full((2, 4), .25), test_p=np.full((2, 4), .25), files=[])
    with pytest.raises(ValueError, match='test_ids'):
        average([unit, {**unit, 'test_ids': unit['test_ids'][::-1]}])


def test_rate_rule_matches_existing_convention():
    p = np.random.default_rng(2).dirichlet(np.ones(4), 100)
    np.testing.assert_array_equal(predict(p), p.argmax(1))
    assert abs((predict(p, k_for_rate(p, .6)) == 0).mean() - .6) < .02
