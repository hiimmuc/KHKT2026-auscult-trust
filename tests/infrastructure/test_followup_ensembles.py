"""Fixed ensemble weights, strict cohorts/seeds, and paired comparisons."""
import json

import numpy as np
import pytest

from scripts.followup_ensembles import GROUPS, aligned_mean, common_patient_cohort, evaluate, paired_contrast, require_seeds, template


def arrays(seed=0, correct=True):
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.eye(4)[y]*.8+.05
    if not correct:
        p = p[:, [1, 0, 2, 3]]
    return dict(probs=p, labels=y, patient=np.array(['a', 'a', 'b', 'b', 'c', 'd']),
                device=np.array([0]*6), cycle_id=np.array([f'record:{i}' for i in range(6)]), seed=seed)


def test_individual_members_get_equal_weight():
    members = [arrays(), arrays(correct=False), arrays()]
    mean = aligned_mean(members)
    np.testing.assert_allclose(mean['probs'], np.mean([m['probs'] for m in members], axis=0))
    assert len(GROUPS) == 6
    assert len(GROUPS['E1+gain6_cv66']) == 4
    assert len(GROUPS['E1_E3+LLRD_cv21']) == 6


@pytest.mark.parametrize('key', ['labels', 'patient', 'device', 'cycle_id'])
def test_rejects_mismatched_metadata(key):
    first, second = arrays(), arrays()
    second[key] = second[key][::-1]
    if np.array_equal(first[key], second[key]):
        second[key] = second[key] + 1
    with pytest.raises(ValueError, match=key):
        aligned_mean([first, second])
    del second[key]
    with pytest.raises(ValueError, match=key):
        aligned_mean([first, second])


def test_rejects_missing_duplicate_or_extra_seeds():
    for seeds in ([0, 1, 2, 3], [0, 1, 2, 3, 3], [0, 1, 2, 3, 4, 5]):
        with pytest.raises(ValueError, match='exactly seeds'):
            require_seeds([dict(seed=s) for s in seeds])


def test_paired_patient_and_seed_bootstrap_is_directional_and_reproducible():
    left, right = [arrays(s, False) for s in range(5)], [arrays(s) for s in range(5)]
    for row in left+right:
        row['pred'] = row['probs'].argmax(1)
    result = paired_contrast(left, right, draws=100)
    assert result['delta_score'] == 100
    assert result['ci95'] == [100, 100]
    assert result == paired_contrast(left, right, draws=100)
    right[0]['patient'] = right[0]['patient'][::-1]
    with pytest.raises(ValueError, match='alignment'):
        paired_contrast(left, right, draws=10)


def test_invalid_probability_is_rejected():
    row = arrays()
    row['probs'][0, 0] = np.nan
    with pytest.raises(ValueError, match='probabilities'):
        aligned_mean([row])


@pytest.mark.parametrize('common_patients', [False, True])
def test_complete_cycle_grid_keeps_missing_recording_exports_pending(tmp_path, monkeypatch, common_patients):
    import scripts.followup_ensembles as followup
    spec = template(tmp_path, 'anchor/seed{seed}', 'anchor+lodo_{device}/seed{seed}',
                    'llrd/seed{seed}', 'llrd+lodo_{device}/seed{seed}', common_patients)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(spec))
    def member(spec, subset, seed, folder=None):
        return {**arrays(seed), 'epoch': spec.get('epoch', 21), 'files': [str(manifest)]}
    monkeypatch.setattr(followup, 'cycle_case', member)
    def missing(*args):
        raise FileNotFoundError('recording exports not ready')
    monkeypatch.setattr(followup, 'recording_arrays', missing)
    receipt = evaluate(manifest, tmp_path / 'out', draws=10)
    assert receipt['status'] == 'pending'
    assert len(receipt['rows']) == 150 and len(receipt['paired_contrasts']) == 30
    assert len(receipt['pending']) == 30
    assert all(row.get('dataset') == 'screening+KAUH' for row in receipt['pending'])
    assert receipt['rows'][1]['weights'] == [1/3]*3
    assert (tmp_path / 'out/receipt.json').exists()


def test_common_cohort_drops_entire_partial_patient_and_reports_losses():
    first = arrays()
    positions = np.array([5, 4, 3, 2, 0])
    second = {key: value[positions] if isinstance(value, np.ndarray) else value for key, value in first.items()}
    selected, audit = common_patient_cohort({'old': first, 'new': second})
    assert audit['retained_cycles'] == 4
    assert audit['retained_patients'] == ['b', 'c', 'd']
    assert audit['members']['old']['dropped_patients'] == ['a']
    for key in ('probs', 'labels', 'patient', 'device', 'cycle_id'):
        np.testing.assert_array_equal(selected['old'][key], selected['new'][key])


def test_common_cohort_rejects_shared_id_conflicts_and_duplicate_ids():
    first, second = arrays(), arrays()
    second['labels'][0] = 1
    with pytest.raises(ValueError, match='metadata mismatch'):
        common_patient_cohort({'old': first, 'new': second})
    second = arrays()
    second['cycle_id'][1] = second['cycle_id'][0]
    with pytest.raises(ValueError, match='Duplicate'):
        common_patient_cohort({'old': first, 'new': second})


def test_trusted_recording_export_accepts_legacy_object_strings(tmp_path):
    from scripts.followup_ensembles import recording_arrays
    path = tmp_path / 'screening.npz'
    np.savez(path, probs=np.array([[.7, .1, .1, .1]]), label=np.array([0]),
             patient=np.array(['105'], dtype=object), stem=np.array(['105_record'], dtype=object),
             window=np.array([0]), device=np.array([0]))
    arrays, keys = recording_arrays(path, 'screening')
    assert arrays['patient'].tolist() == ['105']
    assert 'stem' in keys and 'window' in keys
