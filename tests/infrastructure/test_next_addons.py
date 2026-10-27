import numpy as np
import pytest
from scripts.next_addons import audit_fusion
from src.evaluation.ensemble_study import adaptation_split


def test_patient_split_is_label_free_disjoint_and_deterministic():
    patients = np.repeat(np.arange(10), 3)
    a, e = adaptation_split(patients, np.random.default_rng(20261026))
    assert not set(patients[a]) & set(patients[e])
    assert len(a) + len(e) == len(patients)
    assert np.array_equal(a, adaptation_split(patients, np.random.default_rng(20261026))[0])


def test_oof_inventory_does_not_admit_final_test_or_claim_certification(tmp_path):
    final = tmp_path / 'member/seed0/test_preds.npz'
    final.parent.mkdir(parents=True)
    final.touch()
    cv = tmp_path / 'member/cv0/val_preds.npz'
    cv.parent.mkdir(parents=True)
    cv.touch()
    report = audit_fusion([tmp_path])
    assert report['status'] == 'blocked'
    assert report['source_cv_prediction_files'] == [str(cv)]
    assert len(report['missing_complementary_member_oof']) == 5


def test_sc_spectrum_uses_only_unlabelled_adaptation_audio(monkeypatch):
    from argparse import Namespace
    from scripts.next_addons import fit_sc
    from src.processing import correction
    seen = []
    def spectrum(waves):
        seen.extend(waves)
        return np.ones(513)
    monkeypatch.setattr(correction, 'mean_spectrum', spectrum)
    args = Namespace(sc_reference='arithmetic', sc_mode='static', sc_limit_freq_low=50,
                     sc_limit_freq_high=2000, sc_limit_freq_diff=20)
    coefficients = fit_sc([1, 2, 999, 999], np.array(['B', 'E', 'B', 'E']), [0, 1], np.ones(513), args)
    assert seen == [1, 2]
    assert set(coefficients) == {'B', 'E'}
    assert all(np.array_equal(c, np.ones(513)) for c in coefficients.values())


def test_saved_summary_preserves_recording_ids_and_checks_frozen_patients(tmp_path):
    import json
    from scripts.next_addons import summarize_sc
    patients = np.array(['p0', 'p1', 'p2', 'p3'])
    (tmp_path / 'freeze.json').write_text(json.dumps({'evaluation_patients': patients.tolist()}))
    data = dict(label=np.array([0, 0, 1, 1]), patient=patients, stem=patients,
                filter=np.repeat('Bell', 4), window=np.zeros(4, dtype=int),
                probs=np.array([[.8, .1, .05, .05], [.8, .1, .05, .05], [.1, .8, .05, .05], [.1, .8, .05, .05]]))
    for seed in range(5):
        for mode in ('baseline', 'adapted'):
            np.savez(tmp_path / f'{mode}-seed{seed}.npz', **data)
    result = summarize_sc(tmp_path)
    assert result['Bell']['delta_score'] == 0
    assert result['Bell']['baseline']['score']['mean'] == 100
    (tmp_path / 'freeze.json').write_text(json.dumps({'evaluation_patients': ['p0']}))
    with pytest.raises(ValueError, match='frozen evaluation patients'):
        summarize_sc(tmp_path)


def test_teacher_audit_accepts_complete_aligned_oof_and_rejects_misalignment(tmp_path):
    import json
    from scripts.next_addons import audit_teachers
    from src.processing.splits import cv_split
    patients = np.repeat(np.arange(9).astype(str), 2)
    devices = np.zeros(len(patients), dtype=int)
    for cell in ('anchor', 'llrd'):
        for fold in range(3):
            _, val = cv_split(patients, devices, 3, fold)
            unit = tmp_path / 'outputs/train/improvements-20261026-s0' / cell / f'cv{fold}'
            unit.mkdir(parents=True)
            np.savez(unit / 'val_preds.npz', epochs=np.array([21, 66]),
                     probs=np.full((2, len(val), 4), .25), labels=np.zeros(len(val), dtype=int),
                     device=devices[val], patient=patients[val], stem=patients[val])
            (unit / 'train_args.json').write_text(json.dumps({'cv_folds': 3, 'cv_fold': fold}))
            (unit / 'report.json').write_text('{}')
    out = tmp_path / 'report'
    out.mkdir()
    assert audit_teachers(tmp_path, out)['members']['anchor']['patients'] == 9
    path = tmp_path / 'outputs/train/improvements-20261026-s0/llrd/cv0/val_preds.npz'
    with np.load(path) as source:
        data = dict(source)
    data['labels'] = np.ones_like(data['labels'])
    np.savez(path, **data)
    with pytest.raises(ValueError, match='alignment mismatch'):
        audit_teachers(tmp_path, out)
