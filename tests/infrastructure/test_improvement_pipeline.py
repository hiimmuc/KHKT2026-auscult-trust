import importlib.util
import json
import sys

import pytest

spec = importlib.util.spec_from_file_location('improvement_unit', 'scripts/improvement_unit.py')
unit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(unit)


def test_inference_receipt_reuses_verified_output_and_rejects_changes(tmp_path):
    artifact, receipt = tmp_path / 'prediction', tmp_path / 'receipt.json'
    command = [sys.executable, '-c', f'from pathlib import Path; Path({str(artifact)!r}).write_text("prediction")']
    unit.checked_step(command, [artifact], 'frozen', receipt)
    assert json.loads(receipt.read_text())['checkpoint_sha256'] == 'frozen'
    unit.checked_step(command, [artifact], 'frozen', receipt)
    with pytest.raises(ValueError, match='Changed frozen'):
        unit.checked_step(command, [artifact], 'changed', receipt)
    artifact.write_text('modified')
    with pytest.raises(ValueError):
        unit.checked_step(command, [artifact], 'frozen', receipt)


def test_paired_interval_is_zero_for_identical_models():
    import numpy as np
    report_spec = importlib.util.spec_from_file_location('improvement_report', 'scripts/improvement_report.py')
    report = importlib.util.module_from_spec(report_spec)
    report_spec.loader.exec_module(report)
    u = {'test_y': np.array([0, 1, 0, 1]), 'test_pat': np.array(['a', 'a', 'b', 'b']),
         'test_p': np.array([[.9, .1], [.1, .9], [.8, .2], [.2, .8]])}
    assert report.paired_score_interval([(u, u)], draws=20) == {'ci': [0., 0.], 'estimable_draws': 20}


def test_final_report_rebases_collected_receipts_and_chains_analyses(tmp_path, monkeypatch):
    import hashlib
    import numpy as np
    report_spec = importlib.util.spec_from_file_location('improvement_report_e2e', 'scripts/improvement_report.py')
    report = importlib.util.module_from_spec(report_spec)
    report_spec.loader.exec_module(report)
    source, snapshot = tmp_path / 'collected', tmp_path / 'frozen'
    manifest = tmp_path / 'manifest.json'
    spec = {'source': str(source), 'snapshot': str(snapshot), 'run': 'final', 'name': 'final',
            'devices': ['Meditron'], 'cells': {'control': 'control', 'gain': 'gain'}}
    y = np.tile(np.arange(4), 4)
    patient = np.repeat(np.array(['a', 'b', 'c', 'd']), 4)
    probs = np.eye(4)[y] * .8 + .05
    for cohort in ('official', 'Meditron'):
        for seed in range(5):
            for arm in ('control', 'gain'):
                cell = arm + ('' if cohort == 'official' else '+lodo_meditron')
                relative = report.Path('outputs/train/final') / cell / f'seed{seed}'
                unit = source / relative
                unit.mkdir(parents=True)
                if cohort == 'official':
                    (unit / 'export').mkdir()
                    np.savez(unit / 'export/test.npz', probs=probs, labels=y, patient=patient)
                    for name in ('screening.npz', 'kauh.npz'):
                        (unit / name).write_bytes(b'prediction')
                else:
                    for name in ('test_preds.npz', 'cal_preds.npz'):
                        np.savez(unit / name, epochs=[21], probs=probs[None], labels=y, patient=patient, device=np.zeros(len(y), int))
                    (unit / 'report.json').write_text(json.dumps({'fixed': {'21': {'test': {'all': {'score': 100.}}}}}))
                files = {str(snapshot / relative / p.relative_to(unit)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in unit.rglob('*') if p.is_file()}
                (unit / 'pipeline-complete.json').write_text(json.dumps({'files': files}))
    manifest.write_text(json.dumps(spec))
    commands = []
    monkeypatch.setattr(report.subprocess, 'run', lambda command, **kwargs: commands.append(command))
    monkeypatch.setattr(report.sys, 'argv', ['improvement_report.py', str(manifest)])
    report.main()
    summary = json.loads((source / 'outputs/evaluation/final/summary.json').read_text())
    assert summary['cohorts']['Meditron']['gain_minus_control_ci'] == [0., 0.]
    assert (source / 'outputs/evaluation/final/complete.json').exists()
    assert [c[2] for c in commands] == ['auscult_trust.evaluation.kauh', 'auscult_trust.evaluation.screening']
