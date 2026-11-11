"""Follow-up role counts, source-only gates, native recovery, and complete receipts."""
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts import followup_campaign as campaign
from scripts import improvement_campaign as core
from src.artifacts import atomic_json, sha256_file


def state(tmp_path):
    return dict(name=campaign.NAME, snapshot=str(tmp_path), archive_machine='E', stack=[],
                machines={'B': {'local': True}, 'D': {'host': 'd'}, 'E': {'host': 'e'}, 'F': {'host': 'f'}},
                deadline='2026-10-26T23:00:00+07:00', recovery_report=str(tmp_path / 'recovery/report.json'),
                ensemble_manifest=str(tmp_path / 'ensemble.json'), selected={'anchor': 66}, baselines={})


def recovery(tmp_path):
    path = tmp_path / 'recovery/report.json'
    report = dict(status='verified', verified_files=11, verified_bytes=120,
                  unit_receipt_integrity={'verified': 3, 'units': 3, 'mismatches': []},
                  unmanifested_recovered_files=[], canonical_matches=10,
                  canonical_final_comparisons=[{'matches_canonical': True}] * 10,
                  failed_log_events=['historical OOM retained'])
    atomic_json(path, report)
    atomic_json(path.parent / 'metadata/source-manifest.json', {'file_count': 11, 'bytes': 120})
    return path, report


def test_initial_queue_has_three_aux_source_jobs_then_25_epoch66_anchor_jobs(tmp_path):
    spec = state(tmp_path)
    jobs = campaign.initial_jobs()
    assert len(jobs) == 28
    assert all(job['arm'] == 'event_heads' and job['seed'] == 0 and 'cohort' not in job for job in jobs[:3])
    assert {job['fold'] for job in jobs[:3]} == {0, 1, 2}
    assert all(job['arm'] == 'anchor' and job['epoch'] == 66 and job['cell'].startswith('gain6-cv66') for job in jobs[3:])
    assert {job['cohort'] for job in jobs[3:]} == {'official', *core.DEVICES}
    core.set_phase(spec, 'screen', jobs)
    assert {job['machine'] for job in jobs} <= {'B', 'D', 'E', 'F'}
    assert 'A' not in spec['machines'] and spec['archive_machine'] == 'E'


def test_native_recovery_requires_counts_receipts_and_canonical_matches(tmp_path):
    path, report = recovery(tmp_path)
    assert campaign.recovery_verified(path)
    for key, value in [('status', 'copying'), ('verified_files', 10), ('verified_bytes', 119),
                       ('canonical_matches', 9), ('unmanifested_recovered_files', ['extra']), ('errors', ['bad hash'])]:
        atomic_json(path, {**report, key: value})
        assert not campaign.recovery_verified(path)
    atomic_json(path, {**report, 'unit_receipt_integrity': {'verified': 2, 'units': 3, 'mismatches': []}})
    assert not campaign.recovery_verified(path)


def install_transport(monkeypatch, calls):
    monkeypatch.setattr(core, 'verify_source', lambda _: None)
    monkeypatch.setattr(core, 'collect', lambda _: calls.append('collect'))
    monkeypatch.setattr(core, 'verify_job', lambda *_: False)
    monkeypatch.setattr(core, 'launch', lambda *_: calls.append('launch'))


def test_archive_gate_precedes_any_collection_or_launch(tmp_path, monkeypatch):
    spec, calls = state(tmp_path), []
    core.set_phase(spec, 'screen', campaign.initial_jobs())
    install_transport(monkeypatch, calls)
    campaign.tick(spec, tmp_path / 'manifest.json')
    assert not calls and 'waiting_for_native_e_recovery' in spec['status']
    recovery(tmp_path)
    campaign.tick(spec, tmp_path / 'manifest.json', datetime(2026, 10, 26, 10, tzinfo=timezone.utc))
    assert calls == ['collect', 'launch']


@pytest.mark.parametrize('first_pass,confirm_pass', [(False, False), (True, False), (True, True)])
def test_source_only_transitions_keep_all_final_cohorts(tmp_path, monkeypatch, first_pass, confirm_pass):
    spec = state(tmp_path)
    core.set_phase(spec, 'screen', campaign.initial_jobs())
    decisions = iter([first_pass, confirm_pass])
    seen = []
    monkeypatch.setattr(core, 'source_reports', lambda spec, arm, seeds: seen.append((arm, seeds)) or [])
    monkeypatch.setattr(core, 'gate', lambda *_: {'pass': next(decisions), 'epoch': 17})
    campaign.transition(spec)
    if first_pass:
        assert spec['phase'] == 'confirm' and len(spec['jobs']) == 3
        assert all(job['arm'] == 'event_heads' and job['seed'] == 1 and 'cohort' not in job for job in spec['jobs'])
        campaign.transition(spec)
        if confirm_pass:
            assert spec['phase'] == 'aux_final' and len(spec['jobs']) == 25
            assert spec['selected'] == {'anchor': 66, 'event_heads': 17}
            campaign.transition(spec)
    assert spec['phase'] == 'report'
    assert len(spec['jobs']) == (50 if first_pass and confirm_pass else 25)
    assert all('cohort' in job for job in spec['jobs'])
    assert all(arm in ('event_heads', 'anchor') for arm, _ in seen)


def test_deadline_stops_admission_without_dropping_pending_work(tmp_path, monkeypatch):
    spec, calls = state(tmp_path), []
    recovery(tmp_path)
    core.set_phase(spec, 'screen', campaign.initial_jobs())
    install_transport(monkeypatch, calls)
    campaign.tick(spec, tmp_path / 'manifest.json', datetime(2026, 10, 26, 16, tzinfo=timezone.utc))
    assert calls == ['collect'] and spec['status']['admission_closed']
    assert len(spec['jobs']) == 28 and spec['phase'] == 'screen'


def test_model_subtask_receipt_does_not_complete_pending_ensembles(tmp_path, monkeypatch):
    spec = state(tmp_path)
    recovery(tmp_path)
    atomic_json(spec['ensemble_manifest'], {'fixed': True})
    spec['ensemble_manifest_sha256'] = sha256_file(spec['ensemble_manifest'])
    destination = tmp_path / 'outputs/evaluation' / campaign.NAME
    atomic_json(destination / 'summary.json', {'rows': []})
    atomic_json(destination / 'complete.json', {'summary_sha256': sha256_file(destination / 'summary.json')})
    calls = []
    def evaluate(argv, **kwargs):
        output = Path(argv[argv.index('--out') + 1])
        output.mkdir(parents=True, exist_ok=False)
        calls.append(output)
        atomic_json(output / 'receipt.json', {'status': 'pending' if len(calls) == 1 else 'complete',
            'pending': ['missing prediction'] if len(calls) == 1 else [],
            'inputs': {str(Path(spec['ensemble_manifest']).resolve()): spec['ensemble_manifest_sha256']}})
    monkeypatch.setattr(campaign.subprocess, 'run', evaluate)
    spec['phase'] = 'report'
    assert not campaign.finish(spec)
    assert spec['phase'] == 'report' and not (destination / 'followup-complete.json').exists()
    assert campaign.finish(spec) and spec['phase'] == 'complete'
    assert len(calls) == 2
    assert core.load(destination / 'ensembles-attempt-001/receipt.json')['status'] == 'pending'
    assert len(core.load(destination / 'followup-complete.json')['files']) == 3
    assert campaign.finish(spec) and len(calls) == 2  # Reuse a verified complete receipt.


def test_partial_ensemble_directory_is_preserved_and_complete_inputs_are_verified(tmp_path, monkeypatch):
    spec = state(tmp_path)
    recovery(tmp_path)
    atomic_json(spec['ensemble_manifest'], {'fixed': True})
    spec['ensemble_manifest_sha256'] = sha256_file(spec['ensemble_manifest'])
    destination = tmp_path / 'outputs/evaluation' / campaign.NAME
    atomic_json(destination / 'summary.json', {'rows': []})
    atomic_json(destination / 'complete.json', {'summary_sha256': sha256_file(destination / 'summary.json')})
    partial = destination / 'ensembles'
    partial.mkdir()
    (partial / 'unfinished.txt').write_text('preserve partial evidence')
    def evaluate(argv, **kwargs):
        output = Path(argv[argv.index('--out') + 1])
        output.mkdir(exist_ok=False)
        atomic_json(output / 'receipt.json', {'status': 'complete', 'pending': [],
            'inputs': {str(Path(spec['ensemble_manifest']).resolve()): spec['ensemble_manifest_sha256']}})
    monkeypatch.setattr(campaign.subprocess, 'run', evaluate)
    assert campaign.finish(spec)
    assert (destination / 'ensembles-attempt-001/unfinished.txt').read_text() == 'preserve partial evidence'
    Path(spec['ensemble_manifest']).write_text('changed after completion')
    with pytest.raises(ValueError, match='integrity'):
        campaign.finish(spec)


def test_prepare_reuses_six_source_folds_and_freezes_ensemble_plan(tmp_path, monkeypatch):
    from types import SimpleNamespace
    root, old = tmp_path / 'new', tmp_path / 'old'
    root.mkdir()
    for name in ('requirements-train.lock', 'requirements-references.lock.json', 'pyproject.toml'):
        (root / name).write_text('frozen')
    hosts = tmp_path / 'hosts.json'
    atomic_json(hosts, state(root)['machines'])
    previous = tmp_path / 'previous.json'
    atomic_json(previous, {'snapshot': str(old), 'name': core.NAME, 'audits': {'cv0': {}, 'cv1': {}, 'cv2': {}}})
    for seed in (0, 1):
        for fold in range(3):
            source = old / 'outputs/train' / f'{core.NAME}-s{seed}' / 'anchor' / f'cv{fold}'
            source.mkdir(parents=True)
            (source / 'report.json').write_text('{}')
            (source / 'val_preds.npz').write_bytes(b'original source OOF')
    baseline = tmp_path / 'baselines.json'
    atomic_json(baseline, {label: {cohort: {str(seed): str(tmp_path / label / cohort / f'seed{seed}') for seed in range(5)}
                        for cohort in ['official', *core.DEVICES]} for label in ('control', 'gain', 'llrd')})
    ensemble = tmp_path / 'ensemble.json'
    atomic_json(ensemble, {'fixed_groups': True})
    monkeypatch.setattr(core, 'audits_complete', lambda _: True)
    args = SimpleNamespace(snapshot=root, data='source-data', references='frozen-refs', hosts=hosts,
                           previous_manifest=previous, baselines=baseline, recovery_report=tmp_path / 'not-verified.json',
                           ensemble_manifest=ensemble, deadline='2026-10-26T23:00:00+07:00')
    campaign.prepare(args)
    saved = core.load(root / 'logs' / campaign.NAME / 'programme.json')
    assert len(saved['jobs']) == 28 and saved['archive_machine'] == 'E'
    assert len(saved['source_anchor_reuse']) == 2 and all(len(row['files']) == 6 for row in saved['source_anchor_reuse'])
    assert saved['ensemble_manifest_sha256'] == sha256_file(ensemble)
    for seed in (0, 1):
        alias = root / 'outputs/train' / f'{campaign.NAME}-s{seed}' / 'anchor'
        assert alias.is_symlink() and alias.resolve() == old / 'outputs/train' / f'{core.NAME}-s{seed}' / 'anchor'
