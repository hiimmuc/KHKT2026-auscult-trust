import copy
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import improvement_campaign as campaign
from src.artifacts import atomic_json, sha256_file


def curve(value=60.0, epoch=7, recording=None):
    history = []
    for e in range(1, 101):
        history.append({'epoch': e, 'val': {'all': {'score': value if e == epoch else value - 2}},
                        'recording_val': {'all': {'score': (value if recording is None else recording) if e == epoch else value - 3}},
                        'test': {'all': {'score': 99999}}})
    return {'history': history}


def test_source_gate_does_not_round_threshold_or_use_test_metrics():
    anchor = [curve(60)] * 3
    below = [curve(60.4999)] * 3
    assert campaign.gate(below, anchor)['pass'] is False
    exact = [curve(60.5)] * 3
    result = campaign.gate(exact, anchor)
    assert result['pass'] is True
    assert result['epoch'] == 7
    for report in exact:
        for row in report['history']:
            row['test']['all']['score'] = -99999
    assert campaign.gate(exact, anchor) == result


def test_gate_needs_two_positive_folds_and_consistent_confirm_seeds():
    anchor = [curve()] * 3
    assert not campaign.gate([curve(62), curve(60), curve(60)], anchor)['pass']
    assert campaign.gate([curve(61), curve(61), curve(59.6)], anchor)['pass']
    assert not campaign.gate([curve(62)] * 3 + [curve(59.9)] * 3, anchor * 2)['pass']
    with pytest.raises(ValueError, match='three folds'):
        campaign.gate([curve()], anchor)


def test_recording_head_uses_binary_source_curve_and_cycle_guard():
    anchor = [curve(60, recording=70)] * 3
    head = [curve(59.5, recording=71)] * 3
    assert campaign.gate(head, anchor, recording=True)['pass']
    head = [curve(59.499, recording=71)] * 3
    result = campaign.gate(head, anchor, recording=True)
    assert result['pass']
    assert not result['cycle_eligible']


def test_gate_requires_complete_source_horizon_and_earliest_tie():
    report = curve()
    report['history'][2]['val']['all']['score'] = 60
    assert campaign.gate([report] * 3, [curve(59)] * 3)['epoch'] == 3
    report['history'].pop()
    with pytest.raises(ValueError, match='100-epoch'):
        campaign.source_curve(report)


def test_stack_excludes_conflicting_pooling_noise_and_recording_options():
    for arms in [('class_topk', 'class_attention'), ('band_noise', 'band_noise_consistency'),
                 ('recording_head', 'class_topk'), ('recording_head', 'class_attention')]:
        assert not campaign.compatible(arms)
        assert campaign.stack_members(list(arms)) == []
    assert campaign.stack_members(['llrd', 'patient_cap', 'class_topk']) == ['llrd', 'patient_cap', 'class_topk']
    assert campaign.stack_members(['class_topk', 'class_attention', 'llrd']) == ['class_topk', 'llrd']


def spec(tmp_path):
    return {'snapshot': str(tmp_path), 'machines': {'A': {'local': True}, 'B': {'host': 'b'}, 'D': {'host': 'd'}, 'F': {'host': 'f'}},
            'stack': [], 'baselines': {}, 'deadline': '2026-10-26T23:00:00+07:00'}


def test_phase_transitions_screen_confirm_stack_final_without_target_inputs(tmp_path, monkeypatch):
    state = spec(tmp_path)
    campaign.set_phase(state, 'screen', campaign.cv_jobs(['anchor', *campaign.ARMS], [0]))
    assert len(state['jobs']) == 27
    monkeypatch.setattr(campaign, 'source_reports', lambda state, arm, seeds:
                        [curve(60 if arm == 'anchor' else (61 if arm in ('llrd', 'patient_cap', 'stack') else 59))] * (3 * len(seeds)))
    campaign.transition(state)
    assert state['phase'] == 'confirm'
    assert state['shortlist'] == ['llrd', 'patient_cap']
    assert len(state['jobs']) == 9
    assert {j['seed'] for j in state['jobs']} == {1}
    campaign.transition(state)
    assert state['phase'] == 'stack'
    assert len(state['jobs']) == 6
    campaign.transition(state)
    assert state['phase'] == 'final'
    assert len(state['selected']) == 1
    assert len(state['jobs']) == 25
    assert {j['cohort'] for j in state['jobs']} == {'official', *campaign.DEVICES}
    assert all(j['epoch'] == 7 for j in state['jobs'])
    campaign.transition(state)
    assert state['phase'] == 'report'


def test_screen_no_winners_stops_without_target_evaluation(tmp_path, monkeypatch):
    state = spec(tmp_path)
    state['phase'] = 'screen'
    monkeypatch.setattr(campaign, 'source_reports', lambda *args: [curve()] * 3)
    campaign.transition(state)
    assert state['phase'] == 'complete_no_survivors'
    assert not state.get('selected')


def test_generated_commands_have_unique_identities_and_fixed_source_horizon(tmp_path):
    state = spec(tmp_path)
    jobs = campaign.cv_jobs(['anchor', *campaign.ARMS], [0, 1])
    identities = [(j['run'], j['cell'], j['unit']) for j in jobs]
    assert len(identities) == len(set(identities)) == 54
    for job in jobs:
        line = campaign.command(state, job)
        assert '--epochs 100' in line
        assert '--selection cv' in line
        assert 'improvement_unit.py' not in line
        assert 'PIPELINE_DONE=' in line and '/completion.json' in line
        assert '/pipeline-complete.json' not in line
        assert '--report_epochs' not in line
    job = campaign.final_jobs({'llrd': 17})[0]
    line = campaign.command(state, job)
    assert '--epochs 100' in line and '--report_epochs 17' in line
    assert 'PIPELINE_DONE=' in line and 'scripts/improvement_unit.py' in line


def test_worker_reuses_native_sweep_admission_limits(tmp_path, monkeypatch):
    state = spec(tmp_path)
    state.update(phase='screen', source_hashes={}, data='/data')
    calls = []
    monkeypatch.setattr(campaign.subprocess, 'run', lambda *a, **kw: calls.append((a, kw)))
    campaign.worker(state, 'F', 'screen')
    command, kwargs = calls[0][0][0], calls[0][1]
    assert 'scripts/sweep.py' in command
    for flag, value in [('--workers', '2'), ('--gpu-gb', '32'), ('--disk-gb', '30'), ('--ram-gb', '20')]:
        assert command[command.index(flag) + 1] == value
    assert kwargs['env']['AT_DATA'] == '/data'
    assert command[command.index('--deadline') + 1] == state['deadline']


def test_source_gate_rejects_unpaired_validation_cohorts():
    anchor, candidate = curve(), curve(61)
    anchor['_cohort_fingerprint'] = 'source_patient_pool_a'
    candidate['_cohort_fingerprint'] = 'source_patient_pool_b'
    with pytest.raises(ValueError, match='cohorts differ'):
        campaign.gate([candidate] * 3, [anchor] * 3)


def test_launch_uses_persistent_services_without_training_in_test(tmp_path, monkeypatch):
    state = spec(tmp_path)
    campaign.set_phase(state, 'screen', campaign.cv_jobs(['anchor'], [0]))
    calls = []
    monkeypatch.setattr(campaign, 'run_command', lambda *a, **kw: calls.append((a, kw)))
    monkeypatch.setattr(campaign.subprocess, 'run', lambda *a, **kw: None)
    campaign.launch(state, tmp_path / 'logs' / campaign.NAME / 'programme.json')
    services = [(args, kwargs) for args, kwargs in calls if 'input' in kwargs]
    assert services
    for args, kwargs in services:
        content = kwargs['input']
        assert 'Restart=on-failure' in content and 'WantedBy=default.target' in content
        assert ' worker ' in content and '--phase screen' in content
        script = args[2][-1]
        assert 'systemctl --user enable --now' in script
        assert 'systemd-run' not in script


def test_compatible_stack_preserves_every_effective_recipe_option(tmp_path):
    import itertools
    import shlex
    from src.training.config import parse_args
    state = spec(tmp_path)
    for size in (2, 3):
        for members in itertools.combinations(campaign.ARMS, size):
            if not campaign.compatible(members):
                continue
            state['stack'] = list(members)
            job = campaign.cv_jobs(['stack'], [0])[0]
            words = shlex.split(campaign.command(state, job))
            args = parse_args(words[words.index('--config'):])
            assert args.layer_decay == (0.8 if 'llrd' in members else 1.0)
            assert args.patient_cap == ('patient_cap' in members)
            assert args.class_pool == ('topk' if 'class_topk' in members else 'attention' if 'class_attention' in members else 'mean')
            objective = next((a for a in members if a in ('class_aware', 'recording_head', 'band_noise', 'band_noise_consistency')), None)
            expected = {'class_aware': 'ce_pafa_class_aware', 'recording_head': 'ce_recording_pafa',
                        'band_noise': 'ce_device_pafa', 'band_noise_consistency': 'ce_device_pafa', None: 'ce_pafa'}[objective]
            assert args.method == expected
            if objective in ('band_noise', 'band_noise_consistency'):
                assert args.num_workers == 0
                assert args.consistency_weight == (0.5 if objective == 'band_noise_consistency' else 0.0)


def test_incompatible_objectives_and_recording_patient_cap_are_never_stacked():
    for left, right in [('class_aware', 'recording_head'), ('class_aware', 'band_noise'),
                        ('recording_head', 'band_noise'), ('recording_head', 'patient_cap'),
                        ('class_topk', 'band_noise'), ('class_attention', 'band_noise_consistency')]:
        assert not campaign.compatible([left, right])


def test_final_fit_stops_at_source_epoch_without_shortening_cosine_horizon(tmp_path):
    import shlex
    from src.training.config import parse_args
    state = spec(tmp_path)
    job = campaign.final_jobs({'llrd': 17})[0]
    words = shlex.split(campaign.command(state, job))
    args = parse_args(words[words.index('--config'):])
    assert args.epochs == 100
    assert args.stop_epoch == args.report_epoch_list[0] == 17


def test_recording_report_keeps_learned_head_separate_from_cycle_controls(tmp_path):
    path = tmp_path / 'recording.npz'
    np.savez(path, stem=['r0', 'r0', 'r1'], patient=['a', 'a', 'b'], label=[0, 0, 1],
             probs=[[0.8, 0.1, 0.05, 0.05], [0.6, 0.2, 0.1, 0.1], [0.1, 0.4, 0.3, 0.2]],
             recording_stem=['r1', 'r0'], recording_patient=['b', 'a'], recording_label=[1, 0],
             recording_probs=[[0.3, 0.7], [0.9, 0.1]])
    endpoints = campaign.recording_endpoints(path)
    assert set(endpoints) == {'cycle_mean', 'cycle_max', 'learned_head'}
    assert endpoints['cycle_mean']['test_p'][:, 1] == pytest.approx([0.3, 0.9])
    assert endpoints['cycle_max']['test_p'][:, 1] == pytest.approx([0.4, 0.9])
    assert endpoints['learned_head']['test_p'][:, 1] == pytest.approx([0.1, 0.7])
    assert endpoints['learned_head']['stem'].tolist() == ['r0', 'r1']
    np.testing.assert_array_equal(endpoints['learned_head']['test_y'], endpoints['cycle_mean']['test_y'])


def test_source_curve_accepts_real_history_json_schema():
    from src.training.main import history_json
    raw = []
    for epoch in range(1, 101):
        raw.append({'epoch': epoch, 'train_acc': 1, 'train_loss': 2,
                    'val': {'all': {'sp': 50, 'se': 70, 'score': 60 + int(epoch == 17), 'hs': 58}},
                    'test': None, 'recording_val': {'all': {'score': 70 + int(epoch == 19)}}})
    serialized = history_json(raw)
    assert serialized[0]['val']['score'] == 60
    report = {'history': serialized}
    assert campaign.gate([report] * 3, [report] * 3)['epoch'] == 17
    assert campaign.gate([report] * 3, [report] * 3, recording=True)['epoch'] == 19


def test_recording_survivor_has_independent_confirmation_slot(tmp_path, monkeypatch):
    state = spec(tmp_path)
    state['phase'] = 'screen'
    monkeypatch.setattr(campaign, 'source_reports', lambda state, arm, seeds:
                        [curve(60 if arm == 'anchor' else (58 if arm == 'recording_head' else 62),
                               recording=60 if arm == 'anchor' else 61)] * (3 * len(seeds)))
    campaign.transition(state)
    assert 'recording_head' in state['shortlist']
    assert len(state['shortlist']) == 3
    assert not state['screen_results']['recording_head']['cycle_eligible']
    campaign.transition(state)
    assert 'recording_head' not in state['stack']
    if state['phase'] == 'stack':
        campaign.transition(state)
    assert 'recording_head' in state['selected']
    assert len(state['selected']) == 2


def test_negative_result_writes_auditable_completion_without_target_fits(tmp_path):
    state = spec(tmp_path)
    state['screen_results'] = {'llrd': campaign.gate([curve(59)] * 3, [curve(60)] * 3)}
    state['source_hashes'] = {'immutable.py': 'abc'}
    campaign.negative_report(state)
    destination = tmp_path / 'outputs/evaluation' / campaign.NAME
    assert campaign.load(destination / 'complete.json')['status'] == 'complete_no_survivors'
    assert 'llrd' in (destination / 'report.md').read_text()
    assert campaign.load(destination / 'summary.json')['source_audit']['source_hashes'] == {'immutable.py': 'abc'}


def test_new_phase_skips_low_disk_workers(tmp_path):
    state = spec(tmp_path)
    state['machines']['F']['admit'] = False
    campaign.set_phase(state, 'final', campaign.final_jobs({'band_noise': 12}))
    assert 'F' not in {job['machine'] for job in state['jobs']}


def test_true_head_triage_calibrates_on_source_patients_and_transfers_same_threshold(tmp_path, monkeypatch):
    from src.processing import splits
    from src.evaluation import screening
    source = tmp_path / 'source.npz'
    target = tmp_path / 'target.npz'
    np.savez(source, stem=['r0', 'r1'], patient=['a', 'b'], label=[0, 1], device=[0, 0],
             probs=[[0.9, 0.1, 0, 0], [0.2, 0.8, 0, 0]], recording_stem=['r0', 'r1'],
             recording_patient=['a', 'b'], recording_label=[0, 1], recording_device=[0, 0],
             recording_probs=[[0.8, 0.2], [0.4, 0.6]])
    np.savez(target, stem=['k0'], patient=['external'], label=[1], filter=['Bell'],
             probs=[[0.3, 0.7, 0, 0]], recording_stem=['k0'], recording_patient=['external'],
             recording_label=[1], recording_filter=['Bell'], recording_probs=[[0.1, 0.9]])
    def partition(table, **kwargs):
        assert set(table.patient) == {'a', 'b'}
        return [{'calibration': ['a'], 'evaluation': ['b']}]
    calls = []
    def triage(threshold, probabilities, labels, prior):
        calls.append((threshold, probabilities.copy()))
        return {'coverage': 1.0, 'size': 1.0}
    monkeypatch.setattr(splits, 'calibration_eval_splits', partition)
    monkeypatch.setattr(screening, 'triage', triage)
    rows = campaign.recording_triage(source, target)
    assert {row['endpoint'] for row in rows} == {'cycle_mean', 'cycle_max', 'learned_head'}
    assert len(rows) == 6
    assert all(calls[i][0] == calls[i + 1][0] for i in range(0, len(calls), 2))
    assert calls[-1][1][:, 1] == pytest.approx([0.9])


def test_source_completion_requires_oof_and_hash_verified_best_checkpoint(tmp_path):
    from src.artifacts import atomic_json, sha256_file
    state = spec(tmp_path)
    job = campaign.cv_jobs(['anchor'], [0])[0]
    job['machine'] = 'A'
    out = campaign.unit_path(state, job)
    checkpoint = campaign.unit_path(state, job, 'checkpoints') / 'best.pth'
    out.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    atomic_json(out / 'report.json', curve())
    np.savez(out / 'val_preds.npz', epochs=np.arange(1, 101), labels=[0], probs=np.zeros((100, 1, 4)))
    checkpoint.write_bytes(b'certified source checkpoint')
    snapshot = checkpoint.with_name('snapshot_epoch_66.pth')
    snapshot.write_bytes(b'fixed teacher snapshot')
    import os
    atomic_json(out / 'completion.json', {'schema': 1, 'files': {
        'report.json': sha256_file(out / 'report.json'), 'val_preds.npz': sha256_file(out / 'val_preds.npz'),
        os.path.relpath(checkpoint, out): sha256_file(checkpoint),
        os.path.relpath(snapshot, out): sha256_file(snapshot)}})
    assert campaign.verify_job(state, job)
    assert campaign.load(out / 'collected-checkpoint.json')['sha256'] == sha256_file(checkpoint)
    checkpoint.write_bytes(b'tampered source checkpoint')
    with pytest.raises(ValueError):
        campaign.verify_job(state, job)


def test_deadline_never_launches_new_training(tmp_path, monkeypatch):
    from src.artifacts import atomic_json
    state = spec(tmp_path)
    state['machines'] = {'A': {'local': True}}
    state.update(source_hashes={}, deadline='2000-01-01T00:00:00+00:00')
    campaign.set_phase(state, 'screen', campaign.cv_jobs(['anchor'], [0]))
    path = tmp_path / 'logs' / campaign.NAME / 'programme.json'
    atomic_json(path, state)
    launches = []
    monkeypatch.setattr(campaign, 'launch', lambda *args: launches.append(args))
    campaign.run(path, once=True)
    assert not launches
    assert campaign.load(path)['status']['admission_closed'] is True


def test_source_preprocessing_audit_is_persistent_cpu_only_and_idempotent(tmp_path, monkeypatch):
    state = spec(tmp_path)
    state.update(data='/source-data')
    job = campaign.cv_jobs(['anchor'], [0])[0]
    from src.artifacts import atomic_json
    atomic_json(campaign.unit_path(state, job) / 'collected-checkpoint.json', {'sha256': 'certified'})
    monkeypatch.setattr(campaign.Path, 'home', lambda: tmp_path / 'home')
    calls = []
    monkeypatch.setattr(campaign.subprocess, 'run', lambda command, **kw: calls.append(command))
    campaign.launch_audit(state, job)
    campaign.launch_audit(state, job)
    assert len(calls) == 2
    service = next((tmp_path / 'home/.config/systemd/user').glob('*.service')).read_text()
    assert 'CUDA_VISIBLE_DEVICES=' in service and 'OMP_NUM_THREADS=2' in service
    assert 'PYTHONPATH=' + str(tmp_path) in service
    assert '--max-recordings 16' in service and '--unit-dir' in service
    assert 'Restart=on-failure' in service
    assert campaign.audit_results(state)['cv0']['status'] == 'pending'


def test_final_completion_waits_for_all_three_verified_source_audits(tmp_path, monkeypatch):
    from src.artifacts import atomic_json
    state = spec(tmp_path)
    state.update(source_hashes={}, phase='complete_no_survivors', audits={}, screen_results={})
    manifest = tmp_path / 'logs' / campaign.NAME / 'programme.json'
    for fold in range(3):
        path = tmp_path / f'audit{fold}.json'
        state['audits'][f'cv{fold}'] = {'summary': str(path), 'checkpoint_sha256': f'checkpoint{fold}'}
        if fold < 2:
            atomic_json(path, {'frozen_source_comparison': {'status': 'complete', 'checkpoint_sha256': f'checkpoint{fold}'}})
    atomic_json(manifest, state)
    campaign.run(manifest, once=True)
    assert not (tmp_path / 'outputs/evaluation' / campaign.NAME / 'complete.json').exists()
    assert not campaign.audits_complete(state)
    atomic_json(tmp_path / 'audit2.json', {'frozen_source_comparison': {'status': 'complete', 'checkpoint_sha256': 'checkpoint2'}})
    assert campaign.audits_complete(state)
    campaign.run(manifest, once=True)
    assert (tmp_path / 'outputs/evaluation' / campaign.NAME / 'complete.json').exists()
    atomic_json(tmp_path / 'audit2.json', {'frozen_source_comparison': {'status': 'complete', 'checkpoint_sha256': 'wrong'}})
    with pytest.raises(ValueError, match='checkpoint differs'):
        campaign.audits_complete(state)


def _collect_fixture(tmp_path):
    state = {'snapshot': str(tmp_path), 'machines': {'A': {'local': True}, 'B': {'host': 'b'}, 'D': {'host': 'd'}}}
    jobs = campaign.cv_jobs(['active', 'done'], [0])[:2]
    for job in jobs:
        job['machine'] = 'D'
    state['jobs'] = jobs
    payload = b'certified final weights'
    expected = hashlib.sha256(payload).hexdigest()
    active = campaign.unit_path(state, jobs[0], 'checkpoints') / 'best.pth'
    active.parent.mkdir(parents=True)
    active.write_bytes(b'active mutable weights')
    completed = campaign.unit_path(state, jobs[1])
    checkpoint = campaign.unit_path(state, jobs[1], 'checkpoints') / 'best.pth'
    completed.mkdir(parents=True)
    atomic_json(completed / 'report.json', {'source_only': True})
    atomic_json(completed / 'completion.json', {'schema': 1, 'files': {
        'report.json': sha256_file(completed / 'report.json'), os.path.relpath(checkpoint, completed): expected}})
    return state, jobs, active, checkpoint, payload, expected


def install_transport(monkeypatch, checkpoint, payload, expected, vanished=False):
    calls = []
    def command(spec, machine, argv, **kw):
        calls.append(argv)
        return SimpleNamespace(stdout=expected + '  archived.pth\n', returncode=0)
    def transfer(argv, **kw):
        calls.append(argv)
        if argv[0] == 'rsync' and argv[-1] == str(checkpoint):
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(payload)
        return SimpleNamespace(returncode=24 if vanished and '--include=*.npz' in argv else 0, args=argv)
    monkeypatch.setattr(campaign, 'run_command', command)
    monkeypatch.setattr(campaign.subprocess, 'run', transfer)
    return calls


def test_active_checkpoints_are_never_fetched_or_reverse_archived(tmp_path, monkeypatch):
    state, jobs, active, checkpoint, payload, expected = _collect_fixture(tmp_path)
    calls = install_transport(monkeypatch, checkpoint, payload, expected)
    campaign.collect(state)
    transfers = [a for a in calls if a[0] == 'rsync']
    fetched = [a for a in transfers if a[-1] == str(checkpoint)]
    assert len(fetched) == 1
    assert fetched[0][-2] == 'd:' + str(checkpoint)
    assert not any('d:' + str(active) in a for a in transfers)
    archives = [a for a in transfers if a[-2] == str(checkpoint)]
    assert len(archives) == 1
    assert archives[0][-1] == 'b:' + str(tmp_path / 'archive/checkpoints/train' / jobs[1]['run'] / 'done/cv0/best.pth')
    assert active.read_bytes() == b'active mutable weights'
    calls.clear()
    campaign.collect(state)
    assert not any(a[0] == 'rsync' and '--checksum' in a for a in calls)


def test_invalid_report_hash_prevents_any_checkpoint_fetch(tmp_path, monkeypatch):
    state, jobs, active, checkpoint, payload, expected = _collect_fixture(tmp_path)
    calls = install_transport(monkeypatch, checkpoint, payload, expected)
    (campaign.unit_path(state, jobs[1]) / 'report.json').write_text('tampered')
    with pytest.raises(ValueError, match='integrity'):
        campaign.collect(state)
    assert not any(a[0] == 'rsync' and '--checksum' in a for a in calls)


def test_checkpoint_content_is_verified_after_completed_transfer(tmp_path, monkeypatch):
    state, jobs, active, checkpoint, payload, expected = _collect_fixture(tmp_path)
    calls = install_transport(monkeypatch, checkpoint, b'wrong bytes', expected)
    with pytest.raises(ValueError, match='integrity'):
        campaign.collect(state)
    assert not any(a[0] == 'rsync' and a[-2] == str(checkpoint) for a in calls)


def test_vanished_active_output_metadata_does_not_restart_controller(tmp_path, monkeypatch):
    state, jobs, active, checkpoint, payload, expected = _collect_fixture(tmp_path)
    install_transport(monkeypatch, checkpoint, payload, expected, vanished=True)
    campaign.collect(state)
    assert checkpoint.read_bytes() == payload


def test_phase_allocations_account_for_host_throughput_and_workload_cost(tmp_path):
    from collections import Counter
    state = spec(tmp_path)
    jobs = campaign.cv_jobs(['anchor', *campaign.ARMS], [0])
    campaign.set_phase(state, 'screen', jobs)
    counts = Counter(job['machine'] for job in jobs)
    assert set(counts) == {'A', 'B', 'D', 'F'}
    assert counts['A'] < counts['B'] and counts['A'] < counts['D']
    identities = [(job['run'], job['cell'], job['unit']) for job in jobs]
    assert len(identities) == len(set(identities)) == 27
    lines = [line for machine in counts for line in
             (tmp_path / 'logs' / campaign.NAME / 'screen' / f'{machine}.jobs.txt').read_text().splitlines()]
    assert len(lines) == len(set(lines)) == 27
    # With equal throughput, a recording job takes half the projected slot and prefers F on ties.
    equal = spec(tmp_path / 'equal')
    equal['machines']['A']['throughput_weight'] = 1
    first = campaign.cv_jobs(['recording_head'], [0])[:1]
    campaign.set_phase(equal, 'screen', first)
    assert first[0]['machine'] == 'F'


def test_phase_throughput_override_changes_only_future_assignments(tmp_path):
    from collections import Counter
    state = spec(tmp_path)
    before = campaign.cv_jobs(['anchor', *campaign.ARMS], [0])
    campaign.set_phase(state, 'screen', before)
    recorded = copy.deepcopy(before)
    state['machines']['A']['throughput_weight'] = 2
    after = campaign.cv_jobs(['anchor', *campaign.ARMS], [1])
    campaign.set_phase(state, 'confirm', after)
    assert before == recorded
    assert Counter(job['machine'] for job in after)['A'] > Counter(job['machine'] for job in before)['A']
    assert state['phases']['screen']['jobs'] == recorded


def test_phase_rejects_invalid_throughput_weights(tmp_path):
    state = spec(tmp_path)
    state['machines']['A']['throughput_weight'] = 0
    with pytest.raises(ValueError, match='throughput weights'):
        campaign.set_phase(state, 'screen', campaign.cv_jobs(['anchor'], [0]))


def test_optional_recording_refinement_uses_original_overlay_and_only_changes_pcsl(tmp_path):
    import shlex
    from src.training.config import parse_args
    state = spec(tmp_path)
    for arm, stack in [('recording_head_pcsl0', []), ('stack', ['llrd', 'recording_head_pcsl0'])]:
        state['stack'] = stack
        for job in [campaign.cv_jobs([arm], [0])[0], campaign.final_jobs({arm: 17})[0]]:
            words = shlex.split(campaign.command(state, job))
            args = parse_args(words[words.index('--config'):])
            assert 'experiments/train/recording_head.yaml' in args.config_files
            assert not any('recording_head_pcsl0.yaml' in filename for filename in args.config_files)
            assert args.method == 'ce_recording_pafa'
            assert args.lambda_pcsl == 0
            assert args.lambda_gpal == 0.0005
            assert args.recording_weight == 0.5
            assert args.recording_bag_cycles == 4
            assert args.epochs == 100
            assert args.layer_decay == (0.8 if arm == 'stack' else 1.0)
    original = campaign.cv_jobs(['recording_head'], [0])[0]
    words = shlex.split(campaign.command(state, original))
    assert parse_args(words[words.index('--config'):]).lambda_pcsl == 50


def test_optional_arms_rank_best_recording_variant_independently_and_keep_original_findings(tmp_path, monkeypatch):
    state = spec(tmp_path)
    state.update(phase='screen', arms=[*campaign.ARMS, 'recording_head_pcsl0'])
    def reports(spec, arm, seeds):
        cycle, recording = {'anchor': (60, 70), 'recording_head': (59, 71),
                            'recording_head_pcsl0': (59.6, 72), 'llrd': (61, 60),
                            'stack': (60.5, 72.5)}.get(arm, (59, 69))
        return [curve(cycle, recording=recording)] * (3 * len(seeds))
    monkeypatch.setattr(campaign, 'source_reports', reports)
    campaign.transition(state)
    assert set(state['screen_results']) == set(state['arms'])
    assert state['screen_results']['recording_head']['mean_delta'] == 1
    assert state['screen_results']['recording_head_pcsl0']['mean_delta'] == 2
    assert state['shortlist'] == ['llrd', 'recording_head_pcsl0']
    assert len(state['jobs']) == 9
    campaign.transition(state)
    assert set(state['stack']) == {'llrd', 'recording_head_pcsl0'}
    campaign.transition(state)
    assert set(state['selected']) == {'llrd', 'stack'}
    assert len(state['jobs']) == 50


def test_refined_recording_arm_obeys_recording_compatibility_and_cost(tmp_path):
    for other in ('recording_head', 'patient_cap', 'class_topk', 'class_attention',
                  'class_aware', 'band_noise', 'band_noise_consistency'):
        assert not campaign.compatible(['recording_head_pcsl0', other])
    assert campaign.compatible(['recording_head_pcsl0', 'llrd'])
    state = spec(tmp_path)
    jobs = campaign.cv_jobs(['recording_head_pcsl0'], [0])[:1]
    campaign.set_phase(state, 'screen', jobs)
    assert jobs[0]['machine'] == 'F'
    assert len(campaign.cv_jobs(['anchor', *campaign.ARMS], [0])) == 27


def test_recording_cycle_guard_uses_anchor_cycle_best_not_binary_selected_epoch():
    anchor = curve(61, epoch=80, recording=69)
    for row in anchor['history']:
        row['val']['all']['score'] = 61 if row['epoch'] == 80 else 48
        row['recording_val']['all']['score'] = 70 if row['epoch'] == 5 else 65
    candidate = curve(59, epoch=10, recording=72)
    result = campaign.gate([candidate] * 3, [anchor] * 3, recording=True)
    assert result['epoch'] == 10
    assert result['anchor_epoch'] == 5
    assert result['anchor_cycle_epoch'] == 80
    assert result['mean_delta'] == 2
    assert result['cycle_delta'] == -2
    assert result['pass']
    assert not result['cycle_eligible']


def test_source_reports_record_operational_hash_separately_from_frozen_source(tmp_path):
    state = spec(tmp_path)
    amendments = [{'arm': 'recording_head_pcsl0', 'evidence': 'source-only recording loss instability'}]
    frozen = {'scripts/improvement_campaign.py': 'original_frozen_controller_sha'}
    state.update(scope_amendments=amendments, source_hashes=frozen, screen_results={})
    audit = campaign.source_audit(state)
    assert audit['operational_controller_sha256'] == sha256_file(campaign.__file__)
    assert audit['scope_amendments'] == amendments
    assert audit['source_hashes'] == frozen
    campaign.negative_report(state)
    saved = campaign.load(tmp_path / 'outputs/evaluation' / campaign.NAME / 'summary.json')['source_audit']
    assert saved == audit


def _replica_request(tmp_path):
    root = tmp_path.resolve()
    checkpoint = root / 'checkpoints/train' / (campaign.NAME + '-s0') / 'recording_head/cv0/best.pth'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'finalized checkpoint')
    expected = sha256_file(checkpoint)
    out = root / 'outputs/train' / (campaign.NAME + '-s0') / 'recording_head/cv0'
    out.mkdir(parents=True)
    marker = out / 'completion.json'
    atomic_json(marker, {'schema': 1, 'files': {os.path.relpath(checkpoint, out): expected}})
    archive = root / 'archive/checkpoints/train' / (campaign.NAME + '-s0') / 'recording_head/cv0/best.pth'
    archive.parent.mkdir(parents=True)
    archive.write_bytes(checkpoint.read_bytes())
    request = dict(snapshot=str(root), machine='F', checkpoint=str(checkpoint), checkpoint_sha256=expected,
                   a_sha256=expected, b_sha256=expected, archive_path=str(archive), completion_sha256=sha256_file(marker))
    return request, checkpoint, out, marker, archive


def test_reclamation_writes_pointer_before_unlink_and_keeps_canonical_archive(tmp_path, monkeypatch):
    request, checkpoint, out, marker, archive = _replica_request(tmp_path)
    canonical_a = tmp_path / 'canonical-a.pth'
    canonical_a.write_bytes(checkpoint.read_bytes())
    original = Path.unlink
    def unlink(path, **kwargs):
        assert (out / 'archived-checkpoint.json').is_file()
        assert campaign.load(out / 'archived-checkpoint.json')['checkpoint_sha256'] == request['checkpoint_sha256']
        return original(path, **kwargs)
    monkeypatch.setattr(Path, 'unlink', unlink)
    result = campaign._unlink_f_replica(request)
    assert result['status'] == 'archived'
    assert not checkpoint.exists()
    assert canonical_a.exists() and archive.exists() and marker.exists()


@pytest.mark.parametrize('violation', ['unfinalized', 'no_b_hash', 'wrong_hash', 'active_last', 'active_tmp', 'last_checkpoint', 'old_run', 'wrong_machine'])
def test_reclamation_refuses_unfinalized_unverified_active_or_out_of_scope_files(tmp_path, violation):
    request, checkpoint, out, marker, archive = _replica_request(tmp_path)
    if violation == 'unfinalized':
        marker.unlink()
    elif violation == 'no_b_hash':
        request.pop('b_sha256')
    elif violation == 'wrong_hash':
        checkpoint.write_bytes(b'changed checkpoint')
    elif violation in ('active_last', 'active_tmp'):
        (checkpoint.parent / ('last.pth' if violation == 'active_last' else 'last.pth.tmp')).write_bytes(b'resume state')
    elif violation == 'last_checkpoint':
        request['checkpoint'] = str(checkpoint.parent / 'last.pth')
    elif violation == 'old_run':
        request['checkpoint'] = str(tmp_path / 'checkpoints/train/historical/recording_head/cv0/best.pth')
    else:
        request['machine'] = 'D'
    if violation in ('unfinalized', 'active_last', 'active_tmp'):
        assert campaign._unlink_f_replica(request)['status'] in ('unfinalized', 'active')
    else:
        with pytest.raises(ValueError):
            campaign._unlink_f_replica(request)
    assert checkpoint.exists() and archive.exists()
    assert not (out / 'archived-checkpoint.json').exists()


def test_low_disk_reclamation_requires_fresh_b_hash_and_never_calls_unlink_on_mismatch(tmp_path, monkeypatch):
    request, checkpoint, out, marker, archive = _replica_request(tmp_path)
    state = spec(tmp_path)
    job = campaign.cv_jobs(['recording_head'], [0])[0]
    job['machine'] = 'F'
    state['jobs'] = [job]
    atomic_json(out / 'collected-checkpoint.json', {'sha256': request['checkpoint_sha256'], 'archive_path': str(archive)})
    calls = []
    def command(spec, machine, argv, **kwargs):
        calls.append((machine, argv))
        return SimpleNamespace(stdout='Available\n30000000000\n' if argv[0] == 'df' else 'wrong_hash archive\n')
    monkeypatch.setattr(campaign, 'run_command', command)
    with pytest.raises(ValueError, match='verified B archive'):
        campaign.reclaim_f_replicas(state)
    assert checkpoint.exists()
    assert not any('-c' in argv for _, argv in calls)


@pytest.mark.parametrize('local_state', ['missing', 'corrupt'])
def test_collector_restores_archived_f_checkpoint_from_b(tmp_path, monkeypatch, local_state):
    state, jobs, active, checkpoint, payload, expected = _collect_fixture(tmp_path)
    jobs[1]['machine'] = 'F'
    state['machines']['F'] = {'host': 'f'}
    archive = tmp_path / 'archive/checkpoints/train' / jobs[1]['run'] / 'done/cv0/best.pth'
    out = campaign.unit_path(state, jobs[1])
    atomic_json(out / 'archived-checkpoint.json', {'checkpoint_sha256': expected, 'archive_path': str(archive)})
    atomic_json(out / 'collected-checkpoint.json', {'sha256': expected, 'archive_path': str(archive)})
    if local_state == 'corrupt':
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b'corrupt local replica')
    calls = install_transport(monkeypatch, checkpoint, payload, expected)
    monkeypatch.setattr(campaign, 'reclaim_f_replicas', lambda spec: None)
    campaign.collect(state)
    fetches = [argv for argv in calls if argv[0] == 'rsync' and argv[-1] == str(checkpoint)]
    assert len(fetches) == 1 and fetches[0][-2] == 'b:' + str(archive)
    assert checkpoint.read_bytes() == payload


def test_low_disk_reclamation_verifies_a_and_b_then_uses_f_only_remote_guard(tmp_path, monkeypatch):
    request, checkpoint, out, marker, archive = _replica_request(tmp_path)
    state = spec(tmp_path)
    job = campaign.cv_jobs(['recording_head'], [0])[0]
    job['machine'] = 'F'
    state['jobs'] = [job]
    atomic_json(out / 'collected-checkpoint.json', {'sha256': request['checkpoint_sha256'], 'archive_path': str(archive)})
    calls, free = [], [30, 36]
    def command(spec, machine, argv, **kwargs):
        calls.append((machine, argv))
        if argv[0] == 'df':
            return SimpleNamespace(stdout='Available\n' + str(free.pop(0) * 1024 ** 3))
        if argv[0] == 'sha256sum':
            assert machine == 'B' and sha256_file(checkpoint) == request['checkpoint_sha256']
            return SimpleNamespace(stdout=request['checkpoint_sha256'] + ' archive')
        assert machine == 'F' and '-c' in argv
        import json
        remote = json.loads(argv[-1])
        assert remote['completion_sha256'] == sha256_file(marker)
        assert remote['a_sha256'] == remote['b_sha256'] == request['checkpoint_sha256']
        assert 'checkpoint.unlink()' in argv[-2]
        return SimpleNamespace(stdout=json.dumps({**remote, 'status': 'archived', 'bytes': len(b'finalized checkpoint')}))
    monkeypatch.setattr(campaign, 'run_command', command)
    campaign.reclaim_f_replicas(state)
    assert campaign.load(out / 'collected-checkpoint.json')['f_reclaimed'] is True
    assert campaign.load(out / 'archived-checkpoint.json')['checkpoint_sha256'] == request['checkpoint_sha256']
    assert checkpoint.exists()  # Mocked F action never touches the canonical A file.
    assert [machine for machine, argv in calls if argv[0] == 'sha256sum'] == ['B']
    assert len(state['storage_reclamation']) == 1


def test_future_queue_reservations_match_measured_normal_and_consistency_peaks(tmp_path):
    import shlex
    from src.training.config import parse_args
    from scripts.sweep import mem_gb
    state = spec(tmp_path)
    for arm in ['anchor', *campaign.ARMS, 'recording_head_pcsl0']:
        job = campaign.cv_jobs([arm], [0])[0]
        command = campaign.command(state, job)
        assert mem_gb(command, 0) == (16 if arm == 'band_noise_consistency' else 11)
        words = shlex.split(command)
        args = parse_args(words[words.index('--config'):])
        assert args.epochs == 100 and args.seed == 0 and args.cv_folds == 3
    state['stack'] = ['llrd', 'band_noise_consistency']
    assert mem_gb(campaign.command(state, campaign.cv_jobs(['stack'], [0])[0]), 0) == 16
    state['stack'] = ['llrd', 'recording_head_pcsl0']
    assert mem_gb(campaign.command(state, campaign.final_jobs({'stack': 17})[0]), 0) == 11


def test_future_phase_budgets_bound_orphans_and_allow_measured_mixed_pairs(tmp_path):
    state = spec(tmp_path)
    campaign.set_phase(state, 'confirm', campaign.cv_jobs(['anchor'], [1]))
    budgets = {machine: config['gpu_gb'] for machine, config in state['machines'].items()}
    assert budgets == {'A': 28, 'B': 28, 'D': 28, 'F': 32}
    assert all(3 * 11 > budget for budget in budgets.values())
    assert all(16 + 11 <= budget for budget in budgets.values())
    assert 2 * 16 <= budgets['F']
    assert all(2 * 16 > budgets[machine] for machine in ('A', 'B', 'D'))


def test_followup_names_do_not_mutate_legacy_identity_or_require_a(tmp_path, monkeypatch):
    state = spec(tmp_path)
    state['name'] = 'followup-20261026'
    state['machines'] = {'B': {'local': True}, 'D': {'host': 'd'}, 'E': {'host': 'e'}, 'F': {'host': 'f'}}
    state['data'] = 'data'
    jobs = campaign.cv_jobs(['anchor', 'llrd'], [0], name=state['name'])
    campaign.set_phase(state, 'screen', jobs)
    assert all(job['run'] == 'followup-20261026-s0' for job in jobs)
    assert (tmp_path / 'logs/followup-20261026/screen/E.jobs.txt').is_file()
    assert state['machines']['E']['gpu_gb'] == 28
    assert campaign.cv_jobs(['anchor'], [0])[0]['run'] == campaign.NAME + '-s0'
    assert campaign.final_jobs({'llrd': 17}, name=state['name'])[0]['run'] == 'followup-20261026-final'
    calls = []
    monkeypatch.setattr(campaign, 'verify_source', lambda _: None)
    monkeypatch.setattr(campaign.subprocess, 'run', lambda argv, **kw: calls.append((argv, kw)))
    campaign.worker(state, 'E', 'screen')
    argv, options = calls[-1]
    assert str(tmp_path / 'logs/followup-20261026/screen/E.jobs.txt') in argv
    assert options['env']['RUN_ID'] == state['name']
    assert argv[argv.index('--gpu-gb') + 1] == '28'


def test_followup_launcher_service_and_report_paths_are_namespaced(tmp_path, monkeypatch):
    state = spec(tmp_path)
    state.update(name='followup-20261026', phase='screen', jobs=[{'machine': 'A'}], screen_results={})
    calls = []
    monkeypatch.setattr(campaign, 'run_command', lambda *args, **kw: calls.append((args, kw)))
    campaign.launch(state, tmp_path / 'logs/followup-20261026/programme.json')
    assert 'auscult-followup-20261026-screen-A' in calls[-1][0][2][-1]
    campaign.negative_report(state)
    assert (tmp_path / 'outputs/evaluation/followup-20261026/complete.json').is_file()
    assert not (tmp_path / 'outputs/evaluation' / campaign.NAME).exists()


def test_followup_collector_archives_to_e_and_records_archive_machine(tmp_path, monkeypatch):
    state, jobs, active, checkpoint, payload, expected = _collect_fixture(tmp_path)
    state.update(name='followup-20261026', archive_machine='E')
    state['machines']['E'] = {'host': 'e'}
    calls = install_transport(monkeypatch, checkpoint, payload, expected)
    campaign.collect(state)
    archives = [argv for argv in calls if argv[0] == 'rsync' and argv[-2] == str(checkpoint)]
    assert len(archives) == 1 and archives[0][-1].startswith('e:')
    assert campaign.load(campaign.unit_path(state, jobs[1]) / 'collected-checkpoint.json')['archive_machine'] == 'E'


@pytest.mark.parametrize('local_state', ['missing', 'corrupt'])
def test_followup_collector_restores_from_selected_e_archive(tmp_path, monkeypatch, local_state):
    state, jobs, active, checkpoint, payload, expected = _collect_fixture(tmp_path)
    state.update(name='followup-20261026', archive_machine='E')
    state['machines'].update(E={'host': 'e'}, F={'host': 'f'})
    jobs[1]['machine'] = 'F'
    archive = tmp_path / 'archive/checkpoints/train' / jobs[1]['run'] / 'done/cv0/best.pth'
    out = campaign.unit_path(state, jobs[1])
    atomic_json(out / 'archived-checkpoint.json', {'checkpoint_sha256': expected, 'archive_path': str(archive), 'archive_machine': 'E'})
    atomic_json(out / 'collected-checkpoint.json', {'sha256': expected, 'archive_path': str(archive), 'archive_machine': 'E'})
    if local_state == 'corrupt':
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b'corrupt')
    calls = install_transport(monkeypatch, checkpoint, payload, expected)
    monkeypatch.setattr(campaign, 'reclaim_f_replicas', lambda _: None)
    campaign.collect(state)
    fetches = [argv for argv in calls if argv[0] == 'rsync' and argv[-1] == str(checkpoint)]
    assert len(fetches) == 1 and fetches[0][-2] == 'e:' + str(archive)
    assert checkpoint.read_bytes() == payload


@pytest.mark.parametrize('programme', ['followup-20261026', 'fivefold-20261026'])
def test_remote_guard_binds_explicit_followup_name_and_archive_metadata(tmp_path, programme):
    request, checkpoint, out, marker, archive = _replica_request(tmp_path)
    request.update(programme_name=programme, archive_machine='E')
    with pytest.raises(ValueError, match='Only finalized'):
        campaign._unlink_f_replica(request)
    replacement = tmp_path / 'checkpoints/train' / (programme + '-s0') / 'recording_head/cv0/best.pth'
    replacement.parent.mkdir(parents=True)
    replacement.write_bytes(checkpoint.read_bytes())
    followup_out = tmp_path / 'outputs/train' / (programme + '-s0') / 'recording_head/cv0'
    followup_out.mkdir(parents=True)
    followup_marker = followup_out / 'completion.json'
    atomic_json(followup_marker, {'files': {str(replacement): request['checkpoint_sha256']}})
    request.update(checkpoint=str(replacement), completion_sha256=sha256_file(followup_marker),
                   archive_path=str(tmp_path / 'archive/checkpoints/train' / (programme + '-s0') / 'recording_head/cv0/best.pth'))
    result = campaign._unlink_f_replica(request)
    assert result['status'] == 'archived' and result['archive_machine'] == 'E'
    assert result['programme_name'] == programme
    assert checkpoint.exists() and not replacement.exists()


def test_followup_low_disk_reclamation_requires_e_hash_and_names_f_request(tmp_path, monkeypatch):
    state = spec(tmp_path)
    state.update(name='followup-20261026', archive_machine='E')
    state['machines']['E'] = {'host': 'e'}
    job = campaign.cv_jobs(['recording_head'], [0], name=state['name'])[0]
    job['machine'] = 'F'
    state['jobs'] = [job]
    checkpoint = campaign.unit_path(state, job, 'checkpoints') / 'best.pth'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'certified followup weights')
    expected = sha256_file(checkpoint)
    out = campaign.unit_path(state, job)
    out.mkdir(parents=True)
    marker = out / 'completion.json'
    atomic_json(marker, {'files': {str(checkpoint): expected}})
    archive = tmp_path / 'archive/checkpoints/train' / job['run'] / job['cell'] / job['unit'] / checkpoint.name
    atomic_json(out / 'collected-checkpoint.json', {'sha256': expected, 'archive_path': str(archive), 'archive_machine': 'E'})
    calls, free = [], [30, 36]
    def command(spec, machine, argv, **kwargs):
        calls.append((machine, argv))
        if argv[0] == 'df':
            return SimpleNamespace(stdout='Available\n' + str(free.pop(0) * 1024 ** 3))
        if argv[0] == 'sha256sum':
            assert machine == 'E'
            return SimpleNamespace(stdout=expected + ' archive')
        assert machine == 'F' and '-c' in argv
        import json
        request = json.loads(argv[-1])
        assert request['programme_name'] == state['name'] and request['archive_machine'] == 'E'
        assert request['a_sha256'] == request['b_sha256'] == expected
        return SimpleNamespace(stdout=json.dumps({**request, 'status': 'archived', 'bytes': 1}))
    monkeypatch.setattr(campaign, 'run_command', command)
    campaign.reclaim_f_replicas(state)
    pointer = campaign.load(out / 'archived-checkpoint.json')
    assert pointer['archive_machine'] == 'E'
    assert [machine for machine, argv in calls if argv[0] == 'sha256sum'] == ['E']
    assert checkpoint.exists()


def test_receipt_entries_from_another_mount_are_rebased_to_this_snapshot(tmp_path):
    spec = {'snapshot': str(tmp_path / 'snap')}
    out = tmp_path / 'snap/outputs/train/run/cell/unit'
    foreign = '/storage/hdd/disk0/namdp15/auscult-trust-A/campaigns/run/outputs/train/run/cell/unit/export/model.sha256'
    assert campaign.canonical_path(spec, foreign, out) == (tmp_path / 'snap/outputs/train/run/cell/unit/export/model.sha256').resolve()
    assert campaign.canonical_path(spec, 'export/test.npz', out) == (out / 'export/test.npz').resolve()
    archived = '/storage/hdd/x/archive/checkpoints/train/run/cell/unit/best.pth'
    assert campaign.canonical_path(spec, archived, out) == Path(archived).resolve()
