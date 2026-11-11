"""Source-only improvement gates and phased queues for the existing sweep scheduler."""
import argparse
import fcntl
import itertools
import inspect
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.artifacts import atomic_json, sha256_file, verify_file
from src.training.reporting import cv_epoch

NAME = 'improvements-20261026'
ARMS = ['llrd', 'patient_cap', 'class_topk', 'class_attention', 'class_aware', 'recording_head', 'band_noise', 'band_noise_consistency']
RECORDING_ARMS = {'recording_head', 'recording_head_pcsl0'}
GPU_BUDGETS = {'A': 28, 'B': 28, 'C': 28, 'D': 28, 'E': 28, 'F': 32}
DEVICES = ['AKGC417L', 'Litt3200', 'LittC2SE', 'Meditron']
BASE = ['base', 'encoder_beats', 'grid/pafa_beats', 'sc', 'sc_static', 'sc_gain']


def programme_name(spec):
    return spec.get('name', NAME)


def run_controller(manifest, tick, interval=60):
    """Lock the manifest and call `tick(spec, manifest)` every `interval` s until its phase is complete."""
    manifest = Path(manifest).resolve()
    with (manifest.parent / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            spec = load(manifest)
            tick(spec, manifest)
            if spec['phase'] == 'complete':
                return
            time.sleep(interval)


def load(path):
    return json.loads(Path(path).read_text())


def run_command(spec, machine, command, **kwargs):
    m = spec['machines'][machine]
    argv = command if m.get('local') else ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', m['host'], shlex.join(command)]
    return subprocess.run(argv, check=True, **kwargs)


def compatible(arms):
    selected = set(arms)
    objectives = selected & (RECORDING_ARMS | {'class_aware', 'band_noise', 'band_noise_consistency'})
    return not (len(objectives) > 1 or {'class_topk', 'class_attention'} <= selected or
                (bool(selected & RECORDING_ARMS) and 'patient_cap' in selected) or
                (bool(selected & (RECORDING_ARMS | {'band_noise', 'band_noise_consistency'})) and
                 bool(selected & {'class_topk', 'class_attention'})))


def stack_members(winners):
    """Highest-ranked compatible subset, with the fewest departures on an equal score."""
    choices = [tuple(c) for n in range(2, len(winners) + 1) for c in itertools.combinations(winners, n) if compatible(c)]
    return list(max(choices, key=lambda c: (len(c), -sum(winners.index(a) for a in c)))) if choices else []


def source_curve(report, recording=False):
    key = 'recording_val' if recording else 'val'
    history = report['history']
    if len(history) != 100 or [h['epoch'] for h in history] != list(range(1, 101)):
        raise ValueError('Source screening requires the complete 100-epoch validation curve')
    def score(row):
        if not recording and row.get('val') is None and row.get('swa', {}).get('count', 0) < 2 and report.get('swa'):
            return -np.inf
        if not recording and row.get('val_score') is not None:
            return row['val_score']
        values = row[key]
        return values.get('all', values)['score']
    curve = np.asarray([score(h) for h in history], dtype=float)
    first = np.flatnonzero(np.isfinite(curve))
    if not len(first) or not np.isfinite(curve[first[0]:]).all() or not np.isneginf(curve[:first[0]]).all():
        raise ValueError('Nonfinite source validation curve')
    return curve


def gate(candidate, anchor, recording=False):
    """Earliest mean-curve epoch, unrounded paired fold deltas and the cycle guard."""
    if len(candidate) != len(anchor) or len(candidate) not in (3, 6):
        raise ValueError("Paired source gates require three folds and one or two seeds")
    for left, right in zip(candidate, anchor):
        if left.get("_cohort_fingerprint") != right.get("_cohort_fingerprint"):
            raise ValueError("Source validation cohorts differ between paired arms")
    curves = [source_curve(r, recording) for r in candidate]
    baseline = [source_curve(r, recording) for r in anchor]
    epoch, base_epoch = cv_epoch(curves), cv_epoch(baseline)
    deltas = np.asarray([a[epoch - 1] - b[base_epoch - 1] for a, b in zip(curves, baseline)])
    fold_deltas = deltas.reshape(-1, 3).mean(0)
    anchor_cycle_epoch = cv_epoch([source_curve(report) for report in anchor])
    cycle_delta = float(np.mean([source_curve(a)[epoch - 1] - source_curve(b)[anchor_cycle_epoch - 1]
                                for a, b in zip(candidate, anchor)]))
    return {'epoch': epoch, 'anchor_epoch': base_epoch, 'anchor_cycle_epoch': anchor_cycle_epoch, 'mean_delta': float(deltas.mean()),
            'fold_deltas': fold_deltas.tolist(), 'seed_deltas': deltas.reshape(-1, 3).mean(1).tolist(),
            'cycle_delta': cycle_delta, 'cycle_eligible': bool(not recording or cycle_delta >= -0.5),
            'pass': bool(deltas.mean() >= 0.5 and np.sum(fold_deltas > 0) >= 2 and
                         (len(candidate) == 3 or np.all(deltas.reshape(-1, 3).mean(1) > 0)))}


def source_reports(spec, arm, seeds):
    root = Path(spec['snapshot'])
    reports = []
    for seed in seeds:
        for fold in range(3):
            unit = root / 'outputs/train' / f'{programme_name(spec)}-s{seed}' / arm / f'cv{fold}'
            report = load(unit / 'report.json')
            with np.load(unit / 'val_preds.npz', allow_pickle=False) as arrays:
                cohort = {key: arrays[key].tolist() for key in ('labels', 'patient', 'stem', 'device')}
            report['_cohort_fingerprint'] = hashlib.sha256(json.dumps(cohort, sort_keys=True).encode()).hexdigest()
            reports.append(report)
    return reports


def cv_jobs(arms, seeds, *, name=NAME):
    return [dict(arm=arm, seed=seed, fold=fold, run=f'{name}-s{seed}', cell=arm, unit=f'cv{fold}')
            for seed in seeds for fold in range(3) for arm in arms]


def final_jobs(selected, *, name=NAME):
    return [dict(arm=arm, epoch=epoch, seed=seed, cohort=cohort, run=name + '-final',
                 cell=arm + ('' if cohort == 'official' else '+lodo_' + cohort.lower()), unit=f'seed{seed}')
            for arm, epoch in selected.items() for cohort in ['official', *DEVICES] for seed in range(5)]


def command(spec, job):
    root = Path(spec['snapshot'])
    overlays = spec.get('recipes', {}).get(job['arm'],
        spec['stack'] if job['arm'] == 'stack' else ([] if job['arm'] == 'anchor' else [job['arm']]))
    recipes = ['recording_head' if name == 'recording_head_pcsl0' else name for name in overlays]
    configs = ','.join('experiments/train/' + name + '.yaml' for name in BASE + recipes)
    memory_gb = 16 if set(overlays) & {'band_noise_consistency', 'multiscale'} else 11
    env = [f'RUN_ID={job["run"]}', f'MEM_GB={memory_gb}']
    argv = [str(root / '.venv-train/bin/python'), '-u']
    if 'cohort' in job:
        marker = root / 'outputs/train' / job['run'] / job['cell'] / job['unit'] / 'pipeline-complete.json'
        env.append('PIPELINE_DONE=' + str(marker))
        argv += ['scripts/improvement_unit.py']
    else:
        marker = root / 'outputs/train' / job['run'] / job['cell'] / job['unit'] / 'completion.json'
        env.append('PIPELINE_DONE=' + str(marker))
        argv += ['-m', 'src.training.main']
    workers = '0' if set(overlays) & {'band_noise', 'band_noise_consistency'} else '2'
    argv += ['--config', configs, '--cell', job['cell'], '--seed', str(job['seed']), '--epochs', '100', '--num_workers', workers]
    if 'recording_head_pcsl0' in overlays:
        argv += ['--lambda_pcsl', '0']
    argv += job.get('extra_args', [])
    if 'cohort' in job:
        argv += ['--selection', 'fixed', '--report_epochs', str(job['epoch']), '--stop_epoch', str(job['epoch'])]
        if job['cohort'] != 'official':
            argv += ['--holdout_device', job['cohort']]
    else:
        argv += ['--selection', 'cv', '--cv_folds', str(job.get('cv_folds', 3)), '--cv_fold', str(job['fold'])]
    training = ' '.join(env) + ' ' + shlex.join(argv)
    if job.get('teacher_unit'):
        teacher = job['teacher_unit']
        target = root / 'outputs/train' / job['run'] / job['cell'] / job['unit'] / 'teacher.npz'
        checkpoint = root / 'checkpoints/train' / teacher['run'] / teacher['cell'] / teacher['unit'] / 'snapshot_epoch_66.pth'
        exported = [str(root / '.venv-train/bin/python'), '-m', 'src.training.distillation', '--units',
                    str(root / 'outputs/train' / teacher['run'] / teacher['cell'] / teacher['unit']),
                    '--checkpoints', str(checkpoint), '--expected_epoch', '66', '--out', str(target)]
        return ' '.join(env) + ' mkdir -p ' + shlex.quote(str(target.parent)) + ' && ' + \
            ' '.join(env) + ' ' + shlex.join(exported) + ' && ' + training + ' --soft_targets ' + shlex.quote(str(target)) + \
            ' --kd_hard_weight 0.5 --kd_temperature 2'
    return training


def set_phase(spec, phase, jobs):
    """Only job queues/state change; every constituent recipe stays frozen."""
    spec['phase'], spec['jobs'] = phase, jobs
    for machine, config in spec['machines'].items():
        config['gpu_gb'] = GPU_BUDGETS[machine]
    machines = [m for m, config in spec['machines'].items() if config.get('admit', True)]
    if not machines:
        raise ValueError('No workers meet phase admission capacity')
    weights = {m: float(spec['machines'][m].get('throughput_weight', 0.55 if m == 'A' else 1.0)) for m in machines}
    if any(not np.isfinite(weight) or weight <= 0 for weight in weights.values()):
        raise ValueError('Worker throughput weights must be finite and positive')
    loads = dict.fromkeys(machines, 0.0)
    costs = {'recording_head': 0.5, 'recording_head_pcsl0': 0.5, 'band_noise': 1.5, 'band_noise_consistency': 2.0,
             'multiscale': 2.0, 'swa': 1.3, 'lora': 0.7, 'lora_class_aware': 0.7}
    for i, job in enumerate(jobs):
        heavy = job['arm'] in RECORDING_ARMS | {'band_noise', 'band_noise_consistency'}
        preferred = ['F', *machines] if heavy and 'F' in machines else machines[i % len(machines):] + machines[:i % len(machines)]
        members = spec['stack'] if job['arm'] == 'stack' else [job['arm']]
        cost = max(costs.get(arm, 1.0) for arm in members)
        if job.get('machine') not in machines:
            job['machine'] = min(dict.fromkeys(preferred), key=lambda m: (loads[m] + cost) / weights[m])
        loads[job['machine']] += cost
    directory = Path(spec['snapshot']) / 'logs' / programme_name(spec) / phase
    directory.mkdir(parents=True, exist_ok=True)
    for machine in machines:
        path = directory / (machine + '.jobs.txt')
        path.write_text(''.join(command(spec, j) + '\n' for j in jobs if j['machine'] == machine))
    spec.setdefault('phases', {})[phase] = {'jobs': jobs, 'created': datetime.now(timezone.utc).isoformat()}


def transition(spec):
    """Advance only after certified source/final outputs; target scores never gate selection."""
    phase = spec['phase']
    if phase == 'screen':
        anchor = source_reports(spec, 'anchor', [0])
        arms = spec.get('arms', ARMS)
        if len(arms) != len(set(arms)) or set(arms) - (set(ARMS) | RECORDING_ARMS):
            raise ValueError('Screen arms must be distinct supported recipes')
        results = {a: gate(source_reports(spec, a, [0]), anchor, a in RECORDING_ARMS) for a in arms}
        spec['screen_results'] = results
        recording = sorted((a for a in arms if a in RECORDING_ARMS and results[a]['pass']),
                           key=lambda a: results[a]['mean_delta'], reverse=True)[:1]
        cycle = sorted((a for a in arms if a not in RECORDING_ARMS and results[a]['pass']),
                       key=lambda a: results[a]['mean_delta'], reverse=True)
        winners = cycle[:3 - len(recording)] + recording
        spec['shortlist'] = winners
        if winners:
            set_phase(spec, 'confirm', cv_jobs(['anchor', *winners], [1], name=programme_name(spec)))
        else:
            spec['phase'] = 'complete_no_survivors'
    elif phase == 'confirm':
        anchor = source_reports(spec, 'anchor', [0, 1])
        results = {a: gate(source_reports(spec, a, [0, 1]), anchor, a in RECORDING_ARMS) for a in spec['shortlist']}
        spec['confirm_results'] = results
        winners = sorted((a for a in spec['shortlist'] if results[a]['pass']), key=lambda a: results[a]['mean_delta'], reverse=True)
        spec['winners'] = winners
        spec['stack'] = stack_members([a for a in winners if results[a]['cycle_eligible']])
        if spec['stack']:
            set_phase(spec, 'stack', cv_jobs(['stack'], [0, 1], name=programme_name(spec)))
        else:
            select_final(spec)
    elif phase == 'stack':
        recording = bool(RECORDING_ARMS & set(spec['stack']))
        result = gate(source_reports(spec, 'stack', [0, 1]), source_reports(spec, 'anchor', [0, 1]), recording)
        spec['stack_result'] = result
        select_final(spec)
    elif phase == 'final':
        spec['phase'] = 'report'
    else:
        raise ValueError('No transition from ' + phase)


def select_final(spec):
    ranked = [(a, spec['confirm_results'][a]) for a in spec['winners']]
    if spec.get('stack_result', {}).get('pass'):
        ranked.append(('stack', spec['stack_result']))
    def is_recording(arm):
        return arm in RECORDING_ARMS or (arm == 'stack' and bool(RECORDING_ARMS & set(spec['stack'])))
    selected = []
    for recording in (False, True):
        candidates = [item for item in ranked if is_recording(item[0]) == recording]
        if candidates:
            selected.append(max(candidates, key=lambda item: item[1]['mean_delta']))
    spec['selected'] = {arm: result['epoch'] for arm, result in selected}
    if spec['selected']:
        set_phase(spec, 'final', final_jobs(spec['selected'], name=programme_name(spec)))
    else:
        spec['phase'] = 'complete_no_survivors'


def verify_source(spec):
    for name, digest in spec['source_hashes'].items():
        verify_file(Path(spec['snapshot']) / name, digest)


def unit_path(spec, job, kind='outputs'):
    return Path(spec['snapshot']) / kind / 'train' / job['run'] / job['cell'] / job['unit']


def canonical_path(spec, name, out):
    """Map a completion-receipt entry to this snapshot.

    Receipts keep the path the writing machine used, for example an HDD mount on A, so a relative entry is resolved
    under the unit and an absolute one is rebased onto the snapshot's outputs or checkpoints by its `<kind>/train/...` tail.
    Archive paths are returned unchanged.
    """
    path = Path(name)
    if not path.is_absolute():
        return (out / name).resolve()
    if 'archive' not in path.parts:
        for i in range(len(path.parts) - 1, 0, -1):
            if path.parts[i] == 'train' and path.parts[i - 1] in ('outputs', 'checkpoints'):
                return (Path(spec['snapshot']) / Path(*path.parts[i - 1:])).resolve()
    return path.resolve()


def verify_job(spec, job):
    out = unit_path(spec, job)
    marker = out / ('pipeline-complete.json' if 'cohort' in job else 'completion.json')
    if not marker.is_file():
        return False
    # Receipts retain absolute worker paths; all workers share the frozen snapshot path.
    files = load(marker)['files']
    for name, digest in files.items():
        artifact = canonical_path(spec, name, out)
        if not artifact.is_file():
            return False
        verify_file(artifact, digest)
    if 'cohort' not in job:
        source_curve(load(out / 'report.json'))
        with np.load(out / 'val_preds.npz', allow_pickle=False) as arrays:
            curve = source_curve(load(out / 'report.json'))
            expected = np.flatnonzero(np.isfinite(curve)) + 1
            if not np.array_equal(arrays['epochs'], expected) or arrays['probs'].shape[:2] != (len(expected), len(arrays['labels'])):
                raise ValueError('Missing source OOF epochs or rows')
        checkpoint = unit_path(spec, job, 'checkpoints') / 'best.pth'
    else:
        checkpoint = unit_path(spec, job, 'checkpoints') / f'report_epoch_{job["epoch"]}.pth'
    if not checkpoint.exists():
        return False
    digest = sha256_file(checkpoint)
    receipt = out / 'collected-checkpoint.json'
    if receipt.exists():
        verify_file(checkpoint, load(receipt)['sha256'])
    else:
        atomic_json(receipt, {'path': str(checkpoint), 'sha256': digest, 'machine': job['machine']})
    return True


def _certified_checkpoint(spec, job):
    """Only a published receipt may authorize checkpoint collection."""
    out = unit_path(spec, job).resolve()
    checkpoint = (unit_path(spec, job, 'checkpoints') /
                  (f'report_epoch_{job["epoch"]}.pth' if 'cohort' in job else 'best.pth')).resolve()
    marker = out / ('pipeline-complete.json' if 'cohort' in job else 'completion.json')
    if not marker.is_file():
        return None
    receipt = load(marker)
    expected = receipt.get('checkpoint_sha256') if 'cohort' in job else None
    for name, digest in receipt['files'].items():
        path = canonical_path(spec, name, out)
        if path == checkpoint:
            expected = digest
        else:
            if not (path.is_relative_to(out) or path.is_relative_to(checkpoint.parent)):
                raise ValueError('Completion receipt references another training unit')
            if not path.is_file():
                return None
            verify_file(path, digest)
    if not expected:
        raise ValueError('Completion receipt has no finalized checkpoint hash')
    return checkpoint, expected



def _unlink_f_replica(request):
    """Remote stdlib-only guard; the controller verifies canonical and archive hashes."""
    import hashlib
    import json
    import os
    import re
    import tempfile
    from pathlib import Path

    def sha(path):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b''):
                digest.update(chunk)
        return digest.hexdigest()

    root = Path(request['snapshot'])
    checkpoint = Path(request['checkpoint'])
    if request.get('machine') != 'F' or not root.is_absolute() or root.resolve() != root or checkpoint.resolve() != checkpoint:
        raise ValueError('Replica reclamation requires an exact F snapshot path')
    base = root / 'checkpoints/train'
    if not checkpoint.is_relative_to(base):
        raise ValueError('Checkpoint is outside programme storage')
    parts = checkpoint.relative_to(base).parts
    programme = request.get('programme_name', 'improvements-20261026')
    if programme not in {'improvements-20261026', 'followup-20261026', 'next-improvements-20261026',
                         'fivefold-20261026', 'target-20261027'}:
        raise ValueError('Unsupported replica programme')
    if (len(parts) != 4 or not (parts[0] == programme or parts[0].startswith(programme + '-'))
            or not (parts[3] == 'best.pth' or re.fullmatch(r'report_epoch_[0-9]+\.pth', parts[3]))):
        raise ValueError('Only finalized programme checkpoint replicas may be reclaimed')
    expected = request['checkpoint_sha256']
    archive = root / 'archive/checkpoints/train' / Path(*parts)
    if (not re.fullmatch('[0-9a-f]{64}', expected) or request.get('a_sha256') != expected or
            request.get('b_sha256') != expected or request.get('archive_path') != str(archive)):
        raise ValueError('Verified canonical and archive copies are required')
    out = root / 'outputs/train' / Path(*parts[:3])
    if out.resolve() != out:
        raise ValueError('Replica output metadata must remain inside the snapshot')
    marker = out / ('completion.json' if parts[3] == 'best.pth' else 'pipeline-complete.json')
    if not marker.is_file():
        return {'status': 'unfinalized'}
    if sha(marker) != request['completion_sha256']:
        raise ValueError('Remote completion receipt changed')
    receipt = json.loads(marker.read_text())
    bound = receipt.get('checkpoint_sha256') if parts[3] != 'best.pth' else None
    for name, digest in receipt['files'].items():
        path = (Path(name) if Path(name).is_absolute() else out / name).resolve()
        if path == checkpoint:
            bound = digest
    if bound != expected:
        raise ValueError('Remote receipt does not certify this checkpoint')
    if any((checkpoint.parent / name).exists() for name in ('last.pth', 'last.pth.tmp')):
        return {'status': 'active'}
    if not checkpoint.is_file():
        return {'status': 'already_archived'}
    if sha(checkpoint) != expected:
        raise ValueError('Remote checkpoint differs from its completion receipt')
    pointer = {**request, 'programme_name': programme, 'archive_machine': request.get('archive_machine', 'B'), 'status': 'archived', 'bytes': checkpoint.stat().st_size}
    with tempfile.NamedTemporaryFile(mode='w', dir=out, prefix='.archive-', delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(pointer, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, out / 'archived-checkpoint.json')
    checkpoint.unlink()
    return pointer


def reclaim_f_replicas(spec):
    """Keep canonical and archive copies while freeing finalized F replicas below 35 GiB."""
    archive_machine = spec.get('archive_machine', 'B')
    if 'F' not in spec['machines'] or archive_machine not in spec['machines'] or spec['machines']['F'].get('local'):
        return
    def free_gib():
        result = run_command(spec, 'F', ['df', '-B1', '--output=avail', spec['snapshot']], capture_output=True, text=True, timeout=30)
        return int(result.stdout.splitlines()[-1]) / 1024 ** 3
    if free_gib() >= 35:
        return
    jobs = { (j['run'], j['cell'], j['unit']): j for phase in spec.get('phases', {}).values() for j in phase['jobs'] }
    jobs.update({(j['run'], j['cell'], j['unit']): j for j in spec['jobs']})
    for job in jobs.values():
        if job['machine'] != 'F':
            continue
        certified = _certified_checkpoint(spec, job)
        if certified is None:
            continue
        checkpoint, expected = certified
        out = unit_path(spec, job)
        if not checkpoint.is_file():
            continue
        collected = out / 'collected-checkpoint.json'
        if not collected.is_file():
            continue
        saved = load(collected)
        archive_path = Path(spec['snapshot']) / 'archive/checkpoints/train' / job['run'] / job['cell'] / job['unit'] / checkpoint.name
        if saved.get('f_reclaimed') or saved.get('sha256') != expected or saved.get('archive_path') != str(archive_path) or saved.get('archive_machine', 'B') != archive_machine:
            continue
        verify_file(checkpoint, expected)
        remote = run_command(spec, archive_machine, ['sha256sum', str(archive_path)], capture_output=True, text=True, timeout=60)
        if not remote.stdout.split() or remote.stdout.split()[0] != expected:
            raise ValueError(f'Cannot reclaim F replica without a verified {archive_machine} archive')
        marker = out / ('pipeline-complete.json' if 'cohort' in job else 'completion.json')
        request = dict(snapshot=spec['snapshot'], programme_name=programme_name(spec), archive_machine=archive_machine, machine='F', checkpoint=str(checkpoint), checkpoint_sha256=expected,
                       a_sha256=expected, b_sha256=expected, archive_path=str(archive_path), completion_sha256=sha256_file(marker))
        program = inspect.getsource(_unlink_f_replica) + '\nimport json,sys\nprint(json.dumps(_unlink_f_replica(json.loads(sys.argv[1]))))'
        result = run_command(spec, 'F', [str(Path(spec['snapshot']) / '.venv-train/bin/python'), '-c', program, json.dumps(request)],
                             capture_output=True, text=True, timeout=120)
        pointer = json.loads(result.stdout)
        if pointer.get('status') in ('archived', 'already_archived'):
            saved['f_reclaimed'] = True
            atomic_json(collected, saved)
        if pointer.get('status') == 'archived':
            atomic_json(out / 'archived-checkpoint.json', pointer)
            spec.setdefault('storage_reclamation', []).append(pointer)
            if free_gib() >= 35:
                break


def collect(spec):
    root = Path(spec['snapshot'])
    archive_machine = spec.get('archive_machine', 'B')
    transport = ['-e', 'ssh -o BatchMode=yes -o ConnectTimeout=10']
    # Mutable output metadata can update while training; checkpoints never enter this pass.
    for machine, config in spec['machines'].items():
        if config.get('local'):
            continue
        for run in sorted({job['run'] for job in spec['jobs'] if job['machine'] == machine}):
            destination = root / 'outputs/train' / run
            destination.mkdir(parents=True, exist_ok=True)
            run_command(spec, machine, ['mkdir', '-p', str(destination)])
            result = subprocess.run(['rsync', '-az', '--include=*/', '--include=*.json', '--include=*.npz',
                                     '--include=*.sha256', '--exclude=*', *transport,
                                     f'{config["host"]}:{destination}/', str(destination) + '/'], timeout=300)
            if result.returncode not in (0, 24):
                raise subprocess.CalledProcessError(result.returncode, result.args)
    # Snapshot teachers are separate checkpoints in the same unit's receipt.
    for job in spec['jobs']:
        out = unit_path(spec, job).resolve()
        marker = out / ('pipeline-complete.json' if 'cohort' in job else 'completion.json')
        if not marker.is_file():
            continue
        checkpoint_dir = unit_path(spec, job, 'checkpoints').resolve()
        main_checkpoint = checkpoint_dir / (f'report_epoch_{job["epoch"]}.pth' if 'cohort' in job else 'best.pth')
        for name, digest in load(marker)['files'].items():
            path = canonical_path(spec, name, out)
            if path == main_checkpoint or not path.is_relative_to(checkpoint_dir):
                continue
            if path.is_file() and sha256_file(path) == digest:
                continue
            config = spec['machines'][job['machine']]
            if config.get('local'):
                raise ValueError('Local snapshot differs from its completion receipt')
            path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['rsync', '-az', '--checksum', *transport,
                            f'{config["host"]}:{path}', str(path)], check=True, timeout=300)
            verify_file(path, digest)
    for job in spec['jobs']:
        certified = _certified_checkpoint(spec, job)
        if certified is None:
            continue
        checkpoint, expected = certified
        out = unit_path(spec, job)
        receipt_path = out / 'collected-checkpoint.json'
        saved = load(receipt_path) if receipt_path.exists() else {}
        stat = checkpoint.stat() if checkpoint.is_file() else None
        cached = (stat is not None and saved.get('sha256') == expected and
                  saved.get('size') == stat.st_size and saved.get('mtime_ns') == stat.st_mtime_ns)
        if not cached:
            matches = checkpoint.is_file() and sha256_file(checkpoint) == expected
            if not matches:
                machine = job['machine']
                config = spec['machines'][machine]
                pointer_path = out / 'archived-checkpoint.json'
                restore = load(pointer_path) if machine == 'F' and pointer_path.exists() else None
                expected_archive = root / 'archive/checkpoints/train' / job['run'] / job['cell'] / job['unit'] / checkpoint.name
                if machine == 'F' and restore is None and saved.get('sha256') == expected and saved.get('archive_path') == str(expected_archive):
                    restore = {'checkpoint_sha256': expected, 'archive_path': str(expected_archive),
                               'archive_machine': saved.get('archive_machine', 'B')}
                if restore is not None:
                    archive_path = root / 'archive/checkpoints/train' / job['run'] / job['cell'] / job['unit'] / checkpoint.name
                    if (restore.get('checkpoint_sha256') != expected or restore.get('archive_path') != str(archive_path) or
                            restore.get('archive_machine', 'B') != archive_machine):
                        raise ValueError('Archived checkpoint pointer differs from completion receipt')
                    config = spec['machines'][archive_machine]
                    source = str(archive_path) if config.get('local') else f'{config["host"]}:{archive_path}'
                else:
                    source = f'{config.get("host", "")}:{checkpoint}'
                if config.get('local') and restore is None:
                    raise ValueError('Finalized local checkpoint differs from its completion receipt')
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(['rsync', '-az', '--checksum', *transport,
                                source, str(checkpoint)], check=True, timeout=300)
            verify_file(checkpoint, expected)
            stat = checkpoint.stat()
            archive_path = saved.get('archive_path') if saved.get('sha256') == expected else None
            previous_archive_machine = saved.get('archive_machine', 'B')
            saved = {'path': str(checkpoint), 'sha256': expected, 'machine': job['machine'],
                     'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
            if archive_path is not None:
                saved['archive_path'] = archive_path
                saved['archive_machine'] = previous_archive_machine
            atomic_json(receipt_path, saved)
        archive = spec['machines'].get(archive_machine)
        destination = root / 'archive/checkpoints/train' / job['run'] / job['cell'] / job['unit'] / checkpoint.name
        if archive and (saved.get('archive_path') != str(destination) or saved.get('archive_machine', 'B') != archive_machine):
            run_command(spec, archive_machine, ['mkdir', '-p', str(destination.parent)])
            target = str(destination) if archive.get('local') else f'{archive["host"]}:{destination}'
            subprocess.run(['rsync', '-az', '--checksum', *transport, str(checkpoint), target], check=True, timeout=300)
            digest = run_command(spec, archive_machine, ['sha256sum', str(destination)], capture_output=True, text=True, timeout=60)
            if digest.stdout.split()[0] != expected:
                raise ValueError('Archived finalized checkpoint failed its hash check')
            saved['archive_path'] = str(destination)
            saved['archive_machine'] = archive_machine
            atomic_json(receipt_path, saved)
    reclaim_f_replicas(spec)


def launch(spec, manifest):
    root = Path(spec['snapshot'])
    phase = spec['phase']
    for machine in spec['machines']:
        if not any(j['machine'] == machine for j in spec['jobs']):
            continue
        if not spec['machines'][machine].get('local'):
            destination = root / 'logs' / programme_name(spec)
            run_command(spec, machine, ['mkdir', '-p', str(destination)])
            subprocess.run(['rsync', '-az', str(Path(manifest).parent) + '/',
                            f'{spec["machines"][machine]["host"]}:{destination}/'], check=True, timeout=120)
        service = f'auscult-{programme_name(spec)}-{phase}-{machine}'
        exec_start = shlex.join([str(root / '.venv-train/bin/python'), '-u', str(root / 'scripts/improvement_campaign.py'),
                                 'worker', str(Path(manifest).resolve()), '--machine', machine, '--phase', phase])
        content = ('[Unit]\nDescription=AuscultTrust frozen improvement phase\nAfter=network-online.target\n'
                   '[Service]\nType=simple\nWorkingDirectory=' + str(root) + '\nExecStart=' + exec_start +
                   '\nRestart=on-failure\nRestartForceExitStatus=SIGTERM\nRestartSec=60\nKillMode=process\n[Install]\nWantedBy=default.target\n')
        script = ('mkdir -p "$HOME/.config/systemd/user" && cat > "$HOME/.config/systemd/user/' + service + '.service" && '
                  'systemctl --user daemon-reload && systemctl --user enable --now ' + shlex.quote(service + '.service'))
        run_command(spec, machine, ['bash', '-c', script], input=content, text=True)


def worker(spec, machine, phase):
    if spec['phase'] != phase:
        return
    verify_source(spec)
    root = Path(spec['snapshot'])
    m = spec['machines'][machine]
    env = {**os.environ, 'AT_DATA': m.get('data', spec['data']), 'AT_REPOS': m.get('references', spec['references']),
           'RUN_ID': programme_name(spec), 'SWEEP_TAG': '-' + phase + '-' + machine, 'PYTHONPATH': str(root)}
    subprocess.run([str(root / '.venv-train/bin/python'), '-u', 'scripts/sweep.py',
                    str(root / 'logs' / programme_name(spec) / phase / (machine + '.jobs.txt')),
                    '--workers', str(m.get('workers', 2)), '--gpu-gb', str(m.get('gpu_gb', GPU_BUDGETS[machine])),
                    '--mem-gb', '11', '--ram-gb', '20', '--disk-gb', '30', '--threads', '3', '--num-workers', '2',
                    '--retries', '0', '--deadline', spec['deadline']], cwd=root, env=env, check=True)


def prepare(args):
    root = Path(args.snapshot).resolve()
    path = root / 'logs' / NAME / 'programme.json'
    if path.exists():
        raise FileExistsError(path)
    if datetime.fromisoformat(args.deadline).tzinfo is None:
        raise ValueError('Deadline must include timezone')
    machines = load(args.hosts)
    if not machines or set(machines) - {'A', 'B', 'D', 'E', 'F'}:
        raise ValueError('Expected named A/B/D/E/F workers')
    files = [p for folder in ('src', 'scripts', 'experiments', 'tests') for p in (root / folder).rglob('*')
             if p.is_file() and p.suffix in ('.py', '.yaml', '.json', '.sh') and '__pycache__' not in p.parts]
    files += [root / name for name in ('requirements-train.lock', 'requirements-references.lock.json', 'pyproject.toml')]
    for name in BASE + ARMS:
        if not (root / 'experiments/train' / (name + '.yaml')).is_file():
            raise FileNotFoundError(name)
    baselines = load(args.baselines)
    if set(baselines) != {'control', 'gain'}:
        raise ValueError('Both fixed historical control and gain baselines are required')
    for mapping in baselines.values():
        if set(mapping) != {'official', *DEVICES} or any(set(v) != {str(s) for s in range(5)} for v in mapping.values()):
            raise ValueError('Baselines need five paired seeds for official and all four devices')
        if any(not Path(p).is_absolute() for cohort in mapping.values() for p in cohort.values()):
            raise ValueError('Baseline unit paths must be absolute')
    spec = dict(name=NAME, snapshot=str(root), data=args.data, references=args.references, machines=machines,
                deadline=args.deadline, baselines=baselines, stack=[],
                source_hashes={str(p.relative_to(root)): sha256_file(p) for p in sorted(set(files))},
                frozen_at=datetime.now(timezone.utc).isoformat())
    set_phase(spec, 'screen', cv_jobs(['anchor', *ARMS], [0]))
    atomic_json(path, spec)
    print(path)


def run(manifest, once=False):
    manifest = Path(manifest).resolve()
    with (manifest.parent / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            spec = load(manifest)
            verify_source(spec)
            if spec['phase'] in ('report', 'complete_no_survivors'):
                if not audits_complete(spec):
                    spec['status'] = {'at': datetime.now(timezone.utc).isoformat(), 'waiting_for_source_audits': audit_results(spec)}
                    atomic_json(manifest, spec)
                    if once:
                        return
                    time.sleep(45)
                    continue
                if spec['phase'] == 'complete_no_survivors':
                    negative_report(spec)
                else:
                    report(spec)
                    spec['phase'] = 'complete'
                    atomic_json(manifest, spec)
                return
            if spec['phase'] == 'complete':
                return
            collect(spec)
            complete = [verify_job(spec, job) for job in spec['jobs']]
            if spec['phase'] == 'screen':
                for job, done in zip(spec['jobs'], complete):
                    if done and job['arm'] == 'anchor':
                        launch_audit(spec, job)
            spec['status'] = {'at': datetime.now(timezone.utc).isoformat(), 'complete': sum(complete), 'units': len(complete)}
            deadline = datetime.now(timezone.utc) >= datetime.fromisoformat(spec['deadline'])
            if all(complete):
                refresh_capacity(spec)
                transition(spec)
                atomic_json(manifest, spec)
                if not deadline and spec['phase'] not in ('report', 'complete_no_survivors'):
                    launch(spec, manifest)
            else:
                atomic_json(manifest, spec)
                if not deadline:
                    launch(spec, manifest)
                else:
                    spec['status']['admission_closed'] = True
                    atomic_json(manifest, spec)
            if once:
                return
            time.sleep(45)





def launch_audit(spec, job):
    """Run frozen source preprocessing comparisons off GPUs without holding the next phase."""
    root = Path(spec['snapshot'])
    key = 'cv' + str(job['fold'])
    if key in spec.setdefault('audits', {}):
        return
    destination = root / 'outputs/evaluation' / programme_name(spec) / 'preprocessing' / key
    destination.mkdir(parents=True, exist_ok=True)
    service = f'auscult-{programme_name(spec)}-audit-{key}'
    command = ['/usr/bin/env', 'PYTHONPATH=' + str(root), 'CUDA_VISIBLE_DEVICES=', 'OMP_NUM_THREADS=2', 'OPENBLAS_NUM_THREADS=2',
               'MKL_NUM_THREADS=2', 'AT_DATA=' + spec['data'], 'AT_REPOS=' + spec['references'],
               str(root / '.venv-train/bin/python'), '-u', str(root / 'scripts/preprocessing_audit.py'),
               '--unit-dir', str(unit_path(spec, job)), '--out', str(destination), '--max-recordings', '16']
    content = ('[Unit]\nDescription=Frozen source preprocessing audit\n[Service]\nType=simple\nWorkingDirectory=' + str(root) +
               '\nExecStart=' + shlex.join(command) + '\nRestart=on-failure\nRestartSec=60\n[Install]\nWantedBy=default.target\n')
    directory = Path.home() / '.config/systemd/user'
    directory.mkdir(parents=True, exist_ok=True)
    (directory / (service + '.service')).write_text(content)
    subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
    subprocess.run(['systemctl', '--user', 'enable', '--now', service + '.service'], check=True)
    spec['audits'][key] = {'service': service, 'summary': str(destination / 'summary.json'),
                           'checkpoint_sha256': load(unit_path(spec, job) / 'collected-checkpoint.json')['sha256']}


def audit_results(spec):
    results = {}
    for key, item in spec.get('audits', {}).items():
        path = Path(item['summary'])
        results[key] = {**item, 'status': 'pending'}
        if path.is_file():
            try:
                comparison = load(path).get('frozen_source_comparison', {})
            except json.JSONDecodeError:
                continue
            if comparison.get('status') == 'complete':
                if comparison.get('checkpoint_sha256') != item['checkpoint_sha256']:
                    raise ValueError('Preprocessing audit checkpoint differs from certified source checkpoint')
                results[key].update(status='complete', sha256=sha256_file(path))
    return results


def audits_complete(spec):
    results = audit_results(spec)
    return set(results) == {'cv0', 'cv1', 'cv2'} and all(row['status'] == 'complete' for row in results.values())


def refresh_capacity(spec):
    """At phase boundaries, leave low-disk workers out of new queues."""
    for machine, config in spec['machines'].items():
        result = run_command(spec, machine, ['df', '-B1', '--output=avail', spec['snapshot']],
                             capture_output=True, text=True, timeout=30)
        available = int(result.stdout.splitlines()[-1]) / 1024 ** 3
        config['available_disk_gb'] = available
        config['admit'] = available >= 33


def source_audit(spec):
    audit = {key: spec.get(key) for key in ('screen_results', 'confirm_results', 'stack_result', 'stack', 'selected', 'source_hashes')}
    audit['operational_controller_sha256'] = sha256_file(__file__)
    audit['scope_amendments'] = spec.get('scope_amendments', [])
    return audit


def negative_report(spec):
    destination = Path(spec['snapshot']) / 'outputs/evaluation' / programme_name(spec)
    audit = source_audit(spec)
    atomic_json(destination / 'summary.json', {'status': 'complete_no_survivors', 'source_audit': audit, 'preprocessing_audits': audit_results(spec),
                'scope': 'All source-CV results retained; no candidate met the source gates. No new target evaluation.'})
    lines = ['# Source improvement screen: no survivors', '', '| Arm | Mean source delta | Positive folds | Advanced |', '| --- | --- | --- | --- |']
    for arm, result in spec['screen_results'].items():
        lines.append(f"| {arm} | {result['mean_delta']:.4f} | {sum(d > 0 for d in result['fold_deltas'])}/3 | {result['pass']} |")
    (destination / 'report.md').write_text('\n'.join(lines) + '\n')
    atomic_json(destination / 'complete.json', {'status': 'complete_no_survivors',
                'summary_sha256': sha256_file(destination / 'summary.json')})


def recording_endpoints(path):
    """Keep learned binary heads separate from the two cycle aggregation controls."""
    with np.load(path, allow_pickle=True) as archive:
        data = {key: archive[key] for key in archive.files}
    filters = data.get('filter', np.full(len(data['stem']), ''))
    keys = list(zip(data['stem'].astype(str), filters.astype(str)))
    groups = {key: np.array([i for i, value in enumerate(keys) if value == key]) for key in sorted(set(keys))}
    output = {}
    for aggregation in ('mean', 'max'):
        abnormal = np.array([getattr(np, aggregation)(1 - data['probs'][rows, 0]) for rows in groups.values()])
        first = [rows[0] for rows in groups.values()]
        output['cycle_' + aggregation] = dict(test_p=np.stack([1 - abnormal, abnormal], 1),
            test_y=(data['label'][first] > 0).astype(int), test_pat=data['patient'][first],
            stem=np.array([key[0] for key in groups]), filter=np.array([key[1] for key in groups]))
        if 'device' in data:
            output['cycle_' + aggregation]['device'] = data['device'][first]
    if 'recording_probs' in data:
        rfilters = data.get('recording_filter', np.full(len(data['recording_stem']), ''))
        order = sorted(range(len(rfilters)), key=lambda i: (str(data['recording_stem'][i]), str(rfilters[i])))
        output['learned_head'] = dict(test_p=data['recording_probs'][order],
            test_y=(data['recording_label'][order] > 0).astype(int), test_pat=data['recording_patient'][order],
            stem=data['recording_stem'][order], filter=rfilters[order])
        if 'recording_device' in data:
            output['learned_head']['device'] = data['recording_device'][order]
    return output



def recording_triage(screening_path, kauh_path):
    """Calibrate each actual endpoint on patient-disjoint ICBHI halves, then transfer to KAUH."""
    import pandas as pd
    from src.evaluation.screening import triage
    from src.evaluation.conformal import lac_scores, split_threshold
    from src.processing.splits import calibration_eval_splits
    source, target = recording_endpoints(screening_path), recording_endpoints(kauh_path)
    rows = []
    for endpoint, ic in source.items():
        kr = target[endpoint]
        table = pd.DataFrame({'patient': ic['test_pat'], 'device': ic['device']})
        collected = {}
        for split in calibration_eval_splits(table, n_splits=20, seed=0):
            calibration = np.isin(ic['test_pat'], split['calibration'])
            evaluation = np.isin(ic['test_pat'], split['evaluation'])
            if np.any(calibration & evaluation) or not calibration.any() or not evaluation.any():
                raise ValueError('Recording calibration requires disjoint nonempty patient halves')
            threshold = split_threshold(lac_scores(ic['test_p'][calibration], ic['test_y'][calibration]), 0.1)
            prior = np.bincount(ic['test_y'][calibration], minlength=2) / calibration.sum()
            collected.setdefault('ICBHI evaluation half', []).append(
                triage(threshold, ic['test_p'][evaluation], ic['test_y'][evaluation], prior))
            for name in np.unique(kr['filter']):
                selected = kr['filter'] == name
                collected.setdefault('KAUH ' + str(name), []).append(
                    triage(threshold, kr['test_p'][selected], kr['test_y'][selected], prior))
        for dataset, values in collected.items():
            rows.append(dict(endpoint=endpoint, dataset=dataset, resplits=len(values),
                             **{key: float(np.nanmean([v[key] for v in values])) for key in values[0]}))
    return rows


def report(spec):
    """Frozen comparisons against both specified historical baselines; no selection here."""
    import shutil
    from scripts.improvement_report import paired_score_interval
    from src.evaluation.ensemble_study import classification
    from src.evaluation.lodo import load_unit
    from src.evaluation.calibration import coverage
    root = Path(spec['snapshot'])
    destination = root / 'outputs/evaluation' / programme_name(spec)
    rows, intervals, inputs = [], {}, {}
    recording_rows, recording_pairs, triage_rows = [], {}, []
    for arm in spec['selected']:
        for cohort in ['official', *DEVICES]:
            comparisons = {baseline: [] for baseline in spec['baselines']}
            for seed in range(5):
                job = next(j for j in spec['jobs'] if j['arm'] == arm and j['cohort'] == cohort and j['seed'] == seed)
                paths = {arm: unit_path(spec, job), **{b: Path(mapping[cohort][str(seed)]) for b, mapping in spec['baselines'].items()}}
                units, recording_units = {}, {}
                for label, path in paths.items():
                    marker = path / 'pipeline-complete.json'
                    receipt = load(marker)
                    if label in ('control', 'gain') and receipt['report_epoch'] != {'control': 21, 'gain': 67}[label]:
                        raise ValueError('Historical baseline epoch does not match the frozen comparator')
                    for name, digest in receipt['files'].items():
                        # Historical receipts may name their original frozen snapshot.
                        original = Path(name)
                        tail = original.parts[original.parts.index('seed' + str(seed)) + 1:]
                        verify_file(path.joinpath(*tail), digest)
                    inputs[str(marker)] = sha256_file(marker)
                    if cohort == 'official':
                        with np.load(path / 'export/test.npz', allow_pickle=True) as z:
                            u = dict(test_p=z['probs'].astype(np.float32), test_y=z['labels'], test_pat=z['patient'])
                        for kind in ('screening', 'kauh'):
                            output = root / 'outputs' / kind / programme_name(spec) / label / f'seed{seed}.npz'
                            output.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(path / (kind + '.npz'), output)
                            endpoints = recording_endpoints(path / (kind + '.npz'))
                            recording_units[label, kind] = endpoints
                            for endpoint, values in endpoints.items():
                                recording_rows.append(dict(candidate=arm, arm=label, dataset=kind, seed=seed, endpoint=endpoint,
                                    **classification(values['test_y'], values['test_p'].argmax(1))))
                        for item in recording_triage(path / 'screening.npz', path / 'kauh.npz'):
                            triage_rows.append(dict(candidate=arm, arm=label, seed=seed, **item))
                    else:
                        u = load_unit(path)
                    units[label] = u
                    row = dict(arm=label, candidate=arm, cohort=cohort, seed=seed,
                               **classification(u['test_y'], u['test_p'].argmax(1)))
                    if cohort != 'official':
                        row.update(coverage(u['cal_p'], u['cal_y'], u['test_p'], u['test_y']))
                    rows.append(row)
                if cohort == 'official':
                    for baseline in comparisons:
                        for kind in ('screening', 'kauh'):
                            head = recording_units[arm, kind].get('learned_head')
                            if head is not None:
                                for endpoint in ('cycle_mean', 'cycle_max'):
                                    reference = recording_units[baseline, kind][endpoint]
                                    for key in ('stem', 'filter', 'test_y', 'test_pat'):
                                        np.testing.assert_array_equal(reference[key], head[key])
                                    key = '/'.join((arm, kind, baseline, endpoint))
                                    recording_pairs.setdefault(key, []).append((reference, head))
                for baseline in comparisons:
                    comparisons[baseline].append((units[baseline], units[arm]))
            intervals[arm + '/' + cohort] = {b: paired_score_interval(pairs) for b, pairs in comparisons.items()}
    for baseline in spec['baselines']:
        comparison = 'against_' + baseline
        for kind in ('screening', 'kauh'):
            folder = root / 'outputs' / kind / programme_name(spec) / comparison
            folder.mkdir(parents=True, exist_ok=True)
            for label in [baseline, *spec['selected']]:
                link = folder / label
                if not link.exists():
                    link.symlink_to(Path('..') / label, target_is_directory=True)
        for module, arguments in [('src.evaluation.kauh', ['--dir', str(root / 'outputs/kauh' / programme_name(spec) / comparison), '--base', baseline,
                                                         '--arms', *spec['selected'], '--boot', '1000', '--null_splits', '200']),
                                  ('src.evaluation.screening', ['analyse', '--dir', str(root / 'outputs/screening' / programme_name(spec) / comparison),
                                                               '--kauh_dir', str(root / 'outputs/kauh' / programme_name(spec) / comparison), '--pairs',
                                                               *[a + ':' + a for a in [baseline, *spec['selected']]]])]:
            subprocess.run([sys.executable, '-m', module, *arguments], cwd=root, check=True)
    recording_intervals = {key: paired_score_interval(pairs) for key, pairs in recording_pairs.items()}
    audit = source_audit(spec)
    atomic_json(destination / 'summary.json', {'rows': rows, 'paired_intervals': intervals, 'recording_rows': recording_rows,
                'recording_head_vs_cycle_intervals': recording_intervals, 'recording_triage_rows': triage_rows,
                'preprocessing_audits': audit_results(spec), 'source_audit': audit, 'inputs': inputs,
                'scope': 'Exploratory paired source-selected comparisons; historical baselines fixed at epochs21/67. No target-based selection.'})
    lines = ['# Source-selected improvement campaign', '', '| Candidate | Cohort | Arm | Score mean ± seed SD |', '| --- | --- | --- | --- |']
    for candidate in spec['selected']:
        for cohort in ['official', *DEVICES]:
            for arm in [*spec['baselines'], candidate]:
                scores = [r['score'] for r in rows if r['candidate'] == candidate and r['cohort'] == cohort and r['arm'] == arm]
                lines.append(f'| {candidate} | {cohort} | {arm} | {np.mean(scores):.3f} ± {np.std(scores, ddof=1):.3f} |')
    lines += ['', 'Paired patient/seed intervals and source gates: summary.json. KAUH and screening comparisons retain a separate directory for each fixed baseline.',
              'Exploratory comparisons: all recipe and epoch choices used source patient CV; target outcomes did not select the survivors.']
    (destination / 'report.md').write_text('\n'.join(lines) + '\n')
    atomic_json(destination / 'complete.json', {'summary_sha256': sha256_file(destination / 'summary.json')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'worker'])
    parser.add_argument('manifest', nargs='?')
    parser.add_argument('--snapshot')
    parser.add_argument('--data')
    parser.add_argument('--references')
    parser.add_argument('--hosts')
    parser.add_argument('--baselines')
    parser.add_argument('--deadline', default='2026-10-26T23:00:00+07:00')
    parser.add_argument('--machine')
    parser.add_argument('--phase')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args)
    elif args.command == 'worker':
        worker(load(args.manifest), args.machine, args.phase)
    else:
        run(args.manifest, args.once)


if __name__ == '__main__':
    main()
