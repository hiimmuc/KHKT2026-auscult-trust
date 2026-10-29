"""Distil equal-weight ensembles of finished final units into one BEATs + PAFA student.

Phase `official`: each configuration (teacher group x distillation loss) on the official split, five seeds.
Phase `lodo`: the configurations that pass the fixed gate, four held-out devices, the first three seeds.
The student is the reference recipe at its reference epoch; teacher targets are exported on the training cycles of the same cohort
and seed from the members' own units, in the same job chain as the student fit.
"""
import argparse
import fcntl
import json
import shlex
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from scripts import improvement_campaign as core
from src.artifacts import atomic_json, sha256_file

NAME = 'distill-20261028'
STUDENT_EPOCH = 66
HERE = Path(__file__).resolve().parents[1]
# run from the repository or from a snapshot under <repository>/build/campaigns
CAMPAIGNS = HERE.parent if HERE.parent.name == 'campaigns' else HERE / 'build/campaigns'
# member -> (campaign directory, run, cell, epoch of its final unit)
MEMBERS = {
    'anchor': ('followup-20261026', 'followup-20261026-final', 'gain6-cv66', 66),
    'control': ('target-20261027', 'target-20261027-final', 'pcsl5_control', 78),
    'cal': ('target-20261027', 'target-20261027-final', 'class_aware_light', 4),
    'lca': ('target-20261027', 'target-20261027-final', 'lora_class_aware', 43),
    'lora': ('target-20261027', 'target-20261027-final', 'lora', 83),
    'swa': ('target-20261027', 'target-20261027-final', 'swa', 43),
    'ms': ('target-20261027', 'target-20261027-final', 'multiscale', 68),
}
GROUPS = {'g4': ['anchor', 'control', 'cal', 'ms'], 'g7': list(MEMBERS)}
LOSSES = {'k1': [], 'k2': ['--kd_hard_weight', '0.5', '--kd_temperature', '2']}
CONFIGS = {'kd_g4_k2': ('g4', 'k2'), 'kd_g7_k2': ('g7', 'k2'), 'kd_g4_k1': ('g4', 'k1')}
WEIGHTS = {'A': 0.5, 'B': 1.0, 'C': 1.0, 'D': 1.0, 'E': 1.0}
LODO_SEEDS = (0, 1, 2)
GATE_DELTA, GATE_SEEDS = 0.3, 3
core_command = core.command
REFERENCE = ('followup-20261026', 'followup-20261026-final', 'gain6-cv66')


def unit_dir(member, cohort, seed, kind):
    campaign, run, cell, _ = MEMBERS[member]
    suffix = '' if cohort == 'official' else '+lodo_' + cohort.lower()
    return CAMPAIGNS / campaign / kind / 'train' / run / (cell + suffix) / f'seed{seed}'


def teacher_files(group, cohort, seed):
    units = [unit_dir(m, cohort, seed, 'outputs') for m in GROUPS[group]]
    checkpoints = [unit_dir(m, cohort, seed, 'checkpoints') / f'report_epoch_{MEMBERS[m][3]}.pth' for m in GROUPS[group]]
    return units, checkpoints


def cost(job):
    members = len(GROUPS[CONFIGS[job['arm']][0]])
    return 0.55 + 0.05 * members


def assign(jobs):
    load = dict.fromkeys(WEIGHTS, 0.0)
    for job in sorted(jobs, key=lambda j: -cost(j)):
        job['machine'] = min(load, key=lambda m: (load[m] + cost(job)) / WEIGHTS[m])
        load[job['machine']] += cost(job)
    return jobs


def student_jobs(arms, cohorts, seeds):
    jobs = []
    for arm in arms:
        group, loss = CONFIGS[arm]
        for job in core.final_jobs({arm: STUDENT_EPOCH}, name=NAME):
            if job['cohort'] in cohorts and job['seed'] in seeds:
                job['extra_args'] = ['--soft_targets', 'teacher.npz', *LOSSES[loss]]
                jobs.append(job)
    return assign(jobs)


def command(spec, job):
    """Export the ensemble targets on this unit's training cycles, then fit the student and its frozen exports."""
    root = Path(spec['snapshot'])
    group = CONFIGS[job['arm']][0]
    target = root / 'outputs/train' / job['run'] / job['cell'] / job['unit'] / 'teacher.npz'
    job = dict(job, extra_args=['--soft_targets', str(target), *job['extra_args'][2:]])
    units, checkpoints = teacher_files(group, job['cohort'], job['seed'])
    env = f'RUN_ID={job["run"]} MEM_GB=11'
    export = [str(root / '.venv-train/bin/python'), '-m', 'src.training.distillation', '--units', *map(str, units),
              '--checkpoints', *map(str, checkpoints), '--out', str(target)]
    return env + ' mkdir -p ' + shlex.quote(str(target.parent)) + ' && ' + env + ' ' + shlex.join(export) + ' && ' + \
        core_command(spec, job)


def official_scores(root, run, cell):
    scores = {}
    for seed in range(5):
        report = core.load(Path(root) / 'outputs/train' / run / cell / f'seed{seed}' / 'report.json')
        scores[seed] = report['last']['test']['all']['score']
    return scores


def gate(student, reference):
    delta = [student[s] - reference[s] for s in sorted(student)]
    return {'mean_delta': float(np.mean(delta)), 'seed_deltas': delta,
            'pass': bool(np.mean(delta) >= GATE_DELTA and sum(d > 0 for d in delta) >= GATE_SEEDS)}


def prepare(snapshot, hosts, data, references, deadline):
    root = Path(snapshot).resolve()
    manifest = root / 'logs' / NAME / 'programme.json'
    if manifest.exists():
        raise FileExistsError(manifest)
    if datetime.fromisoformat(deadline).tzinfo is None:
        raise ValueError('--deadline needs a UTC offset')
    machines = core.load(hosts)
    if set(machines) != set(WEIGHTS) or not machines['B'].get('local'):
        raise ValueError('B must control A/B/C/D/E')
    files = [p for folder in ('src', 'scripts', 'experiments') for p in (root / folder).rglob('*')
             if p.is_file() and p.suffix in ('.py', '.yaml', '.json', '.sh') and '__pycache__' not in p.parts]
    files += [root / 'requirements-train.lock', root / 'pyproject.toml']
    spec = dict(name=NAME, snapshot=str(root), data=data, references=references, machines=machines, archive_machine='E',
                deadline=deadline, stack=[], reference_snapshot=str(CAMPAIGNS / REFERENCE[0]), recipes={arm: [] for arm in CONFIGS},
                source_hashes={str(p.relative_to(root)): sha256_file(p) for p in sorted(set(files))},
                frozen_at=datetime.now(timezone.utc).isoformat(), status={})
    set_phase(spec, 'official', student_jobs(CONFIGS, {'official'}, range(5)))
    atomic_json(manifest, spec)
    print(manifest)


def set_phase(spec, phase, jobs):
    previous = core.command
    core.command = lambda s, j: command(s, j)
    try:
        core.set_phase(spec, phase, jobs)
    finally:
        core.command = previous


def tick(spec, manifest):
    core.verify_source(spec)
    if spec['phase'] in ('complete', 'complete_no_survivors'):
        return
    core.collect(spec)
    complete = [core.verify_job(spec, job) for job in spec['jobs']]
    spec['status'] = {'at': datetime.now(timezone.utc).isoformat(), 'complete': sum(complete),
                      'units': len(complete), 'phase': spec['phase']}
    if all(complete):
        root = Path(spec['snapshot'])
        if spec['phase'] == 'official':
            ref = official_scores(Path(spec['reference_snapshot']), REFERENCE[1], REFERENCE[2])
            spec['gate'] = {arm: gate(official_scores(root, f'{NAME}-final', arm), ref) for arm in CONFIGS}
            survivors = [arm for arm, g in spec['gate'].items() if g['pass']]
            if survivors:
                set_phase(spec, 'lodo', student_jobs(survivors, set(core.DEVICES), LODO_SEEDS))
            else:
                spec['phase'] = 'complete_no_survivors'
        else:
            spec['phase'] = 'complete'
    elif datetime.now(timezone.utc) < datetime.fromisoformat(spec['deadline']):
        core.launch(spec, manifest)
    else:
        spec['status']['admission_closed'] = True
    atomic_json(manifest, spec)


def run(manifest):
    manifest = Path(manifest).resolve()
    with (manifest.parent / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            spec = core.load(manifest)
            tick(spec, manifest)
            if spec['phase'] in ('complete', 'complete_no_survivors'):
                return
            time.sleep(60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'run'))
    parser.add_argument('manifest', nargs='?')
    parser.add_argument('--snapshot')
    parser.add_argument('--hosts')
    parser.add_argument('--data')
    parser.add_argument('--references')
    parser.add_argument('--deadline', default=(datetime.now(timezone.utc) + timedelta(days=2)).isoformat())
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.snapshot, args.hosts, args.data, args.references, args.deadline)
    else:
        run(args.manifest)
