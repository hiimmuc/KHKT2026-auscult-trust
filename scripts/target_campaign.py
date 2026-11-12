"""Official, four-device LODO and KAUH evaluation of five source-validated recipes, plus a PCSL-5 PAFA control.

Phase `main`: source CV of the control (5 folds x 5 seeds) and the fixed-epoch finals of the five recipes.
Phase `control_final`: finals of the control at the epoch chosen from its own source-validation curve.
"""
import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from scripts import improvement_campaign as core
from src.artifacts import atomic_json, sha256_file

NAME = 'target-20261027'
CONTROL = 'pcsl5_control'
REFERENCE = 'fivefold-20261026'
# Source-selected epochs of the five-fold, five-seed source-CV run named by REFERENCE.
FINAL_EPOCHS = {'class_aware_light': 4, 'lora_class_aware': 43, 'lora': 83, 'swa': 43, 'multiscale': 68}
# Median wall hours of one 100-epoch five-fold source fit with two jobs per GPU (that run).
CV_HOURS = {CONTROL: 0.62, 'class_aware_light': 0.75, 'lora_class_aware': 0.60, 'lora': 0.52, 'swa': 0.63, 'multiscale': 0.82}
# Relative speed of one machine; measured median fit time 0.62 h (B, D, E), 0.48 h (F), 1.29 h (A, serves the demo API).
WEIGHTS = {'A': 0.5, 'B': 1.0, 'D': 1.0, 'E': 1.0, 'F': 1.2}
F_GPU_GB = 30  # user limit for the shared machine F


def cost(job):
    if 'cohort' not in job:
        return CV_HOURS[job['arm']]
    return CV_HOURS[job['arm']] * job['epoch'] / 100 * 1.25 + 0.08 + (0.15 if job['cohort'] == 'official' else 0)


def assign(first, rest):
    """Preassign machines: `first` jobs lead each queue, `rest` longest first; weighted least-loaded."""
    load = dict.fromkeys(WEIGHTS, 0.0)
    ordered = [*first, *sorted(rest, key=lambda j: -cost(j))]
    for job in ordered:
        job['machine'] = min(load, key=lambda m: (load[m] + cost(job)) / WEIGHTS[m])
        load[job['machine']] += cost(job)
    return ordered, load


def set_phase(spec, phase, jobs):
    previous = core.GPU_BUDGETS['F']
    core.GPU_BUDGETS['F'] = F_GPU_GB
    try:
        core.set_phase(spec, phase, jobs)
    finally:
        core.GPU_BUDGETS['F'] = previous


def control_cv_jobs():
    return [dict(arm=CONTROL, seed=seed, fold=fold, cv_folds=5, run=f'{NAME}-s{seed}', cell=CONTROL, unit=f'cv{fold}')
            for seed in range(5) for fold in range(5)]


def main_jobs():
    return assign(control_cv_jobs(), core.final_jobs(FINAL_EPOCHS, name=NAME))[0]


def reports(root, run_name, arm, seeds=range(5), folds=range(5)):
    out = []
    for seed in seeds:
        for fold in folds:
            unit = Path(root) / 'outputs/train' / f'{run_name}-s{seed}' / arm / f'cv{fold}'
            report = core.load(unit / 'report.json')
            with np.load(unit / 'val_preds.npz', allow_pickle=False) as data:
                meta = {key: data[key].tolist() for key in ('labels', 'patient', 'stem', 'device')}
            report['_cohort'] = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
            out.append(report)
    return out


def control_result(spec):
    """Control epoch from its own mean curve and its source contrast with the reference five-fold run."""
    reference = Path(spec['reference_snapshot'])
    control = reports(spec['snapshot'], NAME, CONTROL)
    anchor = reports(reference, REFERENCE, 'anchor')
    light = reports(reference, REFERENCE, 'class_aware_light')
    for name, group in (('anchor', anchor), ('class_aware_light', light)):
        if [r['_cohort'] for r in control] != [r['_cohort'] for r in group]:
            raise ValueError(f'Control validation cohorts differ from the reference {name}')
    curves = {name: [core.source_curve(r) for r in group] for name, group in
              (('control', control), ('anchor', anchor), ('class_aware_light', light))}
    epoch = core.cv_epoch(curves['control'])
    mean = lambda name, e: float(np.mean([c[e - 1] for c in curves[name]]))
    return {'epoch': epoch, 'score_at_epoch': {name: {str(e): mean(name, e) for e in sorted({epoch, 4, 66})}
                                                for name in curves},
            'epoch_of': {'anchor': core.cv_epoch(curves['anchor']), 'class_aware_light': core.cv_epoch(curves['class_aware_light'])}}


def prepare(snapshot, hosts, data, reference_snapshot, deadline):
    root = Path(snapshot).resolve()
    manifest = root / 'logs' / NAME / 'programme.json'
    if manifest.exists():
        raise FileExistsError(manifest)
    if datetime.fromisoformat(deadline).tzinfo is None:
        raise ValueError('--deadline needs a UTC offset')
    reference_snapshot = str(Path(reference_snapshot).resolve())
    for arm in ('anchor', 'class_aware_light'):
        reports(reference_snapshot, REFERENCE, arm)  # all 25 reports and cohorts must exist before any training starts
    machines = core.load(hosts)
    if set(machines) != set(WEIGHTS) or not machines['B'].get('local'):
        raise ValueError('B must control A/B/D/E/F')
    files = [p for folder in ('src', 'scripts', 'experiments') for p in (root / folder).rglob('*')
             if p.is_file() and p.suffix in ('.py', '.yaml', '.json', '.sh') and '__pycache__' not in p.parts]
    files += [root / 'requirements-train.lock', root / 'pyproject.toml']
    spec = dict(name=NAME, snapshot=str(root), data=data, machines=machines, archive_machine='E',
                deadline=deadline, stack=[], reference_snapshot=reference_snapshot,
                recipes={arm: [arm] for arm in [*FINAL_EPOCHS, CONTROL]},
                source_hashes={str(p.relative_to(root)): sha256_file(p) for p in sorted(set(files))},
                frozen_at=datetime.now(timezone.utc).isoformat(), status={})
    set_phase(spec, 'main', main_jobs())
    atomic_json(manifest, spec)
    print(manifest)


def finish(spec):
    root = Path(spec['snapshot'])
    jobs = [job for phase in spec['phases'].values() for job in phase['jobs']]
    out = root / 'outputs/evaluation' / NAME
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / 'complete.json', {'status': 'complete', 'fits': len(jobs), 'control': spec['control_result'],
                                        'final_epochs': {**FINAL_EPOCHS, CONTROL: spec['control_result']['epoch']}})
    spec['phase'] = 'complete'


def tick(spec, manifest):
    core.verify_source(spec)
    if spec['phase'] == 'complete':
        return
    core.collect(spec)
    complete = [core.verify_job(spec, job) for job in spec['jobs']]
    spec['status'] = {'at': datetime.now(timezone.utc).isoformat(), 'complete': sum(complete),
                      'units': len(complete), 'phase': spec['phase']}
    if all(complete):
        if spec['phase'] == 'main':
            spec['control_result'] = control_result(spec)
            set_phase(spec, 'control_final', assign([], core.final_jobs({CONTROL: spec['control_result']['epoch']}, name=NAME))[0])
        else:
            finish(spec)
    elif datetime.now(timezone.utc) < datetime.fromisoformat(spec['deadline']):
        core.launch(spec, manifest)
    else:
        spec['status']['admission_closed'] = True
    atomic_json(manifest, spec)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'run'))
    parser.add_argument('manifest', nargs='?')
    parser.add_argument('--snapshot')
    parser.add_argument('--hosts')
    parser.add_argument('--data')
    parser.add_argument('--reference-snapshot')
    parser.add_argument('--deadline', default=(datetime.now(timezone.utc) + timedelta(days=7)).isoformat())
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.snapshot, args.hosts, args.data, args.reference_snapshot, args.deadline)
    else:
        core.run_controller(args.manifest, tick)
