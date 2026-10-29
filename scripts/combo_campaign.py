"""Source CV of one single model that stacks class-aware light PAFA and multi-scale BEATs (5 folds x 5 seeds).

Same protocol as `fivefold_campaign` (matched folds and seeds, epoch from the arm's own mean validation curve), compared
with the finished anchor, class_aware_light and multiscale arms of that run. Source CV only; no target result.
"""
import argparse
import fcntl
import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from scripts import improvement_campaign as core
from src.artifacts import atomic_json, sha256_file

NAME = 'combo-20261028'
ARM = 'class_aware_multiscale'
RECIPE = ['class_aware_light', 'multiscale']  # stacked yaml overlays; multiscale makes core.command reserve 16 GB
REFERENCE = ('fivefold-20261026', ('anchor', 'class_aware_light', 'multiscale'))
HERE = Path(__file__).resolve().parents[1]
CAMPAIGNS = HERE.parent if HERE.parent.name == 'campaigns' else HERE / 'build/campaigns'


def jobs():
    return [dict(arm=ARM, seed=seed, fold=fold, cv_folds=5, run=f'{NAME}-s{seed}', cell=ARM, unit=f'cv{fold}')
            for seed in range(5) for fold in range(5)]


def reports(root, run, arm):
    """Source-validation curves and cohort hashes of the 25 units of one arm."""
    curves, cohorts = [], []
    for seed in range(5):
        for fold in range(5):
            unit = Path(root) / 'outputs/train' / f'{run}-s{seed}' / arm / f'cv{fold}'
            curves.append(core.source_curve(core.load(unit / 'report.json')))
            with np.load(unit / 'val_preds.npz', allow_pickle=False) as data:
                meta = {key: data[key].tolist() for key in ('labels', 'patient', 'stem', 'device')}
            cohorts.append(hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest())
    return curves, cohorts


def prepare(snapshot, hosts, data, references, deadline):
    root = Path(snapshot).resolve()
    manifest = root / 'logs' / NAME / 'programme.json'
    if manifest.exists():
        raise FileExistsError(manifest)
    if datetime.fromisoformat(deadline).tzinfo is None:
        raise ValueError('--deadline needs a UTC offset')
    for arm in REFERENCE[1]:
        reports(CAMPAIGNS / REFERENCE[0], REFERENCE[0], arm)  # the reference units must exist before any training starts
    machines = core.load(hosts)
    if not machines['B'].get('local'):
        raise ValueError('B must control the campaign')
    files = [p for folder in ('src', 'scripts', 'experiments') for p in (root / folder).rglob('*')
             if p.is_file() and p.suffix in ('.py', '.yaml', '.json', '.sh') and '__pycache__' not in p.parts]
    files += [root / 'requirements-train.lock', root / 'pyproject.toml']
    spec = dict(name=NAME, snapshot=str(root), data=data, references=references, machines=machines, archive_machine='E',
                deadline=deadline, stack=[], recipes={ARM: RECIPE},
                source_hashes={str(p.relative_to(root)): sha256_file(p) for p in sorted(set(files))},
                frozen_at=datetime.now(timezone.utc).isoformat(), status={})
    core.set_phase(spec, 'cv', jobs())
    atomic_json(manifest, spec)
    print(manifest)


def finish(spec):
    root = Path(spec['snapshot'])
    curves, cohorts = {}, {}
    curves[ARM], cohorts[ARM] = reports(root, NAME, ARM)
    for arm in REFERENCE[1]:
        curves[arm], cohorts[arm] = reports(CAMPAIGNS / REFERENCE[0], REFERENCE[0], arm)
        if cohorts[arm] != cohorts[ARM]:
            raise ValueError(f'{arm} validation cohorts differ from {ARM}')
    anchor_epoch = core.cv_epoch(curves['anchor'])
    baseline = np.asarray([curve[anchor_epoch - 1] for curve in curves['anchor']])
    results = {}
    for arm in (ARM, *REFERENCE[1]):
        epoch = core.cv_epoch(curves[arm])
        scores = np.asarray([curve[epoch - 1] for curve in curves[arm]])
        delta = scores - baseline
        results[arm] = {'epoch': epoch, 'mean_score': float(scores.mean()), 'mean_delta': float(delta.mean()),
                        'seed_deltas': delta.reshape(5, 5).mean(1).tolist(), 'fold_deltas': delta.reshape(5, 5).mean(0).tolist()}
    out = root / 'outputs/evaluation' / NAME
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / 'summary.json', {'scope': 'Source patient CV only; epochs selected on these same folds; no target result',
                                      'folds': 5, 'seeds': 5, 'fits': 25, 'results': results})
    lines = ['# Class-aware light + multi-scale, five-fold, five-seed source CV', '',
             'Epochs come from each arm’s mean source-validation curve; deltas are against the matched anchor.', '',
             '| Arm | Epoch | Mean Score | Δ vs matched anchor | Seed deltas |', '| --- | ---: | ---: | ---: | --- |']
    for arm, row in results.items():
        lines.append(f"| {arm} | {row['epoch']} | {row['mean_score']:.3f} | {row['mean_delta']:+.3f} | "
                     + ', '.join(f'{x:+.2f}' for x in row['seed_deltas']) + ' |')
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    atomic_json(out / 'complete.json', {'status': 'complete', 'fits': 25, 'summary_sha256': sha256_file(out / 'summary.json')})
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
        finish(spec)
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
            if spec['phase'] == 'complete':
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
    parser.add_argument('--deadline', default=(datetime.now(timezone.utc) + timedelta(days=3)).isoformat())
    args = parser.parse_args()
    prepare(args.snapshot, args.hosts, args.data, args.references, args.deadline) if args.command == 'prepare' else run(args.manifest)
