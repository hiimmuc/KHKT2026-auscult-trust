"""Source-gated SWA, LoRA, multi-scale, class-aware and refined KD sweep."""
import argparse
import fcntl
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import improvement_campaign as core
from src.artifacts import atomic_json, sha256_file

NAME = 'next-improvements-20261026'
ARMS = ('lora', 'multiscale', 'swa', 'class_aware_light', 'lora_class_aware')
RECIPES = {'anchor': [], 'lora': ['lora'], 'multiscale': ['multiscale'], 'swa': ['swa'],
           'class_aware_light': ['class_aware_light'], 'lora_class_aware': ['lora_class_aware'], 'kd': []}


def source_jobs(arms, seed):
    jobs = core.cv_jobs(arms, [seed], name=NAME)
    for job in jobs:
        if job['arm'] == 'anchor':
            job['extra_args'] = ['--snapshot_epochs', '66']
    return jobs


def kd_jobs(spec, seed):
    teachers = {j['fold']: j for phase in spec['phases'].values() for j in phase['jobs']
                if j['arm'] == 'anchor' and j['seed'] == seed}
    jobs = core.cv_jobs(['kd'], [seed], name=NAME)
    for job in jobs:
        teacher = teachers[job['fold']]
        job['machine'] = teacher['machine']
        job['teacher_unit'] = {key: teacher[key] for key in ('run', 'cell', 'unit')}
    return jobs


def gate_results(spec, arms, seeds):
    anchor = core.source_reports(spec, 'anchor', seeds)
    results = {arm: core.gate(core.source_reports(spec, arm, seeds), anchor) for arm in arms}
    if 'lora_class_aware' in results:
        interaction = core.gate(core.source_reports(spec, 'lora_class_aware', seeds),
                                core.source_reports(spec, 'lora', seeds))
        results['lora_class_aware']['interaction_vs_lora'] = interaction
        results['lora_class_aware']['pass'] &= interaction['mean_delta'] >= 0.3 and sum(
            delta > 0 for delta in interaction['fold_deltas']) >= 2
    return results


def transition(spec):
    phase = spec['phase']
    if phase == 'screen':
        spec['screen_results'] = gate_results(spec, ARMS, [0])
        core.set_phase(spec, 'kd_screen', kd_jobs(spec, 0))
    elif phase == 'kd_screen':
        spec['screen_results']['kd'] = gate_results(spec, ['kd'], [0])['kd']
        passed = [arm for arm in (*ARMS, 'kd') if spec['screen_results'][arm]['pass']]
        spec['shortlist'] = sorted(passed, key=lambda arm: spec['screen_results'][arm]['mean_delta'], reverse=True)[:2]
        if spec['shortlist']:
            core.set_phase(spec, 'confirm', source_jobs(['anchor', *[a for a in spec['shortlist'] if a != 'kd']], 1))
        else:
            spec['phase'] = 'report'
    elif phase == 'confirm':
        if 'kd' in spec['shortlist']:
            core.set_phase(spec, 'kd_confirm', kd_jobs(spec, 1))
        else:
            select_final(spec)
    elif phase == 'kd_confirm':
        select_final(spec)
    elif phase == 'final':
        spec['phase'] = 'report'
    else:
        raise ValueError('Unsupported phase ' + phase)


def select_final(spec):
    spec['confirm_results'] = gate_results(spec, spec['shortlist'], [0, 1])
    passed = [(arm, result) for arm, result in spec['confirm_results'].items() if result['pass']]
    spec['selected'] = {arm: result['epoch'] for arm, result in sorted(
        passed, key=lambda item: item[1]['mean_delta'], reverse=True)[:1]}
    if spec['selected']:
        core.refresh_capacity(spec)
        core.set_phase(spec, 'final', core.final_jobs(spec['selected'], name=NAME))
    else:
        spec['phase'] = 'report'


def finish(spec):
    root = Path(spec['snapshot'])
    destination = root / 'outputs/evaluation' / NAME
    destination.mkdir(parents=True, exist_ok=True)
    report = {'source_screen': spec['screen_results'], 'source_confirm': spec.get('confirm_results'),
              'selected': spec.get('selected', {}), 'final': [], 'scope':
              'Exploratory source-selected recipe comparison; target labels never select models or epochs.'}
    for job in spec.get('phases', {}).get('final', {}).get('jobs', []):
        row = core.load(core.unit_path(spec, job) / 'report.json')
        score = row['fixed'][str(job['epoch'])]['test']['all']
        report['final'].append({'arm': job['arm'], 'cohort': job['cohort'], 'seed': job['seed'],
                                'epoch': job['epoch'], **{k: score[k] for k in ('sp', 'se', 'score', 'hs')}})
    atomic_json(destination / 'summary.json', report)
    lines = ['# Next source-gated improvement campaign', '', '| Arm | Source screen ΔScore | Confirm ΔScore | Selected |',
             '| --- | ---: | ---: | --- |']
    for arm, item in report['source_screen'].items():
        confirmed = (report['source_confirm'] or {}).get(arm)
        lines.append(f"| {arm} | {item['mean_delta']:+.3f} | {confirmed['mean_delta'] if confirmed else float('nan'):+.3f} | {arm in report['selected']} |")
    (destination / 'report.md').write_text('\n'.join(lines) + '\n')
    atomic_json(destination / 'complete.json', {'status': 'complete', 'summary_sha256': sha256_file(destination / 'summary.json'),
                                                'units': sum(len(x['jobs']) for x in spec['phases'].values())})
    spec['phase'] = 'complete'


def tick(spec, manifest, now=None):
    core.verify_source(spec)
    if spec['phase'] == 'complete':
        return
    if spec['phase'] == 'report':
        finish(spec)
    else:
        core.collect(spec)
        completed = [core.verify_job(spec, job) for job in spec['jobs']]
        spec['status'] = {'complete': sum(completed), 'units': len(completed), 'phase': spec['phase']}
        deadline = (now or datetime.now(timezone.utc)) >= datetime.fromisoformat(spec['deadline'])
        if all(completed):
            transition(spec)
            atomic_json(manifest, spec)
        if spec['phase'] not in ('complete', 'report') and not deadline:
            core.launch(spec, manifest)
        elif deadline:
            spec['status']['admission_closed'] = True
    atomic_json(manifest, spec)


def run(path):
    path = Path(path).resolve()
    with (path.parent / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            spec = core.load(path)
            tick(spec, path)
            if spec['phase'] == 'complete':
                return
            time.sleep(45)


def prepare(snapshot, hosts, data, references):
    root = Path(snapshot).resolve()
    manifest = root / 'logs' / NAME / 'programme.json'
    if manifest.exists():
        raise FileExistsError(manifest)
    machines = core.load(hosts)
    if set(machines) != {'B', 'D', 'E', 'F'} or not machines['B'].get('local'):
        raise ValueError('Only B/D/E/F; B controls campaign')
    files = [p for folder in ('src', 'scripts', 'experiments') for p in (root / folder).rglob('*')
             if p.is_file() and p.suffix in ('.py', '.yaml', '.json', '.sh') and '__pycache__' not in p.parts]
    files += [root / 'requirements-train.lock', root / 'pyproject.toml']
    spec = dict(name=NAME, snapshot=str(root), data=data, references=references, machines=machines,
                archive_machine='E', deadline='2026-10-26T23:00:00+07:00', stack=[], recipes=RECIPES,
                source_hashes={str(p.relative_to(root)): sha256_file(p) for p in sorted(set(files))},
                frozen_at=datetime.now(timezone.utc).isoformat(), status={})
    core.set_phase(spec, 'screen', source_jobs(['anchor', *ARMS], 0))
    atomic_json(manifest, spec)
    print(manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'run'))
    parser.add_argument('manifest', nargs='?')
    for key in ('snapshot', 'hosts', 'data', 'references'):
        parser.add_argument('--' + key)
    args = parser.parse_args()
    prepare(args.snapshot, args.hosts, args.data, args.references) if args.command == 'prepare' else run(args.manifest)
