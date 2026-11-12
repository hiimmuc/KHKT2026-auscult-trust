"""Source-gated auxiliary events and fixed epoch-66 anchor follow-up on B/D/E/F."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import improvement_campaign as core
from src.artifacts import atomic_json, sha256_file

NAME = 'followup-20261026'


def recovery_verified(path):
    """Verify native E recovery completeness; historical training failures are evidence."""
    path = Path(path)
    if not path.is_file():
        return False
    report = core.load(path)
    manifest = path.parent / 'metadata/source-manifest.json'
    if report.get('status') != 'verified' or not manifest.is_file():
        return False
    source = core.load(manifest)
    integrity = report.get('unit_receipt_integrity', {})
    comparisons = report.get('canonical_final_comparisons', [])
    return bool(report.get('verified_files', 0) > 0 and report.get('verified_bytes', 0) > 0 and
        report['verified_files'] == source.get('file_count') and report['verified_bytes'] == source.get('bytes') and
        integrity.get('units', 0) > 0 and integrity.get('verified') == integrity.get('units') and
        integrity.get('mismatches') == [] and report.get('unmanifested_recovered_files') == [] and
        report.get('canonical_matches') == len(comparisons) == 10 and
        all(row.get('matches_canonical') is True for row in comparisons) and not report.get('errors'))


def initial_jobs():
    jobs = core.cv_jobs(['event_heads'], [0], name=NAME)
    for job in core.final_jobs({'anchor': 66}, name=NAME):
        job['cell'] = 'gain6-cv66' + ('' if job['cohort'] == 'official' else '+lodo_' + job['cohort'].lower())
        jobs.append(job)
    return jobs


def transition(spec):
    """Only source validation curves decide whether auxiliary events advance."""
    phase = spec['phase']
    if phase in ('screen', 'confirm'):
        seeds = [0] if phase == 'screen' else [0, 1]
        result = core.gate(core.source_reports(spec, 'event_heads', seeds), core.source_reports(spec, 'anchor', seeds))
        spec[phase + '_results'] = {'event_heads': result}
        if result['pass']:
            if phase == 'screen':
                core.set_phase(spec, 'confirm', core.cv_jobs(['event_heads'], [1], name=NAME))
            else:
                spec['selected']['event_heads'] = result['epoch']
                core.set_phase(spec, 'aux_final', core.final_jobs({'event_heads': result['epoch']}, name=NAME))
            return
    elif phase != 'aux_final':
        raise ValueError('No follow-up transition from ' + phase)
    spec['jobs'] = [job for phase in spec['phases'].values() for job in phase['jobs'] if 'cohort' in job]
    spec['phase'] = 'report'


def finish(spec):
    """Model reports are a subtask; missing ensemble inputs keep the programme pending."""
    root = Path(spec['snapshot'])
    destination = root / 'outputs/evaluation' / NAME
    if not (destination / 'complete.json').is_file():
        for label, mapping in spec['baselines'].items():
            for seeds in mapping.values():
                for path in seeds.values():
                    if core.load(Path(path) / 'pipeline-complete.json')['report_epoch'] != {'control': 21, 'gain': 67, 'llrd': 21}[label]:
                        raise ValueError('Baseline epoch differs from its fixed source recipe')
        core.report(spec)
    core.verify_file(destination / 'summary.json', core.load(destination / 'complete.json')['summary_sha256'])
    ensemble = destination / 'ensembles'
    try:
        receipt = core.load(ensemble / 'receipt.json')
    except (FileNotFoundError, json.JSONDecodeError):
        receipt = {}
    if receipt.get('status') != 'complete' or receipt.get('pending'):
        if ensemble.exists():
            attempt = 1
            while (destination / f'ensembles-attempt-{attempt:03d}').exists():
                attempt += 1
            ensemble.rename(destination / f'ensembles-attempt-{attempt:03d}')
        subprocess.run([str(root / '.venv-train/bin/python'), '-m', 'scripts.followup_ensembles', 'evaluate',
                        spec['ensemble_manifest'], '--out', str(ensemble), '--boot', '1000'], cwd=root, check=True)
        receipt = core.load(ensemble / 'receipt.json')
    if receipt.get('status') != 'complete' or receipt.get('pending'):
        spec['status'] = {'waiting_for_ensembles': receipt.get('pending', []), 'model_reports_complete': True}
        atomic_json(destination / 'pending-work.json', spec['status'])
        return False
    inputs = receipt.get('inputs', {})
    ensemble_manifest = str(Path(spec['ensemble_manifest']).resolve())
    if inputs.get(ensemble_manifest) != spec.get('ensemble_manifest_sha256', sha256_file(ensemble_manifest)):
        raise ValueError('Complete ensemble receipt does not bind the frozen input manifest')
    for path, digest in inputs.items():
        core.verify_file(path, digest)
    recovery = Path(spec['recovery_report'])
    if not recovery_verified(recovery):
        raise ValueError('Native E recovery verification changed before completion')
    atomic_json(destination / 'followup-complete.json', {'status': 'complete', 'files': {
        str(path): sha256_file(path) for path in (destination / 'summary.json', ensemble / 'receipt.json', recovery)}})
    spec['phase'] = 'complete'
    return True


def tick(spec, manifest, now=None):
    core.verify_source(spec)
    if spec.get('ensemble_manifest_sha256'):
        core.verify_file(spec['ensemble_manifest'], spec['ensemble_manifest_sha256'])
    for reuse in spec.get('source_anchor_reuse', []):
        for path, digest in reuse['files'].items():
            core.verify_file(path, digest)
    if spec['phase'] == 'complete':
        return
    if not recovery_verified(spec['recovery_report']):
        spec['status'] = {'waiting_for_native_e_recovery': spec['recovery_report']}
        atomic_json(manifest, spec)
        return
    now = now or datetime.now(timezone.utc)
    deadline = now >= datetime.fromisoformat(spec['deadline'])
    if spec['phase'] == 'report':
        try:
            finish(spec)
        except (FileNotFoundError, subprocess.CalledProcessError) as error:
            spec['status'] = {'waiting_for_report_inputs': str(error)}
            atomic_json(Path(spec['snapshot']) / 'outputs/evaluation' / NAME / 'pending-work.json', spec['status'])
    else:
        core.collect(spec)
        complete = [core.verify_job(spec, job) for job in spec['jobs']]
        spec['status'] = {'complete': sum(complete), 'units': len(complete), 'admission_closed': deadline}
        if all(complete):
            transition(spec)
            atomic_json(manifest, spec)
        if spec['phase'] != 'report' and not deadline:
            core.launch(spec, manifest)
    atomic_json(manifest, spec)


def prepare(args):
    root = Path(args.snapshot).resolve()
    manifest = root / 'logs' / NAME / 'programme.json'
    if manifest.exists():
        raise FileExistsError(manifest)
    if datetime.fromisoformat(args.deadline).tzinfo is None:
        raise ValueError('Deadline requires a timezone')
    machines = core.load(args.hosts)
    if set(machines) != {'B', 'D', 'E', 'F'} or not machines['B'].get('local') or any(machines[m].get('local') for m in ('D', 'E', 'F')):
        raise ValueError('Follow-up requires B local and D/E/F remote; A receives no jobs')
    previous = core.load(args.previous_manifest)
    if previous.get('name', core.NAME) != core.NAME or not core.audits_complete(previous):
        raise ValueError('Prior programme needs all three certified source audits')
    baselines = core.load(args.baselines)
    if set(baselines) != {'control', 'gain', 'llrd'}:
        raise ValueError('Fixed control21/gain67/llrd21 baseline mappings are required')
    for mapping in baselines.values():
        if set(mapping) != {'official', *core.DEVICES} or any(set(rows) != {str(s) for s in range(5)} for rows in mapping.values()):
            raise ValueError('Every baseline needs five seeds on every fixed cohort')
        if any(not Path(path).is_absolute() for rows in mapping.values() for path in rows.values()):
            raise ValueError('Baseline paths must be absolute')
    reuse = []
    for seed in (0, 1):
        source = Path(previous['snapshot']) / 'outputs/train' / f'{core.NAME}-s{seed}' / 'anchor'
        for fold in range(3):
            for name in ('report.json', 'val_preds.npz'):
                if not (source / f'cv{fold}' / name).is_file():
                    raise FileNotFoundError(source / f'cv{fold}' / name)
        alias = root / 'outputs/train' / f'{NAME}-s{seed}' / 'anchor'
        alias.parent.mkdir(parents=True, exist_ok=True)
        if not alias.exists():
            alias.symlink_to(source, target_is_directory=True)
        elif alias.resolve() != source.resolve():
            raise ValueError('Source anchor alias points to a different programme')
        reuse.append({'source': str(source), 'alias': str(alias), 'files': {
            str(path): sha256_file(path) for path in source.glob('cv*/*') if path.name in ('report.json', 'val_preds.npz')}})
    files = [path for folder in ('src', 'scripts', 'experiments', 'tests') for path in (root / folder).rglob('*')
             if path.is_file() and path.suffix in ('.py', '.yaml', '.json', '.sh') and '__pycache__' not in path.parts]
    files += [root / name for name in ('requirements-train.lock', 'third_party/SOURCES.json', 'pyproject.toml')]
    spec = dict(name=NAME, snapshot=str(root), machines=machines, archive_machine='E', data=args.data,
        deadline=args.deadline, recovery_report=str(Path(args.recovery_report).resolve()),
        ensemble_manifest=str(Path(args.ensemble_manifest).resolve()), ensemble_manifest_sha256=sha256_file(args.ensemble_manifest),
        baselines=baselines, stack=[], selected={'anchor': 66},
        audits=previous['audits'], source_anchor_reuse=reuse,
        source_hashes={str(path.relative_to(root)): sha256_file(path) for path in sorted(set(files))},
        scope_amendments=[{'scope': 'event_heads source-only hypothesis; fixed anchor epoch66; no target-selected recipe',
                           'source_anchor_reuse': 'event_weight defaults0 and cannot affect existing PAFA objective'}],
        frozen_at=datetime.now(timezone.utc).isoformat())
    core.set_phase(spec, 'screen', initial_jobs())
    atomic_json(manifest, spec)
    print(manifest)


def run(manifest, once=False):
    manifest = Path(manifest).resolve()
    with (manifest.parent / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            spec = core.load(manifest)
            tick(spec, manifest)
            if once or spec['phase'] == 'complete':
                return
            time.sleep(60 if spec['phase'] == 'report' else 45)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run'])
    parser.add_argument('manifest', nargs='?')
    for name in ('snapshot', 'data', 'hosts', 'previous-manifest', 'baselines', 'recovery-report', 'ensemble-manifest'):
        parser.add_argument('--' + name)
    parser.add_argument('--deadline', default='2026-10-26T23:00:00+07:00')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    prepare(args) if args.command == 'prepare' else run(args.manifest, args.once)


if __name__ == '__main__':
    main()
