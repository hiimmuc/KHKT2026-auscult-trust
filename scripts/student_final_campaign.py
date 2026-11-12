"""Training for the frozen evaluation of the distilled student `kd_g4_k1` (campaign `student-final-20261029`).

Two arms on the reference recipe at epoch 66, soft targets from the G4 teacher group (K1: soft targets only):
- `kd_g4_k1`: the student with trained static SC, LODO seeds 3 and 4 only (seeds 0-2 and the official split exist in `distill-20261028`);
- `kd_g4_k1_nosc`: its twin with spectrum correction off (everything else identical), official split and four held-out devices, seeds 0-4.

Cohorts are placed whole on one machine so that the G4 member units and checkpoints (only on B) are copied once (`sync`).
Subcommands: `prepare` (jobs, placement, manifest), `sync` (copy the members to the other machines; run on B), `run` (controller; run on B).
"""
import argparse
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts import distill_campaign as dc
from scripts import improvement_campaign as core
from auscult_trust.artifacts import atomic_json, sha256_file

NAME = 'student-final-20261029'
dc.NAME = NAME
dc.CONFIGS.clear()
dc.CONFIGS.update({'kd_g4_k1': ('g4', 'k1'), 'kd_g4_k1_nosc': ('g4', 'k1')})
RECIPES = {'kd_g4_k1': [], 'kd_g4_k1_nosc': ['student_nosc']}
WEIGHTS = {'A': 0.5, 'B': 1.0, 'C': 0.8, 'D': 1.0, 'E': 1.0, 'F': 0.8}
GPU = {'F': 30}  # user cap on the shared machine F
LODO_NEW = (3, 4)


def jobs():
    out = dc.student_jobs(['kd_g4_k1'], set(core.DEVICES), set(LODO_NEW))
    out += dc.student_jobs(['kd_g4_k1_nosc'], {'official', *core.DEVICES}, set(range(5)))
    return out


def place(all_jobs):
    """Whole cohorts (cohort, seed) per machine, longest first onto the machine that would end first (weighted LPT)."""
    cohorts = {}
    for job in all_jobs:
        cohorts.setdefault((job['cohort'], job['seed']), []).append(job)
    load = dict.fromkeys(WEIGHTS, 0.0)
    for key, group in sorted(cohorts.items(), key=lambda kv: -len(kv[1])):
        cost = sum(dc.cost(j) for j in group)
        machine = min(load, key=lambda m: (load[m] + cost) / WEIGHTS[m])
        load[machine] += cost
        for job in group:
            job['machine'] = machine
    return load


def prepare(snapshot, hosts, data, deadline):
    root = Path(snapshot).resolve()
    manifest = root / 'logs' / NAME / 'programme.json'
    if manifest.exists():
        raise FileExistsError(manifest)
    machines = core.load(hosts)
    if set(machines) != set(WEIGHTS) or not machines['B'].get('local'):
        raise ValueError('B must control A/B/C/D/E/F')
    core.GPU_BUDGETS.update(GPU)
    files = [p for folder in ('auscult_trust', 'scripts', 'experiments') for p in (root / folder).rglob('*')
             if p.is_file() and p.suffix in ('.py', '.yaml', '.json', '.sh') and '__pycache__' not in p.parts]
    files += [root / 'requirements-train.lock', root / 'pyproject.toml']
    spec = dict(name=NAME, snapshot=str(root), data=data, machines=machines, archive_machine='E',
                deadline=deadline, stack=[], reference_snapshot=str(dc.CAMPAIGNS / dc.REFERENCE[0]), recipes=RECIPES,
                source_hashes={str(p.relative_to(root)): sha256_file(p) for p in sorted(set(files))},
                frozen_at=datetime.now(timezone.utc).isoformat(), status={})
    all_jobs = jobs()
    load = place(all_jobs)
    dc.set_phase(spec, 'train', all_jobs)
    spec['placement'] = {m: round(v, 2) for m, v in load.items()}
    atomic_json(manifest, spec)
    print(manifest, spec['placement'], len(all_jobs), 'jobs')


def sync(manifest):
    """Copy the G4 member units (outputs) and checkpoints each non-B machine needs, from B to the same absolute paths."""
    spec = core.load(manifest)
    wanted = {}
    for job in spec['jobs']:
        if spec['machines'][job['machine']].get('local'):
            continue
        units, checkpoints = dc.teacher_files('g4', job['cohort'], job['seed'])
        wanted.setdefault(job['machine'], set()).update(map(str, units), map(str, checkpoints))
    procs = []
    for machine, paths in wanted.items():
        listing = Path(manifest).parent / f'members-{machine}.txt'
        listing.write_text(''.join(p.lstrip('/') + '\n' for p in sorted(paths)))
        host = spec['machines'][machine]['host']
        parents = sorted({str(Path(p).parent) for p in paths})
        subprocess.run(['ssh', '-o', 'BatchMode=yes', host, 'mkdir', '-p', *parents], check=True)
        procs.append((machine, subprocess.Popen(['rsync', '-a', '-r', '--files-from', str(listing), '/', f'{host}:/'])))
    for machine, proc in procs:
        print(machine, 'rsync exit', proc.wait(), flush=True)


def tick(spec, manifest):
    core.verify_source(spec)
    if spec['phase'] == 'complete':
        return
    core.collect(spec)
    complete = [core.verify_job(spec, job) for job in spec['jobs']]
    spec['status'] = {'at': datetime.now(timezone.utc).isoformat(), 'complete': sum(complete), 'units': len(complete), 'phase': spec['phase']}
    if all(complete):
        spec['phase'] = 'complete'
        out = Path(spec['snapshot']) / 'outputs/evaluation' / NAME
        out.mkdir(parents=True, exist_ok=True)
        atomic_json(out / 'complete.json', {'status': 'complete', 'fits': len(complete)})
    elif datetime.now(timezone.utc) < datetime.fromisoformat(spec['deadline']):
        dc_launch = core.launch
        dc_launch(spec, manifest)
    else:
        spec['status']['admission_closed'] = True
    atomic_json(manifest, spec)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'sync', 'run'))
    parser.add_argument('manifest', nargs='?')
    parser.add_argument('--snapshot')
    parser.add_argument('--hosts')
    parser.add_argument('--data')
    parser.add_argument('--deadline', default=(datetime.now(timezone.utc) + timedelta(days=2)).isoformat())
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.snapshot, args.hosts, args.data, args.deadline)
    elif args.command == 'sync':
        sync(args.manifest)
    else:
        core.run_controller(args.manifest, tick)
