"""Freeze, start and inspect isolated training campaigns; install a non-overlapping hourly tracker."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.artifacts import atomic_json, sha256_file, verify_file


def load(path):
    return json.loads(Path(path).read_text())


def identities(path):
    result = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        cell, seed = re.search(r'--cell (\S+)', line), re.search(r'--seed (\d+)', line)
        run = re.search(r'RUN_ID=(\S+)', line)
        if not all((cell, seed, run)):
            raise ValueError(f'Invalid training job: {line}')
        folds, fold = re.search(r'--cv_folds (\d+)', line), re.search(r'--cv_fold (\d+)', line)
        unit = 'cv' + (fold[1] if fold else '0') if folds and int(folds[1]) > 0 else 'seed' + seed[1]
        result.append((run[1], cell[1], unit))
    return result


def freeze(config, out):
    spec = load(config)
    root = Path(spec['source'])
    files = [p for folder in ('src', 'scripts', 'experiments', 'tests') for p in (root / folder).rglob('*')
             if p.is_file() and p.suffix in ('.py', '.yaml', '.json', '.sh') and '__pycache__' not in p.parts]
    files += [root / name for name in ('requirements-train.lock', 'requirements-references.lock.json', 'pyproject.toml')]
    jobs = [job for machine in spec['machines'].values() for job in identities(root / machine['jobs'])]
    if len(jobs) != len(set(jobs)) or len(jobs) != spec['expected_units']:
        raise ValueError('Duplicate jobs or unexpected unit count')
    files += [root / m['jobs'] for m in spec['machines'].values()]
    spec['source_hashes'] = {str(p.relative_to(root)): sha256_file(p) for p in sorted(set(files))}
    spec['frozen_at'] = datetime.now(timezone.utc).isoformat()
    if Path(out).exists():
        raise FileExistsError(out)
    atomic_json(out, spec)
    print(f'Frozen {len(jobs)} unique units and {len(spec["source_hashes"])} source/config files: {out}')


def alive(pidfile):
    try:
        record = load(pidfile)
        pid = record['pid']
        os.kill(pid, 0)
        command = (Path('/proc') / str(pid) / 'cmdline').read_bytes().decode().strip('\0').split('\0')
        if command != record['command']:
            return None
        return pid
    except (OSError, ValueError, KeyError):
        return None


def start(spec, machine):
    root = Path(spec['snapshot'])
    os.chdir(root)
    for relative, expected in spec['source_hashes'].items():
        verify_file(root / relative, expected)
    logdir = root / 'logs' / spec['name']
    logdir.mkdir(parents=True, exist_ok=True)
    with (logdir / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pidfile = logdir / f'{machine}.pid.json'
        if alive(pidfile):
            raise RuntimeError('Scheduler already running')
        m = spec['machines'][machine]
        for run, cell, seed in identities(root / m['jobs']):
            out, ck = root / 'outputs/train' / run / cell / seed, root / 'checkpoints/train' / run / cell / seed
            if out.exists() or ck.exists():
                raise ValueError(f'Existing unit must be verified or set aside explicitly: {out}')
        env = {**os.environ, 'AT_DATA': spec['data'], 'AT_REPOS': spec['references'],
               'RUN_ID': spec['run'], 'SWEEP_TAG': '-' + machine, 'PYTHONPATH': str(root)}
        cmd = [str(root / '.venv-train/bin/python'), '-u', 'scripts/sweep.py', m['jobs'],
               '--workers', str(m['workers']), '--gpu-gb', str(m.get('gpu_gb', 28)),
               '--mem-gb', str(m.get('mem_gb', 8)), '--ram-gb', '20', '--disk-gb', '30',
               '--threads', '3', '--num-workers', '2', '--ramp', str(m.get('ramp', 90)),
               '--retries', '0', '--deadline', spec['deadline']]
        with (logdir / f'{machine}.scheduler.log').open('ab') as log:
            p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        atomic_json(pidfile, {'pid': p.pid, 'command': cmd, 'started': datetime.now(timezone.utc).isoformat()})
        print(json.dumps({'machine': machine, 'pid': p.pid, 'jobs': len(identities(root / m['jobs']))}))


def inspect(spec, machine):
    root = Path(spec['snapshot'])
    os.chdir(root)
    m = spec['machines'][machine]
    jobs = identities(root / m['jobs'])
    counts = {'complete': 0, 'partial': 0, 'pending': 0, 'invalid': 0}
    for run, cell, seed in jobs:
        out = root / 'outputs/train' / run / cell / seed
        marker = out / 'completion.json'
        if marker.exists():
            try:
                verify_file(out / 'report.json', load(marker)['files']['report.json'])
                counts['complete' if not spec.get('pipeline') or (out / 'pipeline-complete.json').exists() else 'partial'] += 1
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                counts['invalid'] += 1
        else:
            counts['partial' if out.exists() else 'pending'] += 1
    logdir = root / 'logs' / spec['name']
    scheduler = alive(logdir / f'{machine}.pid.json')
    logfile = logdir / f'{machine}.scheduler.log'
    tail = logfile.read_text(errors='replace')[-16000:] if logfile.exists() else ''
    errors = [line for line in tail.splitlines() if any(t in line for t in ('FAILED', 'Traceback', 'out of memory'))]
    joblogs = root / 'logs' / (spec['run'] + '-' + machine)
    nonfinite = []
    for p in joblogs.glob('*.log'):
        with p.open('rb') as stream:
            stream.seek(max(0, p.stat().st_size - 8192))
            text = stream.read().decode(errors='replace')
        if re.search(r'(?:loss|Acc@1)\s+(?:nan|inf)\b', text, re.I) or 'Traceback' in text:
            nonfinite.append(p.name)
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, timeout=10)
    alerts = errors + [f'Failed/nonfinite job log: {p}' for p in nonfinite]
    if counts['invalid']:
        alerts.append('Invalid completion record')
    if not scheduler and counts['complete'] != len(jobs):
        alerts.append('Scheduler absent with unfinished jobs (check deadline/failure)')
    free = round(shutil.disk_usage(root).free / 2**30, 1)
    if free < 30:
        alerts.append('Disk below admission reserve')
    return {'machine': machine, 'units': len(jobs), **counts, 'scheduler_pid': scheduler,
            'disk_free_gib': free, 'gpu_free_mib_util_percent': gpu.stdout.strip(), 'alerts': alerts}


def track(spec):
    def one(item):
        machine, m = item
        command = shlex.join([str(Path(spec['snapshot']) / '.venv-train/bin/python'),
                              str(Path(spec['snapshot']) / 'scripts/campaign.py'), 'inspect',
                              str(Path(spec['snapshot']) / 'campaign.json'), '--machine', machine])
        try:
            args = shlex.split(command) if m.get('local') else [
                'ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', m['host'], command
            ]
            p = subprocess.run(args,
                               capture_output=True, text=True, timeout=60, check=True)
            return json.loads(p.stdout)
        except (subprocess.SubprocessError, ValueError) as error:
            return {'machine': machine, 'alerts': ['Inspection unavailable: ' + type(error).__name__]}
    rows = list(ThreadPoolExecutor(max_workers=3).map(one, spec['machines'].items()))
    record = {'at': datetime.now(timezone.utc).isoformat(), 'campaign': spec['name'], 'machines': rows}
    logdir = Path(spec['source']) / 'logs' / spec['name']
    local_file = logdir / 'local-processes.json'
    local = []
    for entry in load(local_file) if local_file.exists() else []:
        summary = Path(entry['out']) / 'summary.json'
        pidfile = logdir / (entry['name'] + '.pid.json')
        running = alive(pidfile)
        local.append({'name': entry['name'], 'running_pid': running, 'summary': str(summary),
                      'status': 'complete' if summary.exists() else 'running' if running else 'stopped_without_summary'})
    record['local_analyses'] = local
    if spec.get('collect_outputs'):
        collected = []
        destination = Path(spec['source']) / 'outputs/train' / spec['run']
        destination.mkdir(parents=True, exist_ok=True)
        for machine, m in spec['machines'].items():
            if m.get('local'):
                collected.append({'machine': machine, 'exit_code': 0})
                continue
            remote = m['host'] + ':' + str(Path(spec['snapshot']) / 'outputs/train' / spec['run']) + '/'
            command = ['rsync', '-a', '--ignore-existing', '--include=*/', '--include=*.json', '--include=*.npz',
                       '--exclude=*', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=8', remote, str(destination) + '/']
            result = subprocess.run(command, capture_output=True, text=True, timeout=180)
            collected.append({'machine': machine, 'exit_code': result.returncode})
        record['output_collection'] = collected
    post = spec.get('after_complete')
    collected_ok = all(r['exit_code'] == 0 for r in record.get('output_collection', []))
    if post and rows and collected_ok and all(r.get('complete') == r.get('units') and r.get('units', 0) > 0 for r in rows):
        pidfile = logdir / 'postprocess.pid.json'
        if not pidfile.exists():
            with (logdir / 'postprocess.log').open('ab') as log:
                p = subprocess.Popen(post['command'], cwd=spec['source'], stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True,
                                     env={**os.environ, 'OMP_NUM_THREADS': '3', 'OPENBLAS_NUM_THREADS': '3'})
            atomic_json(pidfile, {'pid': p.pid, 'command': post['command']})
        record['postprocess'] = {'pid': alive(pidfile), 'output': post['output'],
                                 'complete': Path(post['output']).exists()}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    atomic_json(logdir / 'status' / f'{stamp}.json', record)
    atomic_json(logdir / 'latest.json', record)
    print(json.dumps(record, indent=2), flush=True)


def cron(spec, manifest):
    root = Path(spec['source'])
    logdir = root / 'logs' / spec['name']
    logdir.mkdir(parents=True, exist_ok=True)
    marker = '# auscult-campaign-' + spec['name']
    p = subprocess.run(['crontab', '-l'], text=True, capture_output=True)
    if p.returncode not in (0, 1) or (p.returncode == 1 and 'no crontab' not in p.stderr):
        raise RuntimeError('Could not read crontab safely')
    previous = p.stdout
    atomic_json(logdir / 'crontab-backup.json', {'previous': previous})
    lines = [line for line in previous.splitlines() if not line.endswith(marker)]
    command = shlex.join(['/usr/bin/flock', '-n', str(logdir / 'track.lock'),
                          str(root / '.venv-train/bin/python'), str(root / 'scripts/campaign.py'), 'track', str(Path(manifest).resolve())])
    entry = f'0 * * * * {command} >> {shlex.quote(str(logdir / "hourly.log"))} 2>&1 {marker}'
    if '%' in entry or '\n' in entry:
        raise ValueError('Unsupported character in cron path')
    subprocess.run(['crontab', '-'], input='\n'.join([*lines, entry]) + '\n', text=True, check=True)
    print(entry)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'start', 'inspect', 'track', 'cron'))
    parser.add_argument('manifest')
    parser.add_argument('--out')
    parser.add_argument('--machine')
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze(args.manifest, args.out)
    else:
        spec = load(args.manifest)
        if args.command == 'start':
            start(spec, args.machine)
        elif args.command == 'inspect':
            print(json.dumps(inspect(spec, args.machine)))
        elif args.command == 'track':
            track(spec)
        else:
            cron(spec, args.manifest)


if __name__ == '__main__':
    main()
