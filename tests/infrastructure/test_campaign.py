import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location('campaign', 'scripts/campaign.py')
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


def test_job_identity_includes_run_cell_and_seed(tmp_path):
    jobs = tmp_path / 'jobs.txt'
    jobs.write_text('RUN_ID=new MEM_GB=10 python -m src.training.main --cell ast+ft+pafa+sc --seed 12\n')
    assert campaign.identities(jobs) == [('new', 'ast+ft+pafa+sc', 'seed12')]
    jobs.write_text('python broken\n')
    with pytest.raises(ValueError, match='Invalid training job'):
        campaign.identities(jobs)


def test_cron_preserves_unrelated_entries_and_is_idempotent(tmp_path, monkeypatch):
    current = ['15 4 * * * /existing/job # keep\n']
    def run(args, **kwargs):
        if args == ['crontab', '-l']:
            return SimpleNamespace(returncode=0, stdout=current[0], stderr='')
        assert args == ['crontab', '-']
        current[0] = kwargs['input']
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(campaign.subprocess, 'run', run)
    manifest = tmp_path / 'manifest.json'
    spec = {'source': str(tmp_path), 'name': 'test'}
    campaign.cron(spec, manifest)
    campaign.cron(spec, manifest)
    assert current[0].startswith('15 4 * * * /existing/job # keep\n')
    assert current[0].count('0 * * * * ') == 1


def test_pid_reuse_is_not_a_live_scheduler(tmp_path, monkeypatch):
    pidfile = tmp_path / 'pid.json'
    pidfile.write_text(json.dumps({'pid': 999999999, 'command': ['unrelated']}))
    assert campaign.alive(pidfile) is None


def test_campaign_assignments_are_unique():
    config = json.loads(Path('experiments/campaigns/report-20261024.json').read_text())
    # Generated lists are runtime artifacts, not required for clean-checkout CI.
    paths = [Path(m['jobs']) for m in config['machines'].values()]
    if not all(p.exists() for p in paths):
        pytest.skip('campaign lists not generated')
    ids = [u for p in paths for u in campaign.identities(p)]
    assert len(ids) == len(set(ids)) == config['expected_units']


def test_freeze_keeps_same_seed_cv_folds_as_distinct_units(tmp_path):
    jobs = tmp_path / 'jobs.txt'
    jobs.write_text(''.join(
        f'RUN_ID=cv python -m src.training.main --cell {cell} --seed 0 --cv_folds 3 --cv_fold {fold}\n'
        for cell in ('pafa', 'pafa+warm5') for fold in range(3)))
    assert campaign.identities(jobs) == [
        ('cv', cell, f'cv{fold}') for cell in ('pafa', 'pafa+warm5') for fold in range(3)]
    for name in ('requirements-train.lock', 'requirements-references.lock.json', 'pyproject.toml'):
        (tmp_path / name).write_text('')
    config, manifest = tmp_path / 'config.json', tmp_path / 'frozen.json'
    config.write_text(json.dumps({'source': str(tmp_path), 'expected_units': 6,
                                 'machines': {'A': {'jobs': 'jobs.txt'}}}))
    campaign.freeze(config, manifest)
    frozen = json.loads(manifest.read_text())
    assert frozen['expected_units'] == 6 and 'jobs.txt' in frozen['source_hashes']


def test_cv_identity_uses_default_fold_and_zero_folds_stays_seed_unit(tmp_path):
    jobs = tmp_path / 'jobs.txt'
    jobs.write_text('RUN_ID=cv python -m src.training.main --cell pafa --seed 2 --cv_folds 3\n'
                    'RUN_ID=cv python -m src.training.main --cell pafa --seed 2 --cv_folds 0 --cv_fold 1\n')
    assert campaign.identities(jobs) == [('cv', 'pafa', 'cv0'), ('cv', 'pafa', 'seed2')]
