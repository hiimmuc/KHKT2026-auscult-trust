import importlib.util

spec = importlib.util.spec_from_file_location("sweep", "scripts/sweep.py")
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


def test_reservations_sum_running_units_only():
    jobs = ["RUN_ID=g MEM_GB=10 python -m auscult_trust.training.main --cell ast+ft+ce --seed 1",
            "RUN_ID=g MEM_GB=13 python -m auscult_trust.training.main --cell ast+ft+ce --seed 12",
            "RUN_ID=g python -m auscult_trust.training.main --cell ast+ft+pafa --seed 0"]
    ps = ("python -m auscult_trust.training.main --cell ast+ft+ce --seed 1 --num_workers 4\n"
          "python -m auscult_trust.training.main --cell ast+ft+pafa --seed 0 --num_workers 4\n")
    assert sum(sweep.mem_gb(c, 8.0) for c in jobs if sweep.running_elsewhere(c, ps)) == 18.0  # seed 12 is not seed 1


def test_saved_job_lists_and_live_legacy_processes_are_supported():
    old = "python -m auscult_trust.training.patchmix_cl.main --cell beats+ft+pafa --seed 0"
    current = sweep.training_command(old)
    assert current == "python -m auscult_trust.training.main --cell beats+ft+pafa --seed 0"
    assert sweep.training_command(current) == current
    assert sweep.running_elsewhere(current, old)


def test_admission_stops_at_deadline_or_low_disk(monkeypatch):
    from datetime import datetime, timezone
    from types import SimpleNamespace
    monkeypatch.setattr(sweep.shutil, 'disk_usage', lambda _: SimpleNamespace(free=40 * 1024**3))
    deadline = '2026-10-24T18:00:00+07:00'
    assert sweep.admission_open(deadline, 30, datetime(2026, 10, 24, 10, tzinfo=timezone.utc))
    assert not sweep.admission_open(deadline, 30, datetime(2026, 10, 24, 11, tzinfo=timezone.utc))
    assert not sweep.admission_open(None, 50)


def test_pipeline_waits_for_exports_and_reserves_inference_worker(tmp_path):
    marker = tmp_path / 'pipeline-complete.json'
    cmd = f'RUN_ID=final PIPELINE_DONE={marker} python scripts/improvement_unit.py --cell pafa+gain --seed 2'
    assert not sweep.finished(cmd)
    assert sweep.running_elsewhere(cmd, cmd)
    marker.write_text('{}')
    assert sweep.finished(cmd)


def test_command_preserves_explicit_loader_workers():
    explicit = 'python -m auscult_trust.training.main --num_workers 0'
    assert sweep.with_loader_workers(explicit, 2) == explicit
    assert sweep.with_loader_workers('python train.py', 2).endswith('--num_workers 2')


def test_cv_seed_identity_does_not_double_reserve_other_seed():
    command = 'python -m auscult_trust.training.main --cell arm --cv_folds 3 --cv_fold 0 --seed 0'
    other = command.replace('--seed 0', '--seed 1')
    assert sweep.running_elsewhere(command, command)
    assert not sweep.running_elsewhere(other, command)


def test_orphan_training_jobs_count_against_worker_limit_after_restart(tmp_path, monkeypatch):
    import sys
    import pytest
    jobs = ['MEM_GB=10 python -m auscult_trust.training.main --cell arm --seed ' + str(seed) for seed in range(3)]
    jobfile = tmp_path / 'jobs.txt'
    jobfile.write_text('\n'.join(jobs))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['sweep.py', str(jobfile), '--workers', '2', '--gpu-gb', '60', '--ramp', '0'])
    monkeypatch.setattr(sweep, 'finished', lambda _: False)
    monkeypatch.setattr(sweep, 'ps_args', lambda: '\n'.join(jobs[:2]))
    monkeypatch.setattr(sweep, 'free_gb', lambda *a: 60)
    monkeypatch.setattr(sweep, 'free_ram_gb', lambda: 100)
    monkeypatch.setattr(sweep, 'admission_open', lambda *a: True)
    starts = []
    monkeypatch.setattr(sweep.subprocess, 'Popen', lambda *a, **kw: starts.append(a))
    class PollComplete(Exception):
        pass
    def stop_after_poll(_):
        raise PollComplete
    monkeypatch.setattr(sweep.time, 'sleep', stop_after_poll)
    with pytest.raises(PollComplete):
        sweep.main()
    assert not starts


def test_owned_training_job_is_not_double_counted_as_an_orphan(tmp_path, monkeypatch):
    import sys
    import pytest
    from types import SimpleNamespace
    jobs = ['MEM_GB=10 python -m auscult_trust.training.main --cell arm --seed ' + str(seed) for seed in range(3)]
    jobfile = tmp_path / 'jobs.txt'
    jobfile.write_text('\n'.join(jobs))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['sweep.py', str(jobfile), '--workers', '2', '--gpu-gb', '60', '--ramp', '0'])
    monkeypatch.setattr(sweep, 'finished', lambda _: False)
    starts = []
    monkeypatch.setattr(sweep, 'ps_args', lambda: '\n'.join(jobs[:len(starts)]))
    monkeypatch.setattr(sweep, 'free_gb', lambda *a: 60)
    monkeypatch.setattr(sweep, 'free_ram_gb', lambda: 100)
    monkeypatch.setattr(sweep, 'admission_open', lambda *a: True)
    def start(command, **kwargs):
        starts.append(command)
        return SimpleNamespace(poll=lambda: None)
    monkeypatch.setattr(sweep.subprocess, 'Popen', start)
    class PollComplete(Exception):
        pass
    polls = []
    def stop_after_two_polls(_):
        polls.append(1)
        if len(polls) == 2:
            raise PollComplete
    monkeypatch.setattr(sweep.time, 'sleep', stop_after_two_polls)
    with pytest.raises(PollComplete):
        sweep.main()
    assert len(starts) == 2
