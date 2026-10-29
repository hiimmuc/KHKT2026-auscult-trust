from pathlib import Path

from scripts import distill_campaign as dc


def spec(tmp_path):
    return dict(snapshot=str(tmp_path), stack=[], recipes={arm: [] for arm in dc.CONFIGS})


def test_official_phase_has_fifteen_balanced_jobs():
    jobs = dc.student_jobs(dc.CONFIGS, {'official'}, range(5))
    assert len(jobs) == 15 and {j['epoch'] for j in jobs} == {dc.STUDENT_EPOCH}
    load = dict.fromkeys(dc.WEIGHTS, 0.0)
    for j in jobs:
        load[j['machine']] += dc.cost(j)
    ends = [load[m] / dc.WEIGHTS[m] for m in load]
    assert max(ends) - min(ends) < 1.5


def test_command_exports_members_of_the_same_cohort_and_seed_then_trains(tmp_path):
    job = [j for j in dc.student_jobs(['kd_g4_k2'], {'Litt3200'}, [1])][0]
    text = dc.command(spec(tmp_path), job)
    export, train = text.split(' && ')[1], text.split(' && ')[2]
    assert export.count('--units') == 1 and text.count('lodo_litt3200/seed1') >= 8  # 4 member units + 4 checkpoints
    assert '--holdout_device Litt3200' in train and '--kd_hard_weight 0.5' in train and '--kd_temperature 2' in train
    assert ' teacher.npz' not in train.replace('/teacher.npz', '')  # the placeholder must not survive as a stray argument
    assert f'--soft_targets {tmp_path}/outputs/train/{dc.NAME}-final/kd_g4_k2+lodo_litt3200/seed1/teacher.npz' in train
    assert f'--report_epochs {dc.STUDENT_EPOCH}' in train


def test_k1_adds_no_loss_options_and_g7_has_all_members(tmp_path):
    job = dc.student_jobs(['kd_g7_k2'], {'official'}, [0])[0]
    assert len(dc.teacher_files('g7', 'official', 0)[0]) == 7
    k1 = dc.student_jobs(['kd_g4_k1'], {'official'}, [0])[0]
    assert dc.command(spec(tmp_path), k1).count('--kd_hard_weight') == 0


def test_gate_needs_mean_gain_and_seed_majority():
    ref = dict.fromkeys(range(5), 63.0)
    assert dc.gate({s: 63.5 for s in range(5)}, ref)['pass']
    assert dc.gate({0: 66.0, 1: 66.0, 2: 66.0, 3: 61.0, 4: 61.0}, ref)['pass']  # three of five seeds above, mean +1.2
    assert not dc.gate({0: 70.0, 1: 70.0, 2: 61.0, 3: 61.0, 4: 61.0}, ref)['pass']  # mean up, only 2 seeds
    assert not dc.gate({s: 63.1 for s in range(5)}, ref)['pass']  # below the margin


def test_member_paths_do_not_nest_when_run_from_a_snapshot():
    path = dc.unit_dir('swa', 'official', 0, 'outputs')
    assert 'campaigns/distill' not in str(path) and path.parts.count('campaigns') == 1
