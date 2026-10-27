from scripts import target_campaign as tc


def test_main_jobs_cover_every_endpoint_and_balance_machines():
    jobs = tc.main_jobs()
    assert len(jobs) == 25 + 5 * 25
    assert [j['arm'] for j in jobs[:25]] == [tc.CONTROL] * 25
    finals = [j for j in jobs if 'cohort' in j]
    assert {(j['arm'], j['cohort'], j['seed']) for j in finals} == {
        (arm, cohort, seed) for arm in tc.FINAL_EPOCHS for cohort in ['official', *tc.core.DEVICES] for seed in range(5)}
    assert all(j['epoch'] == tc.FINAL_EPOCHS[j['arm']] for j in finals)
    load = dict.fromkeys(tc.WEIGHTS, 0.0)
    for j in jobs:
        load[j['machine']] += tc.cost(j)
    ends = [load[m] / tc.WEIGHTS[m] for m in load]
    assert max(ends) - min(ends) < 0.6
    assert len({(j['run'], j['cell'], j['unit']) for j in jobs}) == len(jobs)


def test_control_finals_use_the_chosen_epoch():
    jobs, _ = tc.assign([], tc.core.final_jobs({tc.CONTROL: 17}, name=tc.NAME))
    assert len(jobs) == 25 and {j['epoch'] for j in jobs} == {17} and {j['cell'].split('+')[0] for j in jobs} == {tc.CONTROL}


def test_f_budget_is_limited_only_while_the_phase_is_built(tmp_path):
    spec = dict(name=tc.NAME, snapshot=str(tmp_path), stack=[], recipes={}, machines={m: {} for m in tc.WEIGHTS})
    tc.set_phase(spec, 'control_final', tc.assign([], tc.core.final_jobs({tc.CONTROL: 17}, name=tc.NAME))[0])
    assert spec['machines']['F']['gpu_gb'] == 30 and spec['machines']['B']['gpu_gb'] == 28
    assert tc.core.GPU_BUDGETS['F'] == 32


def _write_unit(root, run, arm, fold, seed, scores):
    import numpy as np
    unit = root / 'outputs/train' / f'{run}-s{seed}' / arm / f'cv{fold}'
    unit.mkdir(parents=True)
    history = [{'epoch': e + 1, 'val': {'all': {'score': float(s)}}} for e, s in enumerate(scores)]
    (unit / 'report.json').write_text(__import__('json').dumps({'history': history}))
    np.savez(unit / 'val_preds.npz', labels=np.arange(4), patient=np.arange(4), stem=np.array(list('abcd')), device=np.zeros(4))


def test_control_epoch_comes_from_its_own_mean_curve_and_cohorts_are_checked(tmp_path):
    import numpy as np
    curve = lambda peak: 50 + 10 * np.exp(-((np.arange(100) + 1 - peak) / 20) ** 2)
    reference, snapshot = tmp_path / 'reference', tmp_path / 'snapshot'
    for seed in range(5):
        for fold in range(5):
            _write_unit(reference, tc.REFERENCE, 'anchor', fold, seed, curve(66))
            _write_unit(reference, tc.REFERENCE, 'class_aware_light', fold, seed, curve(4))
            _write_unit(snapshot, tc.NAME, tc.CONTROL, fold, seed, curve(30))
    result = tc.control_result(dict(snapshot=str(snapshot), reference_snapshot=str(reference)))
    assert result['epoch'] == 30 and result['epoch_of'] == {'anchor': 66, 'class_aware_light': 4}
    assert set(result['score_at_epoch']['control']) == {'4', '30', '66'}
