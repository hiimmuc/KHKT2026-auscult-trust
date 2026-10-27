"""Summarize the frozen two-arm final campaign after every pipeline unit completes."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.artifacts import atomic_json, sha256_file, verify_file
from src.evaluation.ensemble_study import classification
from src.evaluation.lodo import load_unit
from src.evaluation.calibration import coverage


def paired_score_interval(pairs, draws=1000):
    rng = np.random.default_rng(20261025)
    deltas = []
    patients = np.unique(pairs[0][0]['test_pat'])
    for control, gain in pairs:
        np.testing.assert_array_equal(np.unique(control['test_pat']), patients)
        np.testing.assert_array_equal(control['test_y'], gain['test_y'])
        np.testing.assert_array_equal(control['test_pat'], gain['test_pat'])
    for _ in range(draws):
        sample = []
        patient_sample = rng.choice(patients, len(patients))
        for seed in rng.integers(0, len(pairs), len(pairs)):
            control, gain = pairs[seed]
            ix = np.concatenate([np.flatnonzero(control['test_pat'] == p) for p in patient_sample])
            scores = [classification(u['test_y'][ix], u['test_p'][ix].argmax(1))['score'] for u in (control, gain)]
            sample.append(scores[1] - scores[0])
        deltas.append(np.mean(sample))
    finite = np.asarray(deltas)[np.isfinite(deltas)]
    if not len(finite):
        raise ValueError('No estimable paired bootstrap draws')
    return {'ci': np.quantile(finite, [.025, .975]).tolist(), 'estimable_draws': len(finite)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('manifest')
    args = ap.parse_args()
    spec = json.loads(Path(args.manifest).read_text())
    root = Path(spec['source'])
    run = root / 'outputs/train' / spec['run']
    out = root / 'outputs/evaluation' / spec['name']
    out.mkdir(parents=True, exist_ok=True)
    rows, cohorts, inputs = [], {}, {}
    for cohort in ['official', *spec['devices']]:
        pairs = []
        for seed in range(5):
            units = []
            for arm in ('control', 'gain'):
                cell = spec['cells'][arm] + ('' if cohort == 'official' else '+lodo_' + cohort.lower())
                unit = run / cell / f'seed{seed}'
                marker = json.loads((unit / 'pipeline-complete.json').read_text())
                # Collected outputs keep the original snapshot paths in receipts.
                original = Path(spec['snapshot']) / 'outputs/train' / spec['run'] / cell / f'seed{seed}'
                for name, digest in marker['files'].items():
                    relative = Path(name).relative_to(original)
                    path = unit / relative
                    verify_file(path, digest)
                    inputs[str(path)] = digest
                inputs[str(unit / 'pipeline-complete.json')] = sha256_file(unit / 'pipeline-complete.json')
                if cohort == 'official':
                    with np.load(unit / 'export/test.npz', allow_pickle=True) as z:
                        u = {'test_p': z['probs'].astype(np.float32), 'test_y': z['labels'], 'test_pat': z['patient']}
                    for kind in ('screening', 'kauh'):
                        dest = root / 'outputs' / kind / spec['name'] / arm / f'seed{seed}.npz'
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(unit / (kind + '.npz'), dest)
                else:
                    u = load_unit(unit)
                row = dict(cohort=cohort, seed=seed, arm=arm,
                           **classification(u['test_y'], u['test_p'].argmax(1)))
                if cohort != 'official':
                    row.update(coverage(u['cal_p'], u['cal_y'], u['test_p'], u['test_y']))
                rows.append(row)
                units.append(u)
            pairs.append(units)
        cohorts[cohort] = {arm: {'score_mean': float(np.mean([r['score'] for r in rows if r['cohort'] == cohort and r['arm'] == arm])),
            'score_seed_sd': float(np.std([r['score'] for r in rows if r['cohort'] == cohort and r['arm'] == arm], ddof=1))} for arm in ('control', 'gain')}
        interval = paired_score_interval(pairs)
        cohorts[cohort]['gain_minus_control_ci'] = interval['ci']
        cohorts[cohort]['estimable_draws'] = interval['estimable_draws']
    inputs[str(Path(__file__).resolve())] = sha256_file(__file__)
    atomic_json(out / 'summary.json', {'cohorts': cohorts, 'rows': rows, 'inputs': inputs,
        'scope': 'Fixed source-selected epochs; 1000 paired seed/patient draws, nonfinite draws excluded, pointwise exploratory intervals. Official includes published shared patients; clean-patient subset not computed here. LODO uses new train-only reference protocol.'})
    lines = ['# Frozen final campaign', '', '| Cohort | Control Score | Gain Score | Delta | Paired 95% interval |', '| --- | ---: | ---: | ---: | --- |']
    for cohort, result in cohorts.items():
        c, g = result['control']['score_mean'], result['gain']['score_mean']
        lines.append(f'| {cohort} | {c:.2f} | {g:.2f} | {g-c:+.2f} | {result["gain_minus_control_ci"]} |')
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    commands = [
        [sys.executable, '-m', 'src.evaluation.kauh', '--dir', str(root / 'outputs/kauh' / spec['name']), '--base', 'control', '--arms', 'gain', '--boot', '1000', '--null_splits', '200'],
        [sys.executable, '-m', 'src.evaluation.screening', 'analyse', '--dir', str(root / 'outputs/screening' / spec['name']), '--kauh_dir', str(root / 'outputs/kauh' / spec['name']), '--pairs', 'control:control', 'gain:gain'],
    ]
    for command in commands:
        subprocess.run(command, cwd=root, check=True)
    atomic_json(out / 'complete.json', {'summary_sha256': sha256_file(out / 'summary.json'), 'status': 'complete'})


if __name__ == '__main__':
    main()
