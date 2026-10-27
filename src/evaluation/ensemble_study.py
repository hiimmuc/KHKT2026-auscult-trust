"""Fixed registered ensembles and patient-disjoint finite-target k-rate evaluation.

Run `python -m src.evaluation.ensemble_study registered|krate --out <new directory>`.
Historical member predictions are inputs, never overwritten. No weights or epochs are fitted.
"""
import argparse
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from src.artifacts import atomic_json, atomic_npz, file_hashes
from src.evaluation.calibration import coverage, null_delta_p95
from src.evaluation.lodo import load_unit
from src.evaluation.metrics import sp_se
from src.paths import OUTPUTS
from src.processing.icbhi import DEVICES, cycle_table
from src.processing.splits import both_sides_patients, lodo_split
from src.provenance import record_prediction

BEATS = ['beats+ft+' + m for m in ('ce', 'patchmix_cl', 'sg_scl', 'pafa', 'qlung')]
AST = ['ast+ft+' + m for m in ('ce', 'pafa', 'patchmix_cl')]
HTSAT = ['htsat+ft+' + m for m in ('ce', 'patchmix_cl', 'sg_scl')]
REGISTERED = {'R_E1_all11': BEATS + AST + HTSAT, 'R_E2_beats5': BEATS, 'R_E3_ast3': AST, 'R_E4_htsat3': HTSAT}
USER_BEATS = ['beats+ft+' + m for m in ('pafa', 'sg_scl', 'patchmix_cl')]
USER_HTSAT = ['htsat+ft+' + m for m in ('patchmix_cl', 'sg_scl')]
REFERENCE = 'beats+ft+pafa'


def classification(y, pred):
    sp, se = sp_se(y, pred)
    return {'sp': 100 * sp, 'se': 100 * se, 'score': 50 * (sp + se),
            'hs': 200 * sp * se / (sp + se) if sp + se else 0.0}


@lru_cache(maxsize=32)
def cycle_order(folder, held, seed, cal_frac):
    """Reconstruct deterministic annotation order; old exports have no explicit cycle IDs."""
    folder = Path(folder)
    table = cycle_table(folder / 'audio_test_data')
    table['patient'] = table.stem.str.split('_').str[0]
    table['cycle_id'] = table.stem + ':' + table.groupby('stem').cumcount().astype(str)
    table = table[~table.patient.isin(both_sides_patients(folder / 'official_split.txt'))].reset_index(drop=True)
    dev = table.device.map(DEVICES.index).to_numpy()
    _, ca, te = lodo_split(table.patient.to_numpy(), dev, DEVICES.index(held), cal_frac, seed)
    return {name: {'ids': table.cycle_id.to_numpy()[rows], 'y': table.label.to_numpy()[rows],
                   'pat': table.patient.to_numpy()[rows], 'dev': dev[rows]} for name, rows in [('cal', ca), ('test', te)]}


def read_member(root, cell, device, seed):
    unit = root / (cell + '+lodo_' + device.lower()) / f'seed{seed}'
    args = json.loads((unit / 'train_args.json').read_text())
    data = load_unit(unit)
    expected = cycle_order(args['data_folder'], device, seed, args.get('lodo_cal_frac', .2))
    for side in ('cal', 'test'):
        for key, actual in [('y', data[side + '_y']), ('pat', data[side + '_pat'])]:
            if not np.array_equal(expected[side][key].astype(str), actual.astype(str)):
                raise ValueError(f'Annotation-order metadata mismatch: {unit}, {side}, {key}')
        p = data[side + '_p']
        if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=.002):
            raise ValueError(f'Invalid member probabilities: {unit}')
        data[side + '_ids'] = expected[side]['ids']
    if not np.array_equal(expected['cal']['dev'], data['cal_dev']):
        raise ValueError(f'Calibration device order mismatch: {unit}')
    data['files'] = [unit / n for n in ('train_args.json', 'report.json', 'cal_preds.npz', 'test_preds.npz')]
    return data


def average(units):
    ref = units[0]
    for u in units[1:]:
        for key in ('cal_y', 'cal_pat', 'cal_dev', 'test_y', 'test_pat', 'cal_ids', 'test_ids'):
            if not np.array_equal(u[key], ref[key]):
                raise ValueError(f'Ensemble alignment mismatch: {key}')
    return {**ref, 'cal_p': np.mean([u['cal_p'] for u in units], 0),
            'test_p': np.mean([u['test_p'] for u in units], 0),
            'files': list({p for u in units for p in u['files']})}


def adaptation_split(patients, rng):
    """Random patient-level adaptation/evaluation split, without looking at labels."""
    patients = np.asarray(patients)
    ids = rng.permutation(np.unique(patients))
    if len(ids) < 2:
        raise ValueError('At least two target patients required')
    n = min(len(ids) - 1, max(1, int(round(.3 * len(ids)))))
    adapt = rng.permutation(np.flatnonzero(np.isin(patients, ids[:n])))
    evaluation = np.flatnonzero(~np.isin(patients, ids[:n]))
    return adapt, evaluation


def k_for_rate(p, rate):
    ratio = p[:, 1:].max(1) / np.maximum(p[:, 0], 1e-12)
    return float(np.quantile(ratio, rate))


def predict(p, k=1.0):
    ratio = p[:, 1:].max(1) / np.maximum(p[:, 0], 1e-12)
    return np.where(ratio <= k, 0, p[:, 1:].argmax(1) + 1)


def registered(args):
    rows = []
    for seed in range(5):
        for device in DEVICES:
            members = {c: read_member(args.input, c, device, seed) for c in BEATS + AST + HTSAT}
            for name, cells in REGISTERED.items():
                u = average([members[c] for c in cells])
                metrics = classification(u['test_y'], u['test_p'].argmax(1))
                cov = coverage(u['cal_p'], u['cal_y'], u['test_p'], u['test_y'])
                null = null_delta_p95(u, n_splits=args.null_splits, seed=seed)
                folder = args.out / (name + '+lodo_' + device.lower()) / f'seed{seed}'
                folder.mkdir(parents=True, exist_ok=False)
                inputs = {'member_files': file_hashes(u['files']), 'source': file_hashes([Path(__file__)]),
                          'member_epochs': {c: members[c]['epoch'] for c in cells},
                          'alignment': 'annotation order reconstructed; labels/patients/devices verified'}
                for side in ('cal', 'test'):
                    dest = folder / (side + '_preds.npz')
                    atomic_npz(dest, probs=u[side + '_p'][None], epochs=np.array([0]),
                               labels=u[side + '_y'], patient=u[side + '_pat'],
                               device=u['cal_dev'] if side == 'cal' else np.full(len(u['test_y']), DEVICES.index(device)),
                               cycle_id=u[side + '_ids'])
                    record_prediction(dest, inputs)
                atomic_json(folder / 'report.json', {'kind': 'fixed-softmax-ensemble', 'virtual_epoch': 0,
                            'members': inputs, 'fixed': {'0': {'test': {'all': metrics}}}})
                row = {'group': name, 'seed': seed, 'device': device, **metrics, **cov,
                       'null_p95': null, 'above_null_gate': abs(cov['delta']) > null,
                       'reference': classification(u['test_y'], members[REFERENCE]['test_p'].argmax(1))}
                rows.append(row)
                atomic_json(args.out / 'progress.json', {'units': len(rows), 'expected': 80, 'last': row})
                print(name, seed, device, round(metrics['score'], 3), flush=True)
    summary = {}
    rng = np.random.default_rng(20261024)
    for name in REGISTERED:
        selected = [r for r in rows if r['group'] == name]
        scores = np.array([[next(r['score'] for r in selected if r['seed'] == s and r['device'] == d)
                            for d in DEVICES] for s in range(5)])
        delta = np.array([[next(r['score'] - r['reference']['score'] for r in selected if r['seed'] == s and r['device'] == d)
                           for d in DEVICES] for s in range(5)])
        # Seed-only CI; patient-level uncertainty is not inferred from these 5 aggregate values.
        draws = delta.mean(1)[rng.integers(0, 5, size=(args.boot, 5))].mean(1)
        summary[name] = {'macro_mean': float(scores.mean()), 'macro_seed_sd': float(scores.mean(1).std(ddof=1)),
                         'macro_without_akgc417l': float(scores[:, :3].mean()),
                         'delta_vs_pafa': float(delta.mean()), 'seed_bootstrap_delta_ci': np.quantile(draws, [.025, .975]).tolist(),
                         'ci_scope': 'model seeds only; not a patient/seed two-level confidence interval',
                         'tested_folds_by_seed': [(r['seed'], r['device']) for r in selected if r['above_null_gate']]}
    atomic_json(args.out / 'summary.json', {'rows': rows, 'summary': summary, 'selection': 'fixed Amendment-4 equal weights'})


def krate(args):
    groups = {'U_beats3_static': USER_BEATS, 'U_beats3_htsat2_static': USER_BEATS + USER_HTSAT}
    rows, inputs = [], []
    for seed in range(5):
        for device in DEVICES:
            members = {c: read_member(args.input, c + '+sc+sc_static', device, seed) for c in USER_BEATS + USER_HTSAT}
            ref = members[REFERENCE]
            inputs.extend(p for u in members.values() for p in u['files'])
            for name, cells in groups.items():
                u = average([members[c] for c in cells])
                for rep in range(args.repetitions):
                    rng = np.random.default_rng(1_000_000 * seed + 10_000 * DEVICES.index(device) + rep)
                    adaptation, ev = adaptation_split(u['test_pat'], rng)
                    full_rate = float((predict(ref['test_p'][adaptation]) == 0).mean())
                    full_k = k_for_rate(u['test_p'][adaptation], full_rate)
                    y, p = u['test_y'][ev], u['test_p'][ev]
                    controls = {'raw': classification(y, predict(p)), 'pafa': classification(y, predict(ref['test_p'][ev])),
                                'full_adaptation': classification(y, predict(p, full_k))}
                    for tok in (1, 2, 5, 10, 20, 50, 100, 'all'):
                        n = len(adaptation) if tok == 'all' else tok
                        if n > len(adaptation):
                            continue
                        est = adaptation[:n]
                        rate = float((predict(ref['test_p'][est]) == 0).mean())
                        k = k_for_rate(u['test_p'][est], rate)
                        rows.append({'group': name, 'seed': seed, 'device': device, 'repetition': rep,
                                     'N': tok, 'actual_cycles': n, 'adapt_patients': len(np.unique(u['test_pat'][est])),
                                     'eval_patients': len(np.unique(u['test_pat'][ev])), 'eval_cycles': len(ev),
                                     'k': k, 'normal_rate': rate, **classification(y, predict(p, k)), 'controls': controls})
            atomic_json(args.out / 'progress.json', {'completed_folds': seed * 4 + DEVICES.index(device) + 1,
                                                     'expected_folds': 20, 'rows': len(rows)})
            print('finite-N', seed, device, 'rows', len(rows), flush=True)
    summary = {}
    for name in groups:
        summary[name] = {}
        for n in (1, 2, 5, 10, 20, 50, 100, 'all'):
            selected = [r for r in rows if r['group'] == name and r['N'] == n]
            # Equal fold and model-seed weight. Monte Carlo repetitions are not independent model seeds.
            per_seed = [np.mean([np.mean([r['score'] for r in selected if r['seed'] == s and r['device'] == d])
                                 for d in DEVICES if any(r['seed'] == s and r['device'] == d for r in selected)]) for s in range(5)]
            summary[name][str(n)] = {'score': float(np.mean(per_seed)), 'model_seed_sd': float(np.std(per_seed, ddof=1)),
                                     'rows': len(selected), 'missing_fold_repetitions': 5 * 4 * args.repetitions - len(selected)}
    atomic_json(args.out / 'summary.json', {'summary': summary, 'rows': rows,
                'inputs': file_hashes(inputs), 'source': file_hashes([Path(__file__)]),
                'SC': 'fixed existing full-pool estimate; only k-rate has a finite target budget',
                'sampling': '30% adaptation patients, disjoint remaining evaluation patients, nested cycle budgets'})


def sccompare(args):
    """Matched trained/inference SC modes; BEATs dynamic outputs already exist in the historical run."""
    cells = USER_BEATS + ['ast+ft+patchmix_cl', 'ast+ft+pafa'] + USER_HTSAT
    groups = {'U_beats3': USER_BEATS, 'U_ast2': ['ast+ft+patchmix_cl', 'ast+ft+pafa'], 'U_htsat2': USER_HTSAT}
    groups.update({c: [c] for c in cells})
    rows, inputs = [], []
    for mode, suffix in [('none', ''), ('static', '+sc+sc_static'), ('dynamic', '+sc')]:
        for seed in range(5):
            for device in DEVICES:
                members = {}
                for cell in cells:
                    source = args.dynamic if mode == 'dynamic' and not cell.startswith('beats+') else args.input
                    members[cell] = read_member(source, cell + suffix, device, seed)
                    inputs.extend(members[cell]['files'])
                for name, cs in groups.items():
                    u = average([members[c] for c in cs])
                    row = {'mode': mode, 'group': name, 'seed': seed, 'device': device,
                           **classification(u['test_y'], u['test_p'].argmax(1)),
                           **coverage(u['cal_p'], u['cal_y'], u['test_p'], u['test_y'])}
                    rows.append(row)
        print('SC comparison complete:', mode, flush=True)
    summary = {}
    for name in groups:
        summary[name] = {}
        for mode in ('none', 'static', 'dynamic'):
            per_seed = [np.mean([r['score'] for r in rows if r['group'] == name and r['mode'] == mode and r['seed'] == s])
                        for s in range(5)]
            summary[name][mode] = {'macro_mean': float(np.mean(per_seed)), 'macro_seed_sd': float(np.std(per_seed, ddof=1))}
    atomic_json(args.out / 'summary.json', {'summary': summary, 'rows': rows, 'inputs': file_hashes(inputs),
                                           'scope': 'LODO matched training/inference modes; cross-mode ablations not run'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', choices=('registered', 'krate', 'sccompare'))
    parser.add_argument('--input', type=Path, default=OUTPUTS / 'train/lodo-v1')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--null-splits', type=int, default=200)
    parser.add_argument('--boot', type=int, default=1000)
    parser.add_argument('--repetitions', type=int, default=50)
    parser.add_argument('--dynamic', type=Path, help='new AST/HTS-AT dynamic training outputs')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    if args.study == 'sccompare' and args.dynamic is None:
        parser.error('sccompare requires --dynamic')
    {'registered': registered, 'krate': krate, 'sccompare': sccompare}[args.study](args)


if __name__ == '__main__':
    main()
