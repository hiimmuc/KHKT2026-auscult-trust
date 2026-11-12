"""Fixed equal-weight E1/E1+E3 additions; evaluate saved predictions, never fit or select.

Use `template` to write the input manifest, then `evaluate manifest.json --out ...`.
Missing or misaligned artifacts produce a pending receipt, never an incomplete five-seed CI.
Run evaluation on B/D/E/F; this script does not launch training or inference.
Inputs must be trusted project exports: legacy string metadata uses NumPy object arrays.
"""
import argparse
from functools import lru_cache
import inspect
import json
from pathlib import Path

import numpy as np

from auscult_trust.artifacts import atomic_json, atomic_npz, file_hashes
from auscult_trust.evaluation.ensemble_study import classification, k_for_rate, predict
from auscult_trust.processing.icbhi import DEVICES

E1 = ['beats+ft+pafa', 'beats+ft+sg_scl', 'beats+ft+patchmix_cl']
E3 = ['htsat+ft+patchmix_cl', 'htsat+ft+sg_scl']
REFERENCE = E1[0]
GROUPS = {name + suffix: cells + addition for name, cells in [('E1', E1), ('E1_E3', E1 + E3)]
          for suffix, addition in [('', []), ('+gain6_cv66', ['gain6_cv66']), ('+LLRD_cv21', ['LLRD_cv21'])]}
SEEDS = tuple(range(5))
ADDITION_EPOCHS = {'gain6_cv66': 66, 'LLRD_cv21': 21}
CYCLE_KEYS = ('labels', 'patient', 'device', 'cycle_id')


def validate_object_metadata(array):
    for key, values in array.items():
        if values.dtype == object:
            if key not in ('patient', 'stem', 'filter', 'cycle_id') or not all(isinstance(item, str) for item in values.ravel()):
                raise ValueError('Unexpected object metadata field: ' + key)
            array[key] = values.astype(str)
    return array


def aligned_mean(arrays, keys=CYCLE_KEYS):
    if not arrays:
        raise ValueError('No ensemble members')
    reference = arrays[0]
    for array in arrays:
        for key in keys:
            if key not in array or key not in reference or not np.array_equal(array[key], reference[key]):
                raise ValueError('Ensemble alignment mismatch or missing metadata: ' + key)
        p = array['probs']
        if p.ndim != 2 or p.shape != reference['probs'].shape or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=.002):
            raise ValueError('Invalid member probabilities')
    return {**{key: reference[key] for key in keys},
            'probs': np.mean([array['probs'].astype(np.float64) for array in arrays], axis=0)}


def require_seeds(rows):
    seeds = [row['seed'] for row in rows]
    if sorted(seeds) != list(SEEDS):
        raise ValueError(f'Requires exactly seeds 0..4; received {seeds}')
    return sorted(rows, key=lambda row: row['seed'])


def common_patient_cohort(members):
    """One outcome-independent cohort for all six groups; never retain a partial patient."""
    indices, patient_cycles = {}, {}
    for name, member in members.items():
        ids = member['cycle_id'].astype(str)
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate cycle IDs: ' + name)
        indices[name] = {cycle: i for i, cycle in enumerate(ids)}
        patient_cycles[name] = {patient: set(ids[member['patient'] == patient]) for patient in np.unique(member['patient'])}
    reference_name = next(iter(members))
    reference = members[reference_name]
    common_ids = set.intersection(*(set(index) for index in indices.values()))
    ordered = [cycle for cycle in reference['cycle_id'].astype(str) if cycle in common_ids]
    for name, member in members.items():
        for key in ('labels', 'patient', 'device'):
            left = reference[key][[indices[reference_name][cycle] for cycle in ordered]]
            right = member[key][[indices[name][cycle] for cycle in ordered]]
            if not np.array_equal(left, right):
                raise ValueError('Shared cycle-ID metadata mismatch: ' + key)
    candidates = set.intersection(*(set(groups) for groups in patient_cycles.values()))
    retained = {patient for patient in candidates if all(groups[patient] == patient_cycles[reference_name][patient]
                for groups in patient_cycles.values())}
    ordered = [cycle for cycle in reference['cycle_id'].astype(str)
               if cycle in common_ids and reference['patient'][indices[reference_name][cycle]] in retained]
    if not ordered:
        raise ValueError('No provable common whole-patient cycle cohort')
    selected, losses = {}, {}
    for name, member in members.items():
        positions = [indices[name][cycle] for cycle in ordered]
        selected[name] = {**member, **{key: member[key][positions] for key in (*CYCLE_KEYS, 'probs')}}
        losses[name] = dict(original_cycles=len(member['labels']), original_patients=len(patient_cycles[name]),
            dropped_patients=sorted(set(patient_cycles[name])-retained),
            original_class_counts=np.bincount(member['labels'], minlength=4).tolist(),
            retained_class_counts=np.bincount(selected[name]['labels'], minlength=4).tolist())
    return selected, dict(rule='whole_patients_common_cycle_ids', retained_cycles=len(ordered),
                          retained_patients=sorted(retained), members=losses)


def paired_contrast(left, right, draws=1000):
    left, right = require_seeds(left), require_seeds(right)
    for first, second in zip(left, right):
        for key in CYCLE_KEYS:
            if not np.array_equal(first[key], second[key]) or not np.array_equal(first[key], left[0][key]):
                raise ValueError('Paired seed/patient alignment mismatch: ' + key)
    patients = np.unique(left[0]['patient'])
    groups = {patient: np.flatnonzero(left[0]['patient'] == patient) for patient in patients}
    rng, deltas = np.random.default_rng(20261026), []
    for _ in range(draws):
        indices = np.concatenate([groups[p] for p in rng.choice(patients, len(patients))])
        values = []
        for seed in rng.integers(0, 5, 5):
            y = left[seed]['labels'][indices]
            if not (y == 0).any() or not (y != 0).any():
                break
            values.append(classification(y, right[seed]['pred'][indices])['score'] -
                          classification(y, left[seed]['pred'][indices])['score'])
        if len(values) == 5:
            deltas.append(float(np.mean(values)))
    if not deltas:
        raise ValueError('No bootstrap draw contains normal and abnormal patients')
    point = np.mean([classification(r['labels'], r['pred'])['score'] - classification(l['labels'], l['pred'])['score']
                     for l, r in zip(left, right)])
    return dict(delta_score=float(point), ci95=np.quantile(deltas, [.025, .975]).tolist(),
                estimable_draws=len(deltas), requested_draws=draws,
                scope='paired patient and model-seed bootstrap, conditional on fixed full-pool k-rate thresholds')


@lru_cache(maxsize=256)
def load_cycle(unit_name, data_folder=None):
    from auscult_trust.training.state import report_epoch
    from auscult_trust.processing.icbhi import cycle_table
    from auscult_trust.processing.splits import both_sides_patients
    unit = Path(unit_name)
    args = json.loads((unit / 'train_args.json').read_text())
    if not args.get('spectrum_correction') or args.get('sc_mode') != 'static':
        raise ValueError(f'Fixed follow-up requires trained static SC: {unit}')
    epoch = report_epoch(unit)
    path = unit / 'test_preds.npz'
    if not path.exists():
        path = unit / 'export/test.npz'
    with np.load(path, allow_pickle=True) as source:  # trusted, hash-recorded historical exports contain object-string metadata
        array = {key: source[key] for key in ('probs', 'labels', 'patient', 'device')}
        if array['probs'].ndim == 3:
            epochs = source['epochs'].tolist()
            if epoch not in epochs:
                raise ValueError(f'Fixed report epoch {epoch} absent: {path}')
            array['probs'] = array['probs'][epochs.index(epoch)]
        if 'cycle_id' in source:
            array['cycle_id'] = source['cycle_id'].astype(str)
    metadata_files = []
    if 'cycle_id' not in array:
        folder = Path(data_folder or args['data_folder'])
        table = cycle_table(folder / 'audio_test_data')
        table['patient'] = table.stem.str.split('_').str[0]
        table['cycle_id'] = table.stem + ':' + table.groupby('stem').cumcount().astype(str)
        held = args.get('holdout_device')
        if held:
            table = table[(table.device == held) & ~table.patient.isin(both_sides_patients(folder / 'official_split.txt'))]
        else:
            official = dict(line.split() for line in (folder / 'official_split.txt').read_text().splitlines())
            table = table[table.stem.map(official) == 'test']
        expected = dict(labels=table.label.to_numpy(), patient=table.patient.to_numpy(), device=table.device.map(DEVICES.index).to_numpy())
        for key, values in expected.items():
            if not np.array_equal(values.astype(str), array[key].astype(str)):
                raise ValueError(f'Cannot reconstruct legacy cycle IDs: {path}: {key}')
        array['cycle_id'] = table.cycle_id.to_numpy().astype(str)
        metadata_files = [str(folder / 'official_split.txt')] + [str(folder / 'audio_test_data' / (stem+'.txt')) for stem in table.stem.unique()]
    validate_object_metadata(array)
    for key in ('patient', 'cycle_id'):
        array[key] = array[key].astype(str)
    aligned_mean([array])
    provenance_files = [candidate for candidate in (Path(str(path)+'.provenance.json'), unit / 'input_manifest.json', unit / 'freeze.json', unit / 'completion.json') if candidate.is_file()]
    return {**array, 'seed': int(args['seed']), 'epoch': epoch,
            'files': [str(unit / name) for name in ('train_args.json', 'report.json')] + [str(path), *metadata_files, *map(str, provenance_files)]}


def cycle_case(spec, subset, seed, folder=None):
    return load_cycle(spec['official' if subset == 'official' else 'lodo'].format(seed=seed, device=subset.lower()), folder)


def recording_arrays(path, kind):
    keys = ('label', 'patient', 'stem', 'window', 'device' if kind == 'screening' else 'filter')
    with np.load(path, allow_pickle=True) as source:
        arrays = {key: source[key] for key in (*keys, 'probs')}
    return validate_object_metadata(arrays), keys


def recording_metrics(screening, kauh):
    from auscult_trust.evaluation.recordings import icbhi_recordings, kauh_recordings
    from auscult_trust.processing.splits import calibration_eval_splits
    from auscult_trust.evaluation.conformal import lac_scores, split_threshold
    from auscult_trust.evaluation.screening import triage
    import pandas as pd
    source, target = icbhi_recordings(screening), kauh_recordings(kauh)
    results = {}
    frame = pd.DataFrame({'patient': source['patient'], 'device': source['device']})
    for split in calibration_eval_splits(frame, n_splits=20, seed=0):
        cal, evaluation = (np.isin(source['patient'], split[key]) for key in ('calibration', 'evaluation'))
        threshold = split_threshold(lac_scores(source['p'][cal], source['y'][cal]), .1)
        prior = np.bincount(source['y'][cal], minlength=2) / cal.sum()
        cohorts = {'ICBHI': (source['p'][evaluation], source['y'][evaluation])}
        for name in np.unique(target['filter']):
            mask = target['filter'] == name
            cohorts['KAUH_' + str(name)] = (target['p'][mask], target['y'][mask])
        for name, (p, y) in cohorts.items():
            results.setdefault(name, []).append(triage(threshold, p, y, prior))
    return {name: {key: float(np.nanmean([row[key] for row in rows])) for key in rows[0]} for name, rows in results.items()}


def evaluate(manifest, out, draws=1000):
    spec = json.loads(Path(manifest).read_text())
    if spec.get('schema') != 1 or spec.get('seeds') != list(SEEDS):
        raise ValueError('Manifest requires schema=1 and seeds=[0,1,2,3,4]')
    if set(spec['members']) != set(E1 + E3 + ['gain6_cv66', 'LLRD_cv21']):
        raise ValueError('Manifest must specify exactly the seven prespecified members')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    rows, pending, contrasts, recording, input_files = [], [], [], [], {str(manifest), __file__, inspect.getfile(classification)}
    cohort_audits = []
    cohort_rule = spec.get('cohort_rule', 'strict')
    if cohort_rule not in ('strict', 'whole_patients_common_cycle_ids'):
        raise ValueError('Unknown cohort rule')
    completed = {}
    for subset in ['official', *DEVICES]:
        common, errors = {}, {}
        if cohort_rule == 'whole_patients_common_cycle_ids':
            for seed in SEEDS:
                try:
                    units = {cell: cycle_case(member, subset, seed, spec.get('data_folder')) for cell, member in spec['members'].items()}
                    common[seed], audit = common_patient_cohort(units)
                    cohort_audits.append(dict(subset=subset, seed=seed, **audit))
                except (OSError, ValueError, KeyError, AssertionError) as error:
                    errors[seed] = str(error)
        for name, cells in GROUPS.items():
            group_rows = []
            for seed in SEEDS:
                try:
                    if seed in errors:
                        raise ValueError('Common cohort pending: ' + errors[seed])
                    members = [common[seed][cell] if seed in common else cycle_case(spec['members'][cell], subset, seed, spec.get('data_folder')) for cell in cells]
                    reference = common[seed][REFERENCE] if seed in common else cycle_case(spec['members'][REFERENCE], subset, seed, spec.get('data_folder'))
                    if any(member['seed'] != seed for member in [*members, reference]):
                        raise ValueError('Member seed differs from manifest seed')
                    for cell, member in zip(cells, members):
                        if cell in ADDITION_EPOCHS and member['epoch'] != ADDITION_EPOCHS[cell]:
                            raise ValueError(f'Prespecified epoch mismatch: {cell} requires {ADDITION_EPOCHS[cell]}')
                    average = aligned_mean(members)
                    aligned_mean([average, reference])
                    rate = float((predict(reference['probs']) == 0).mean())
                    k = k_for_rate(average['probs'], rate)
                    prediction = predict(average['probs'], k)
                    item = dict(group=name, subset=subset, seed=seed, k=k, reference_normal_rate=rate,
                                members=cells, weights=[1/len(cells)]*len(cells),
                                member_epochs={cell: member['epoch'] for cell, member in zip(cells, members)},
                                protocols={cell: spec['members'][cell]['lodo_protocol'] for cell in cells} if subset != 'official' else {},
                                **classification(average['labels'], prediction), raw=classification(average['labels'], average['probs'].argmax(1)))
                    rows.append(item)
                    group_rows.append({**average, 'seed': seed, 'pred': prediction})
                    files = sorted({path for member in [*members, reference] for path in member['files']})
                    input_files.update(files)
                    dest = out / 'predictions' / subset / name / f'seed{seed}.npz'
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    atomic_npz(dest, **average, pred=prediction, k=np.array(k))
                except (OSError, ValueError, KeyError, AssertionError) as error:
                    pending.append(dict(group=name, subset=subset, seed=seed, reason=str(error),
                                        required_units={cell: spec['members'][cell]['official' if subset == 'official' else 'lodo'].format(seed=seed, device=subset.lower()) for cell in cells}))
            completed[subset, name] = group_rows
        for base in ('E1', 'E1_E3'):
            for left, right in [(base, base+'+gain6_cv66'), (base, base+'+LLRD_cv21'), (base+'+gain6_cv66', base+'+LLRD_cv21')]:
                try:
                    contrasts.append(dict(subset=subset, left=left, right=right,
                                          **paired_contrast(completed[subset, left], completed[subset, right], draws)))
                except ValueError as error:
                    pending.append(dict(subset=subset, contrast=[left, right], reason=str(error)))
    for name, cells in GROUPS.items():
        for seed in SEEDS:
            try:
                means = {}
                for kind in ('screening', 'kauh'):
                    paths = [spec['members'][cell][kind].format(seed=seed) for cell in cells]
                    arrays = [recording_arrays(path, kind) for path in paths]
                    means[kind] = aligned_mean([array for array, _ in arrays], arrays[0][1])
                    input_files.update(paths)
                    input_files.update(str(path)+'.provenance.json' for path in paths if Path(str(path)+'.provenance.json').is_file())
                recording.append(dict(group=name, seed=seed, raw_mean_softmax=True, metrics=recording_metrics(**means)))
            except (OSError, ValueError, KeyError, AssertionError) as error:
                pending.append(dict(group=name, dataset='screening+KAUH', seed=seed, reason=str(error)))
    receipt = dict(schema=1, status='complete' if not pending else 'pending', fixed_groups=GROUPS,
        rule='Equal weight per individual member; full-pool label-free k-rate matches historical PAFA normal rate; no threshold uses target labels',
        scope='Exploratory fixed static-SC comparisons. Historical LODO-v1 and new LODO-v2 training protocols differ; exact held-out cohort alignment is mandatory. No target-based member/epoch selection.',
        recording_rule='Historical raw mean-softmax binary recordings, LAC alpha=.1, 20 fixed patient splits; cycle k-rate does not alter conformal probabilities.',
        rows=rows, paired_contrasts=contrasts, recording=recording, pending=pending, cohort_rule=cohort_rule, cohort_audits=cohort_audits,
        inputs=file_hashes(sorted(input_files)), expected_cycle_rows=150,
        learned_fusion='deferred: no verified patient-OOF training targets', inference_policy='Missing exports require frozen-checkpoint inference on B/D/E/F; no extra architecture training authorized here')
    atomic_json(out / 'receipt.json', receipt)
    lines = ['# Fixed follow-up ensembles', '', receipt['scope'], '', receipt['rule'], '',
             '| Group | Split | Seeds complete | Sp | Se | Score | HS |', '|---|---|---|---|---|---|---|']
    for subset in ['official', *DEVICES]:
        for name in GROUPS:
            selected = [row for row in rows if row['group'] == name and row['subset'] == subset]
            values = [f"{np.mean([row[key] for row in selected]):.3f}" for key in ('sp', 'se', 'score', 'hs')] if len(selected) == 5 else ['pending']*4
            lines.append('| '+' | '.join([name, subset, str(len(selected)), *values])+' |')
    if cohort_audits:
        lines += ['', 'Common cohort was fixed without outcome selection across all seven members. Whole patients with unequal cycle sets are excluded from every group. Lost patient IDs and original/retained class counts are in receipt.json.', '',
                  '| Split | Seed | Retained cycles | Retained patients | Most patients lost by one member |', '|---|---|---|---|---|']
        for audit in cohort_audits:
            lines.append('| '+' | '.join(map(str, [audit['subset'], audit['seed'], audit['retained_cycles'], len(audit['retained_patients']),
                max(len(member['dropped_patients']) for member in audit['members'].values())]))+' |')
    lines += ['', f"Receipt status: {receipt['status']}. Pending artifacts/alignments: {len(pending)}. See receipt.json for paired patient/seed CIs and recording coverage/uncertainty."]
    (out / 'report.md').write_text('\n'.join(lines)+'\n')
    return receipt


def template(legacy, anchor_official, anchor_lodo, llrd_official, llrd_lodo, common_patients=False):
    legacy = Path(legacy)
    members = {}
    for cell in E1 + E3:
        suffix = cell + '+sc+sc_static'
        record_base = 'auscult' if cell == REFERENCE else 'members/' + suffix
        members[cell] = dict(official=str(legacy / 'outputs/train/lodo-v1' / suffix / 'seed{seed}'),
            lodo=str(legacy / 'outputs/train/lodo-v1' / (suffix+'+lodo_{device}') / 'seed{seed}'),
            lodo_protocol='v1', **{kind: str(legacy / 'outputs' / kind / 'final-pafa-static' / record_base / 'seed{seed}.npz') for kind in ('screening', 'kauh')})
    for name, official, lodo in [('gain6_cv66', anchor_official, anchor_lodo), ('LLRD_cv21', llrd_official, llrd_lodo)]:
        members[name] = dict(official=official, lodo=lodo, lodo_protocol='v2', epoch=ADDITION_EPOCHS[name],
            screening=str(Path(official) / 'screening.npz'), kauh=str(Path(official) / 'kauh.npz'))
    return dict(schema=1, seeds=list(SEEDS), members=members,
                cohort_rule='whole_patients_common_cycle_ids' if common_patients else 'strict')


def main():
    parser = argparse.ArgumentParser(__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('template')
    create.add_argument('--legacy-root', type=Path, required=True)
    for name in ('anchor-official', 'anchor-lodo', 'llrd-official', 'llrd-lodo'):
        create.add_argument('--'+name, required=True)
    create.add_argument('--out', type=Path, required=True)
    create.add_argument('--common-patients', action='store_true', help='prespecify one proven whole-patient common cohort across all seven members')
    run = commands.add_parser('evaluate')
    run.add_argument('manifest', type=Path)
    run.add_argument('--out', type=Path, required=True)
    run.add_argument('--boot', type=int, default=1000)
    args = parser.parse_args()
    if args.command == 'template':
        if args.out.exists():
            raise FileExistsError(args.out)
        atomic_json(args.out, template(args.legacy_root, args.anchor_official, args.anchor_lodo, args.llrd_official, args.llrd_lodo, args.common_patients))
    else:
        if args.boot < 1:
            parser.error('--boot must be positive')
        receipt = evaluate(args.manifest, args.out, args.boot)
        print(json.dumps({'status': receipt['status'], 'rows': len(receipt['rows']), 'pending': len(receipt['pending'])}))


if __name__ == '__main__':
    main()
