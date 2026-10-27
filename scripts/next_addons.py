"""Audit learned-fusion inputs and run frozen patient-disjoint spectrum adaptation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from src.artifacts import atomic_json, atomic_npz, file_hashes
from src.evaluation.ensemble_study import adaptation_split, classification
from src.processing.icbhi import DEVICES
from scripts.followup_ensembles import (E1, E3, REFERENCE, SEEDS, paired_contrast)

SPLIT_SEED = 20261026


def audit_fusion(roots):
    """Inventory source CV predictions; no target predictions substitute for OOF."""
    inventory = []
    for root in roots:
        for path in sorted(root.rglob('*pred*.npz')):
            if any(part.startswith('cv') and part[2:].isdigit() for part in path.parts):
                inventory.append(str(path.resolve()))
    missing = [cell for cell in E1 + E3 if not any(cell in path for path in inventory)]
    return {'status': 'blocked', 'source_cv_prediction_files': inventory, 'teacher_oof_for_distillation': 'incomplete: no certified common OOF teacher bundle; use hard+soft KD only after exclusion/temperature provenance is verified',
            'missing_complementary_member_oof': missing,
            'reason': 'No certified complete aligned patient-disjoint OOF plus independent validation bundle supplied; learned weights were not fitted.',
            'required': ['same source patients/cycle IDs across all members', 'each OOF patient excluded from member training',
                         'independent patient validation for fusion selection', 'equal-weight control on identical patients']}


def audit_teachers(root, out):
    """Check saved fixed-epoch source OOF arrays and retained checkpoint availability."""
    from src.processing.splits import cv_split
    members, files = {}, []
    for cell, epoch in [('anchor', 66), ('llrd', 21)]:
        folds, patient_sets, arrays = [], [], []
        for fold in range(3):
            unit = root / 'outputs/train/improvements-20261026-s0' / cell / f'cv{fold}'
            path = unit / 'val_preds.npz'
            with np.load(path, allow_pickle=False) as source:
                epochs = source['epochs'].tolist()
                arrays.append({key: source[key].astype(str) if key in ('patient','stem') else source[key]
                               for key in ('labels','device','patient','stem')})
                probs = source['probs'][epochs.index(epoch)]
            if not np.isfinite(probs).all() or not np.allclose(probs.sum(1), 1, atol=.002):
                raise ValueError('Invalid OOF probabilities')
            args = json.loads((unit / 'train_args.json').read_text())
            report = json.loads((unit / 'report.json').read_text())
            if args['cv_folds'] != 3 or args['cv_fold'] != fold:
                raise ValueError('OOF fold configuration mismatch')
            patients = set(arrays[-1]['patient'])
            if any(patients & earlier for earlier in patient_sets):
                raise ValueError('Patient appears in multiple OOF folds')
            patient_sets.append(patients)
            checkpoint = root / 'archive/checkpoints/train/improvements-20261026-s0' / cell / f'cv{fold}/best.pth'
            files.extend([str(path), str(unit/'train_args.json'), str(unit/'report.json')])
            folds.append({'fold': fold, 'fixed_prediction_epoch': epoch, 'cycles': len(probs),
                'patients': sorted(patients), 'selected_checkpoint': str(checkpoint),
                'checkpoint_exists': checkpoint.exists(), 'cv_selected': report.get('cv_selected'),
                'fixed_epoch_checkpoint_exists': (checkpoint.parent / f'report_epoch_{epoch}.pth').exists()})
        patients = np.concatenate([a['patient'] for a in arrays])
        devices = np.concatenate([a['device'] for a in arrays])
        offset = 0
        for fold, array in enumerate(arrays):
            train, val = cv_split(patients, devices, 3, fold)
            if set(patients[train]) & set(array['patient']) or set(patients[val]) != set(array['patient']):
                raise ValueError('OOF arrays do not match deterministic training-patient folds')
            offset += len(array['patient'])
        members[cell] = {'folds': folds, 'cycles': offset, 'patients': len(set(patients)), 'arrays': arrays}
    for first, second in zip(members['anchor']['arrays'], members['llrd']['arrays']):
        for key in ('labels','device','patient','stem'):
            if not np.array_equal(first[key], second[key]):
                raise ValueError('Teacher OOF alignment mismatch: ' + key)
    for member in members.values():
        del member['arrays']
    receipt = {'status': 'aligned_patient_disjoint_source_oof_available', 'members': members,
        'input_hashes': file_hashes(files),
        'limits': ['fixed epochs selected using these same source CV outcomes; not independent fusion validation',
                  'retained best.pth uses per-fold validation selection; fixed epoch weights absent',
                  'OOF probabilities cover outer validation patients, not outer training targets',
                  'no calibrated teacher temperature supplied; source-only inner validation needed'],
        'recommendation': 'For clean improved KD CV, train source-fold teachers for frozen epochs on outer-training patients only; use explicitly in-sample hard+soft training targets and temperature fixed a priori or selected on inner source split. Do not use full-source teachers for outer CV.'}
    atomic_json(out / 'teacher-oof-audit.json', receipt)
    return receipt


def fit_sc(waves, filters, adaptation, reference, args):
    """Fit one correction per filter from adaptation audio only, without labels."""
    from src.processing.correction import mean_spectrum, spectrum_coefficients
    spectra = {name: mean_spectrum([waves[i] for i in adaptation if filters[i] == name])
               for name in np.unique(filters)}
    return spectrum_coefficients(spectra, reference=args.sc_reference, sc_mode=args.sc_mode,
        limit_freq_low=args.sc_limit_freq_low, limit_freq_high=args.sc_limit_freq_high,
        limit_freq_diff=args.sc_limit_freq_diff, reference_spectra={'ref': reference})


def sc_study(manifest_path, out, roots):
    import torch
    from src import runs
    from src.training.config import load_args
    from src.training.state import report_epoch
    from src.paths import DATA
    from src.processing.kauh import file_table, windows
    from src.processing.correction import apply_spectrum_correction
    from src.evaluation.kauh_infer import window_waves
    from src.inference.model import load_frozen, images_of, predict_images
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(manifest_path.read_text())
    table = file_table(DATA / 'raw/kauh')
    win = windows(table, 8.0).reset_index(drop=True)
    patients = win.patient.to_numpy().astype(str)
    adaptation, evaluation = adaptation_split(patients, np.random.default_rng(SPLIT_SEED))
    assert not set(patients[adaptation]) & set(patients[evaluation])
    units = [Path(manifest['members']['gain6_cv66']['official'].format(seed=seed)) for seed in SEEDS]
    checkpoints = [runs.twin_dir(unit, 'checkpoints') / f'report_epoch_{report_epoch(unit)}.pth' for unit in units]
    files = [str(manifest_path), str(Path(__file__))] + sorted(set(win.wav))
    files += [str(path) for unit, ckpt in zip(units, checkpoints)
              for path in (ckpt, unit/'train_args.json', unit/'sc_state.npz', unit/'report.json')]
    atomic_json(out / 'fusion-feasibility.json', audit_fusion(roots))
    atomic_json(out / 'freeze.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
        'method': 'source-selected static SC; arithmetic saved training reference; 50-2000 Hz; no target knob selection',
        'checkpoint_family': 'gain6_cv66', 'seeds': list(SEEDS), 'split_seed': SPLIT_SEED,
        'adaptation_patients': sorted(set(patients[adaptation])),
        'evaluation_patients': sorted(set(patients[evaluation])), 'labels_for_adaptation': False,
        'baseline': 'same SC-trained frozen checkpoint, raw target waveform, no target-derived coefficient',
        'adapted': 'same checkpoint, per-filter spectrum from adaptation patients only',
        'evaluation': 'recording binary abnormal-sound screening; mean 8s window abnormal probability, threshold 0.5',
        'scope': 'classification-only prototype; conformal calibration/coverage not evaluated; no deployment',
        'input_hashes': file_hashes(files), 'status': 'frozen_before_predictions'})
    print('Frozen patient split and artifact hashes', flush=True)
    waves = window_waves(win)
    args = load_args(units[0])
    references = [np.load(unit / 'sc_state.npz')['reference'] for unit in units]
    if not all(np.array_equal(references[0], ref) for ref in references):
        raise ValueError('Source reference differs between seeds')
    filters = win['filter'].to_numpy()
    coefficients = fit_sc(waves, filters, adaptation, references[0], args)
    atomic_npz(out / 'adaptation_state.npz', **coefficients)
    eval_waves = [waves[i] for i in evaluation]
    corrected = [apply_spectrum_correction(waves[i], coefficients[filters[i]]).astype(np.float32)
                 for i in evaluation]
    images = {'baseline': images_of(eval_waves, args), 'adapted': images_of(corrected, args)}
    metadata = {key: win.iloc[evaluation][key].to_numpy().astype(str) if key in ('patient','stem','filter')
                else win.iloc[evaluation][key].to_numpy() for key in ('label','patient','stem','filter','window')}
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(.32)
    
    for seed, unit, checkpoint in zip(SEEDS, units, checkpoints):
        args = load_args(unit)
        model, classifier = load_frozen(args, checkpoint)
        for mode, values in images.items():
            probs = np.concatenate([predict_images(values[i:i+4], model, classifier, method=args.method)
                                    for i in range(0, len(values), 4)])
            atomic_npz(out / f'{mode}-seed{seed}.npz', **metadata, probs=probs)
            print('Completed', seed, mode, flush=True)
        del model, classifier
        torch.cuda.empty_cache()
    return summarize_sc(out)


def summarize_sc(out):
    """Evaluate saved predictions without rerunning inference or changing the split."""
    from src.evaluation.recordings import kauh_recordings
    rows, predictions, files = [], {}, [str(Path(__file__))]
    frozen = json.loads((out / 'freeze.json').read_text())
    reference = None
    for seed in SEEDS:
        for mode in ('baseline', 'adapted'):
            path = out / f'{mode}-seed{seed}.npz'
            files.append(str(path))
            with np.load(path, allow_pickle=False) as source:
                data = dict(source)
            for key in ('label', 'patient', 'stem', 'filter', 'window'):
                if reference is not None and not np.array_equal(data[key], reference[key]):
                    raise ValueError('Baseline/adaptation/seed metadata mismatch: ' + key)
            reference = data
            if set(data['patient']) != set(frozen['evaluation_patients']):
                raise ValueError('Predictions do not match frozen evaluation patients')
            rec = kauh_recordings(data)
            rec['stem'] = np.unique(data['stem'])
            predictions[mode, seed] = rec
            for name in np.unique(rec['filter']):
                mask = rec['filter'] == name
                rows.append({'seed': seed, 'mode': mode, 'filter': str(name),
                    **classification(rec['y'][mask], rec['p'][mask].argmax(1))})
    summary = {}
    for name in np.unique(reference['filter']):
        mask = predictions['baseline', 0]['filter'] == name
        controls, adapted = [], []
        for seed in SEEDS:
            for mode, destination in [('baseline', controls), ('adapted', adapted)]:
                rec = predictions[mode, seed]
                destination.append({'seed': seed, 'labels': rec['y'][mask], 'patient': rec['patient'][mask],
                    'cycle_id': rec['stem'][mask], 'device': rec['filter'][mask], 'pred': rec['p'][mask].argmax(1)})
        contrast = paired_contrast(controls, adapted, 1000)
        contrast['scope'] = 'paired patient and model-seed bootstrap, conditional on one frozen adaptation cohort'
        summary[str(name)] = {**contrast, **{mode: {metric: {
            'mean': float(np.mean([r[metric] for r in rows if r['mode'] == mode and r['filter'] == name])),
            'sd': float(np.std([r[metric] for r in rows if r['mode'] == mode and r['filter'] == name], ddof=1))}
            for metric in ('sp','se','score','hs')} for mode in ('baseline','adapted')}}
    atomic_json(out / 'summary.json', {'status': 'complete', 'rows': rows, 'summary': summary,
        'report_input_hashes': file_hashes(files),
        'decision': 'exploratory evidence only; no selection or deployment',
        'limitations': ['KAUH software filters are not independent hardware devices',
            'one split; uncertainty conditional on fitted adaptation spectrum',
            'baseline intentionally omits target SC despite SC-trained weights',
            'previously observed target dataset, not a new untouched evaluation',
            'classification-only; no conformal guarantee or recalibration assessed']})
    return summary


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--audit-root', type=Path, action='append', required=True)
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    results = summarize_sc(args.out) if args.summarize_only else sc_study(args.manifest, args.out, args.audit_root)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
