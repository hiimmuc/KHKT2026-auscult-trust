"""Training-only, cycle-aligned soft-target export for frozen teachers and PAFA students."""
import argparse
import json
from pathlib import Path
import shlex

import numpy as np
import torch

from auscult_trust.artifacts import atomic_npz, file_hashes
from auscult_trust.inference.model import load_frozen
from auscult_trust.paths import DATA, THIRD_PARTY
from auscult_trust.provenance import record_prediction, verify_prediction
from auscult_trust.training.config import load_args


def training_rows(dataset):
    """Use precisely the teacher/student optimizer cohort, excluding validation/calibration."""
    if dataset.lodo is not None:
        return dataset.lodo[0]
    if dataset.args.selection == 'cv':
        from auscult_trust.training.dataset import _validation_indices
        return _validation_indices(dataset)[0]
    return np.arange(len(dataset.labels))


def validate_teacher(unit, checkpoint, args, expected_epoch=None):
    """Bind a frozen teacher snapshot to its declared epoch and source cohort."""
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    epoch = int(saved['epoch'])
    if expected_epoch is not None and epoch != expected_epoch:
        raise ValueError('Teacher checkpoint epoch differs from the frozen epoch')
    cohort_keys = ('seed', 'selection', 'cv_folds', 'cv_fold', 'holdout_device',
                   'spectrum_correction', 'sc_mode', 'method', 'test_fold', 'val_frac', 'lodo_cal_frac',
                   'data_folder', 'n_cls', 'sample_rate', 'desired_length')
    if any(saved['args'].get(k) != getattr(args, k, None) for k in cohort_keys):
        raise ValueError('Teacher checkpoint source cohort/config mismatch')
    history = json.loads((Path(unit) / 'report.json').read_text())['history']
    if epoch not in {int(h['epoch']) for h in history}:
        raise ValueError('Teacher checkpoint epoch missing from completed training report')
    if getattr(args, 'soft_targets', None):
        raise ValueError('Student cannot be used as an undeclared teacher')
    return epoch


def aligned_targets(path, ids, labels, patients, devices, train_rows):
    """Reject unverified, reordered or non-training teacher targets before any optimization."""
    certificate = json.loads(Path(str(path) + '.provenance.json').read_text())
    verify_prediction(path, certificate['inputs'])
    # Hashes bind the target export to teacher/data/source inputs; check them again on load.
    for filename, expected in certificate['inputs']['files'].items():
        from auscult_trust.artifacts import verify_file
        verify_file(filename, expected)
    with np.load(path) as z:
        for key, expected in [('cycle_id', ids[train_rows]), ('labels', labels[train_rows]),
                              ('patient', patients[train_rows]), ('device', devices[train_rows])]:
            if not np.array_equal(z[key].astype(str), expected.astype(str)):
                raise ValueError(f'Soft-target training alignment mismatch: {key}')
        p = z['probs'].astype(np.float32)
    if p.shape != (len(train_rows), 4) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError('Invalid teacher probabilities')
    out = np.zeros((len(ids), 4), np.float32)
    out[train_rows] = p
    return out


def export(args):
    from torchvision import transforms
    from auscult_trust.training.dataset import ICBHIDataset
    from auscult_trust.processing.cycles import read_annotations
    from auscult_trust.training.methods import get_method
    if args.out.exists():
        raise FileExistsError(args.out)
    predictions, first, files, contexts = [], None, [], []
    files += [DATA / 'models/beats/BEATs_iter3_plus_AS2M.pt']
    files += list(THIRD_PARTY.rglob('*.py'))
    for unit, checkpoint in zip(args.units, args.checkpoints):
        a = load_args(unit)
        if getattr(a, 'soft_targets', None):
            raise ValueError('Student cannot be used as an undeclared teacher')
        epoch = validate_teacher(unit, checkpoint, a, getattr(args, 'expected_epoch', None))
        ds = ICBHIDataset(True, transforms.ToTensor(), a, print_flag=False)
        rows = training_rows(ds)
        excluded = np.setdiff1d(np.arange(len(ds.labels)), rows)
        if set(ds.patients[rows]) & set(ds.patients[excluded]):
            raise ValueError("Teacher optimizer and held-out patients overlap")
        ann = read_annotations(Path(a.data_folder) / 'audio_test_data')
        ids = np.array([f'{stem}:{i}' for stem in ds.filenames for i in range(len(ann[stem]))])[rows]
        labels, patients, devices = np.asarray(ds.labels)[rows], ds.patients[rows], np.asarray(ds.devices)[rows]
        metadata = dict(cycle_id=ids, labels=labels, patient=patients, device=devices)
        if first is not None and any(not np.array_equal(metadata[k], first[k]) for k in metadata):
            raise ValueError('Teacher training-cycle alignment mismatch')
        first = metadata
        model, classifier = load_frozen(a, checkpoint)
        objective = get_method(a.method)
        p = []
        with torch.no_grad():
            for i in range(0, len(rows), 32):
                images = np.stack([ds.images[j] for j in rows[i:i + 32]])
                device = next(classifier.parameters()).device
                x = torch.from_numpy(images).permute(0, 3, 1, 2).float().to(device)
                p.append(objective.probabilities(classifier(model(x))).cpu().numpy())
        predictions.append(np.concatenate(p))
        contexts.append({'unit': str(unit), 'checkpoint': str(checkpoint), 'seed': a.seed,
                         'heldout_device': a.holdout_device, 'sc': a.spectrum_correction, 'sc_mode': a.sc_mode,
                         'epoch': epoch, 'cv_folds': a.cv_folds, 'cv_fold': a.cv_fold,
                         'selection': a.selection, 'training_patients': sorted(set(patients.tolist())),
                         'excluded_patients': sorted(set(ds.patients[excluded].tolist())),
                         'in_sample_targets': True})
        files += [Path(unit) / 'train_args.json', Path(unit) / 'report.json', checkpoint]
        if a.spectrum_correction:
            files.append(Path(unit) / 'sc_state.npz')
        folder = Path(a.data_folder)
        files += [folder / 'official_split.txt', *sorted((folder / 'audio_test_data').glob('*'))]
        files += list(Path(__file__).resolve().parents[1].rglob('*.py'))
        del model, classifier, ds
        torch.cuda.empty_cache()
        print('Teacher exported:', unit, len(rows), flush=True)
    if len({(c['seed'], c['heldout_device']) for c in contexts}) != 1:
        raise ValueError('Teachers must share seed and held-out fold')
    probabilities = np.mean(predictions, 0)
    atomic_npz(args.out, probs=probabilities, **first)
    record_prediction(args.out, {'files': file_hashes(files), 'teachers': contexts,
                                'target': 'in-sample teacher targets on optimizer training cycles only; raw softmax temperature 1'})
    print('Soft targets:', args.out, probabilities.shape, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--units', type=Path, nargs='+')
    parser.add_argument('--checkpoints', type=Path, nargs='+')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--expected_epoch', type=int, help='require this frozen teacher epoch')
    parser.add_argument('--jobs', type=Path, help='generate the fixed three-teacher static-SC pilot job list')
    parser.add_argument('--run', default='distill-pilot-20261024')
    args = parser.parse_args()
    if args.jobs:
        lines = []
        for fold in ('', '+lodo_litt3200'):
            for seed in range(3):
                cell = 'beats+ft+pafa+kd+sc+sc_static' + fold
                teachers = [Path('inputs/teachers') / ('beats+ft+' + m + '+sc+sc_static' + fold) / f'seed{seed}'
                            for m in ('pafa', 'sg_scl', 'patchmix_cl')]
                checkpoints = [p / f'report_epoch_{100 if i == 0 else 50}.pth' for i, p in enumerate(teachers)]
                target = Path('outputs/distillation') / args.run / cell / f'seed{seed}.npz'
                cfg = ['base', 'encoder_beats', 'grid/pafa_beats', 'sc', 'sc_static']
                if fold:
                    cfg.append('lodo_litt3200')
                prefix = f'RUN_ID={args.run} MEM_GB=10 .venv-train/bin/python -u'
                export_cmd = shlex.join(['-m', 'auscult_trust.training.distillation', '--units', *map(str, teachers),
                                        '--checkpoints', *map(str, checkpoints), '--out', str(target)])
                train_cmd = shlex.join(['-m', 'auscult_trust.training.main', '--config', ','.join('experiments/train/' + c + '.yaml' for c in cfg),
                                       '--cell', cell, '--selection', 'fixed', '--report_epochs', '100', '--seed', str(seed),
                                       '--soft_targets', str(target)])
                lines.append(prefix + ' ' + export_cmd + ' && ' + prefix + ' ' + train_cmd)
        if args.jobs.exists():
            raise FileExistsError(args.jobs)
        args.jobs.parent.mkdir(parents=True, exist_ok=True)
        args.jobs.write_text('\n'.join(lines) + '\n')
        print(args.jobs, len(lines), 'pilot units')
        return
    if not args.units or not args.checkpoints or not args.out:
        parser.error('--units, --checkpoints and --out are required for teacher export')
    if len(args.units) != len(args.checkpoints) or len(args.units) < 1:
        parser.error('At least one unit and one checkpoint per unit are required')
    export(args)


if __name__ == '__main__':
    main()
