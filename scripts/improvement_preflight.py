"""One real source-only AMP optimizer step per improvement; no target datasets."""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

ARMS = ('llrd', 'patient_cap', 'class_topk', 'class_attention', 'class_aware',
        'recording_head', 'band_noise', 'band_noise_consistency')
ANCHOR = ('base', 'encoder_beats', 'grid/pafa_beats', 'sc', 'sc_static', 'sc_gain')
SOURCE_KEYS = ('data_folder', 'test_fold', 'cv_folds', 'cv_fold', 'seed', 'encoder',
               'sample_rate', 'desired_length', 'pad_types', 'n_cls', 'spectrum_correction',
               'sc_mode', 'sc_reference', 'sc_limit_freq_low', 'sc_limit_freq_high',
               'sc_limit_freq_diff', 'sc_test_profile', 'sc_only_device', 'random_gain_db',
               'spec_augment', 'noise_train_prob', 'soft_targets')


def check_source_recipe(base_args, args):
    changed = [key for key in SOURCE_KEYS if getattr(base_args, key) != getattr(args, key)]
    if changed:
        raise ValueError(f'Source cache recipe differs: {changed}')


def recipe(arm):
    from src.training.config import parse_args
    configs = [str(REPO / 'experiments/train' / f'{name}.yaml') for name in (*ANCHOR, arm)]
    return parse_args(['--config', ','.join(configs), '--cv_folds', '3', '--cv_fold', '0',
                       '--seed', '0', '--num_workers', '0'])


def gradient_check(module, require_all=False):
    import torch
    gradients = [p.grad for p in module.parameters() if p.requires_grad]
    present = [g for g in gradients if g is not None]
    if not present or (require_all and len(present) != len(gradients)):
        raise AssertionError('Expected trainable parameters lack gradients')
    if not all(torch.isfinite(g).all() for g in present):
        raise AssertionError('Nonfinite trainable gradients')
    magnitude = sum(g.detach().float().abs().sum().item() for g in present)
    if magnitude == 0:
        raise AssertionError('All trainable gradients are zero')
    return {'parameters_with_gradient': len(present), 'gradient_l1': magnitude}


def source_batch(base, indices, args):
    import numpy as np
    from torch.utils.data import DataLoader, Subset
    from src.training.dataset import _training_sampling
    ds = copy.copy(base)
    ds.args = args
    selected = Subset(ds, indices)
    if args.method in ('ce_recording_pafa', 'ce_device_pafa'):
        if args.method == 'ce_recording_pafa':
            from src.training.recording_bags import wrap_training_dataset
        else:
            from src.training.device_augment import wrap_training_dataset
        selected = wrap_training_dataset(selected, args)
        options = {'shuffle': True, 'drop_last': False}
    else:
        options = _training_sampling(ds, np.asarray(indices), args)
    return next(iter(DataLoader(selected, batch_size=args.batch_size, num_workers=0, **options)))


def trial(arm, base, indices):
    import numpy as np
    import torch
    from src.training.methods import get_method
    from src.training.methods.base import Batch, StepContext
    from src.training.modeling import build_model
    args = recipe(arm)
    check_source_recipe(base.args, args)
    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    args.class_nums = np.bincount(np.asarray(base.labels)[indices], minlength=args.n_cls)
    args_hash = hashlib.sha256(json.dumps(vars(args), sort_keys=True, default=str).encode()).hexdigest()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    values = source_batch(base, indices, args)
    images, labels, devices, patients, quality, *extra = values
    images, labels = images.cuda(), labels.cuda()
    components = build_model(args)
    model, classifier, projector, criteria, optimizer = components
    model.train(), classifier.train(), projector.train()
    method = get_method(args.method)
    batch = Batch(images, labels, devices, patients, quality, extra[0].cuda() if extra else None)
    context = StepContext(model, classifier, projector, criteria, args, 1, 0, 1, method.trainable_frontend)
    scaler = torch.cuda.amp.GradScaler()
    step_start = time.monotonic()
    overflow_attempts = []
    for attempt in range(1, 7):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            result = method.step(context, batch)
        if not torch.isfinite(result.loss):
            raise AssertionError(f'{arm}: nonfinite loss')
        scaler.scale(result.loss).backward()
        scaler.unscale_(optimizer)
        nonfinite = [f'{group}.{name}' for group, module in
                     [('encoder', model), ('classifier', classifier), ('projector', projector)]
                     for name, parameter in module.named_parameters()
                     if parameter.grad is not None and not torch.isfinite(parameter.grad).all()]
        if nonfinite:
            overflow_attempts.append({'scale': scaler.get_scale(), 'parameters': nonfinite})
            scaler.step(optimizer), scaler.update()  # GradScaler skips this overflowed update.
            continue
        gradients = {'encoder': gradient_check(model), 'classifier': gradient_check(classifier, True),
                     'projector': gradient_check(projector, True)}
        if hasattr(classifier, 'recording_head'):
            gradients['recording_head'] = gradient_check(classifier.recording_head, True)
            gradients['cycle_head'] = gradient_check(classifier.cycle_head, True)
        if hasattr(classifier, 'attention'):
            gradients['classification_attention'] = gradient_check(classifier.attention, True)
        scaler.step(optimizer), scaler.update()
        break
    else:
        raise AssertionError(f'{arm}: nonfinite gradients after six AMP scaler backoffs: {nonfinite[:5]}')
    torch.cuda.synchronize()
    output = {'arm': arm, 'status': 'passed', 'args_sha256': args_hash,
              'args': vars(args), 'batch_shape': list(images.shape), 'loss': result.loss.item(),
              'gradients': gradients, 'amp_attempts': attempt, 'amp_final_scale': scaler.get_scale(),
              'amp_overflows': overflow_attempts, 'step_seconds': time.monotonic() - step_start,
              'total_seconds': time.monotonic() - started,
              'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
              'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30}
    return output


def main():
    import numpy as np
    import torch
    from torchvision import transforms
    from src.artifacts import atomic_json
    from src.training.dataset import ICBHIDataset, _validation_indices
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='logs/improvements-20261026/preflight.json')
    ap.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    cli = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Real GPU preflight requires CUDA')
    torch.set_num_threads(4)
    args = recipe('llrd')
    args.layer_decay = 1.0
    start = time.time()
    # Deliberately construct only official-source training audio, not build_loaders' test path.
    base = ICBHIDataset(True, transforms.ToTensor(), args)
    train_indices, val_indices = _validation_indices(base)
    train_patients = sorted(set(map(str, np.asarray(base.patients)[train_indices])))
    val_patients = sorted(set(map(str, np.asarray(base.patients)[val_indices])))
    if set(train_patients) & set(val_patients):
        raise AssertionError('Source train/validation patients overlap')
    report = {'started_unix': start, 'gpu': torch.cuda.get_device_name(), 'source_only': True,
              'official_test_constructed': False, 'source_train_patients': train_patients,
              'source_validation_patients': val_patients, 'source_train_cycles': len(train_indices),
              'source_validation_cycles': len(val_indices), 'arms': []}
    for arm in cli.arms:
        try:
            output = trial(arm, base, train_indices)
        except Exception as exc:
            import traceback
            output = {'arm': arm, 'status': 'failed', 'error': repr(exc), 'traceback': traceback.format_exc()}
        report['arms'].append(output)
        report['updated_unix'] = time.time()
        atomic_json(cli.out, report)
        print(json.dumps({k: output[k] for k in ('arm', 'status', 'loss', 'peak_allocated_gib', 'error') if k in output}), flush=True)
        gc.collect(), torch.cuda.empty_cache()
    report['ended_unix'] = time.time()
    atomic_json(cli.out, report)
    if any(r['status'] != 'passed' for r in report['arms']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
