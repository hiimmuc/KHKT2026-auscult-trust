"""Rank source-only breath cycles for human listening without changing labels."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from src.processing.icbhi import CLASSES, DEVICES
from src.processing.splits import cv_split


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_rows(folder):
    """Keep official train recordings and separately identify cross-side patients."""
    folder = Path(folder)
    split = {}
    sides = {}
    for line in (folder / 'official_split.txt').read_text().splitlines():
        stem, side = line.split()
        fields = stem.split('_')
        if (len(fields) != 5 or not fields[0].isdigit() or fields[-1] not in DEVICES
                or not all(field.isalnum() for field in fields[:4])):
            raise ValueError(f'Invalid ICBHI filename: {stem}')
        if side not in ('train', 'test') or stem in split:
            raise ValueError(f'Invalid/duplicate official partition: {stem} {side}')
        split[stem] = side
        sides.setdefault(fields[0], set()).add(side)
    shared = {p for p, parts in sides.items() if len(parts) > 1}
    rows = []
    for stem in sorted(s for s, side in split.items() if side == 'train'):
        annotation = folder / 'audio_test_data' / f'{stem}.txt'
        table = np.loadtxt(annotation, ndmin=2)
        if table.shape[1] != 4 or not np.isfinite(table).all():
            raise ValueError(f'Invalid annotation table: {annotation}')
        for index, (start, end, crackle, wheeze) in enumerate(table):
            if crackle not in (0, 1) or wheeze not in (0, 1):
                raise ValueError(f'Invalid label: {annotation}:{index + 1}')
            rows.append(dict(cycle_id=f'{stem}:{index}', stem=stem, patient=stem.split('_')[0],
                             device=DEVICES.index(stem.split('_')[-1]), cycle_index=index,
                             annotation_line=index + 1, start=float(start), end=float(end),
                             label=int(crackle) + 2 * int(wheeze)))
    if not rows:
        raise ValueError('No official source cycles')
    return rows, shared


def align_oof(rows, unit, epoch):
    """Require complete held-out patient-CV metadata in original cycle order."""
    unit = Path(unit)
    args = json.loads((unit / 'train_args.json').read_text())
    if (args['selection'] != 'cv' or args['test_fold'] != 'official'
            or args.get('holdout_device') or args['cv_folds'] != 3):
        raise ValueError(f'Not source-only three-fold CV: {unit}')
    train, val = cv_split([r['patient'] for r in rows], [r['device'] for r in rows], 3, args['cv_fold'])
    if {rows[i]['patient'] for i in train} & {rows[i]['patient'] for i in val}:
        raise ValueError('Training and validation patients overlap')
    with np.load(unit / 'val_preds.npz', allow_pickle=False) as arrays:
        for key in ('labels', 'patient', 'stem', 'device'):
            expected = np.array([rows[i]['label' if key == 'labels' else key] for i in val])
            if not np.array_equal(arrays[key], expected):
                raise ValueError(f'OOF metadata mismatch: {unit} {key}')
        matches = np.flatnonzero(arrays['epochs'] == epoch)
        if len(matches) != 1:
            raise ValueError(f'Epoch {epoch} missing or duplicated: {unit}')
        probs = arrays['probs'][matches[0]].astype(float)
    if (probs.shape != (len(val), 4) or not np.isfinite(probs).all()
            or np.any(probs < 0) or np.any(probs > 1)
            or not np.allclose(probs.sum(1), 1, atol=1e-5)):
        raise ValueError(f'Invalid probability array: {unit}')
    return val, probs


def diagnostics(audio, rate, start, end):
    """Measure native PCM; thresholds are listening heuristics, never label findings."""
    duration = len(audio) / rate
    if not all(math.isfinite(x) for x in (start, end)) or not 0 <= start < end <= duration:
        return dict(boundary_valid=False, duration_seconds=end-start,
                    flags='invalid_boundary', clip_fraction='', silence_fraction='',
                    rms='', edge_jump_rms='', native_samples=0)
    first, last = int(start * rate), int(end * rate)
    chunk = audio[first:last]
    if not len(chunk):
        return dict(boundary_valid=False, duration_seconds=end-start,
                    flags='empty_boundary', clip_fraction='', silence_fraction='',
                    rms='', edge_jump_rms='', native_samples=0)
    rms = float(np.sqrt(np.mean(chunk ** 2)))
    clipped = float(np.mean(np.abs(chunk) >= .999))
    silent = float(np.mean(np.abs(chunk) < 1e-4))
    jumps = [float(np.max(np.abs(chunk[0]))), float(np.max(np.abs(chunk[-1])))]
    if first > 0:
        jumps.append(float(np.max(np.abs(audio[first] - audio[first-1]))))
    if last < len(audio):
        jumps.append(float(np.max(np.abs(audio[last] - audio[last-1]))))
    edge = max(jumps) / max(rms, 1e-12)
    flags = []
    if end-start < .5 or end-start > 8:
        flags.append('unusual_duration')
    if clipped >= .001:
        flags.append('clipping')
    if silent >= .5 or rms < 1e-4:
        flags.append('silence')
    if edge > 8:
        flags.append('edge_discontinuity')
    return dict(boundary_valid=True, duration_seconds=end-start, flags=';'.join(flags),
                clip_fraction=clipped, silence_fraction=silent, rms=rms,
                edge_jump_rms=edge, native_samples=len(chunk))


def audit(folder, raw, units, output, epoch, limit=50, snippets=False):
    if not 1 <= limit <= 100 or len(units) < 2:
        raise ValueError('Require at least two OOF members and 1..100 review candidates')
    rows, shared = source_rows(folder)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    def hashed(path):
        path = Path(path).resolve()
        hashes[str(path)] = sha256(path)
    hashed(__file__)
    for name in ('src/processing/splits.py', 'src/processing/icbhi.py'):
        hashed(Path(__file__).resolve().parents[1] / name)
    hashed(Path(folder) / 'official_split.txt')
    members = {}
    for unit in map(Path, units):
        args = json.loads((unit / 'train_args.json').read_text())
        seed = args['seed']
        if args['cv_fold'] not in (0, 1, 2):
            raise ValueError('Invalid source CV fold')
        if (seed, args['cv_fold']) in members:
            raise ValueError('Duplicate seed/fold')
        idx, probs = align_oof(rows, unit, epoch)
        members[seed, args['cv_fold']] = (idx, probs)
        for name in ('train_args.json', 'val_preds.npz', 'report.json'):
            hashed(unit / name)
    seeds = sorted({key[0] for key in members})
    if len(seeds) < 2 or set(members) != {(s, f) for s in seeds for f in range(3)}:
        raise ValueError('Require all three held-out folds for each of at least two seeds')
    predictions = np.empty((len(seeds), len(rows), 4))
    for si, seed in enumerate(seeds):
        for fold in range(3):
            idx, probs = members[seed, fold]
            predictions[si, idx] = probs
    # Hash every source input before reusing cached measurements.
    eligible = [i for i, row in enumerate(rows) if row['patient'] not in shared]
    paths = {}
    for stem in sorted({rows[i]['stem'] for i in eligible}):
        paths[stem] = Path(raw) / f'{stem}.wav'
        for suffix in ('.wav', '.txt'):
            original = paths[stem].with_suffix(suffix)
            processed = Path(folder) / 'audio_test_data' / original.name
            hashed(original)
            if sha256(processed) != hashes[str(original.resolve())]:
                raise ValueError(f'Processed input differs from raw reference: {processed}')
    fingerprint = hashlib.sha256(json.dumps(dict(hashes=hashes, epoch=epoch), sort_keys=True).encode()).hexdigest()
    cache_path = output / 'diagnostics.json'
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    measured = cache.get('cycles', {}) if cache.get('input_fingerprint') == fingerprint else {}
    records = []
    for stem, path in paths.items():
        indices = [i for i in eligible if rows[i]['stem'] == stem]
        missing = [i for i in indices if rows[i]['cycle_id'] not in measured]
        if missing:
            audio, rate = sf.read(path, dtype='float64', always_2d=True)
            if not np.isfinite(audio).all():
                raise ValueError(f'Nonfinite raw audio: {path}')
            for i in missing:
                row = rows[i]
                measured[row['cycle_id']] = diagnostics(audio, rate, row['start'], row['end'])
        for i in indices:
            row = rows[i]
            probs = predictions[:, i].mean(0)
            pred = int(probs.argmax())
            disagree = float(np.mean(predictions[:, i].argmax(1) != row['label']))
            flags = measured[row['cycle_id']]['flags']
            priority = (10 if not measured[row['cycle_id']]['boundary_valid'] else 0) + bool(flags) + disagree + float(probs.max() - probs[row['label']])
            records.append(dict(**row, **measured[row['cycle_id']], wav=str(path.resolve()),
                                annotation=str(path.with_suffix('.txt').resolve()),
                                class_name=CLASSES[row['label']], predicted_class=CLASSES[pred],
                                confidence=float(probs.max()), label_probability=float(probs[row['label']]),
                                model_label_disagreement=disagree,
                                seed_prediction_disagreement=float(len(set(predictions[:, i].argmax(1))) > 1),
                                priority=priority, review_status='unreviewed_heuristic', confirmed_label_error='',
                                reviewer_notes='', snippet=''))
    cache_path.write_text(json.dumps(dict(input_fingerprint=fingerprint, cycles=measured), indent=2) + '\n')
    records.sort(key=lambda r: (-r['priority'], r['cycle_id']))
    selected = records[:limit]
    if snippets:
        destination = output / 'snippets'
        destination.mkdir(exist_ok=True)
        for row in selected:
            if not row['boundary_valid']:
                continue
            with sf.SoundFile(row['wav']) as stream:
                first = max(0, int((row['start']-.25) * stream.samplerate))
                last = min(len(stream), int((row['end']+.25) * stream.samplerate))
                stream.seek(first)
                audio = stream.read(last-first, dtype='float32', always_2d=True)
                name = destination / f"{row['stem']}_cycle{row['cycle_index']}.wav"
                sf.write(name, audio, stream.samplerate, subtype='FLOAT')
                row['snippet'] = str(name.resolve())
    for name, content in (('all_cycles.csv', records), ('review.csv', selected)):
        with (output / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(content)
    summary = dict(scope='official training recordings only; cross-side patients excluded; no target audio or labels loaded',
                   status='human_listening_pending', confirmed_label_errors=0, epoch=epoch, seeds=seeds,
                   source_cycles=len(records), source_patients=len({r['patient'] for r in records}),
                   excluded_shared_patients=sorted(shared), candidates=len(selected),
                   invalid_boundaries=sum(not r['boundary_valid'] for r in records),
                   heuristic_flag_counts={flag: sum(flag in r['flags'].split(';') for r in records)
                                          for flag in ('invalid_boundary', 'unusual_duration', 'clipping', 'silence', 'edge_discontinuity')},
                   input_fingerprint=fingerprint, input_sha256=hashes,
                   interpretation='Ranking and fixed-epoch OOF confidence are heuristic; disagreement does not establish a label error. Native unnormalised snippets include 0.25s context. Raw audio and annotations are unchanged.')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-folder', type=Path, required=True)
    parser.add_argument('--raw-folder', type=Path, required=True)
    parser.add_argument('--units', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epoch', type=int, required=True, help='Previously selected source-CV epoch')
    parser.add_argument('--limit', type=int, default=50)
    parser.add_argument('--snippets', action='store_true')
    args = parser.parse_args()
    result = audit(args.data_folder, args.raw_folder, args.units, args.output, args.epoch, args.limit, args.snippets)
    print(json.dumps({k: v for k, v in result.items() if k != 'input_sha256'}, indent=2))


if __name__ == '__main__':
    main()
