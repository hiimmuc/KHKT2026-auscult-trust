import json

import numpy as np
import pytest
import soundfile as sf

from scripts.source_audio_audit import align_oof, audit, diagnostics, source_rows
from src.processing.splits import cv_split


def test_source_partition_and_filenames(tmp_path):
    audio = tmp_path / 'audio_test_data'
    audio.mkdir()
    train = '101_1b1_Al_sc_Meditron'
    test = '102_1b1_Al_sc_Meditron'
    shared = '101_1b2_Al_sc_Meditron'
    (tmp_path / 'official_split.txt').write_text(f'{train}\ttrain\n{test}\ttest\n{shared}\ttest\n')
    (audio / f'{train}.txt').write_text('0\t1\t1\t0\n1\t2\t0\t1\n')
    rows, excluded = source_rows(tmp_path)
    assert [r['cycle_id'] for r in rows] == [f'{train}:0', f'{train}:1']
    assert [r['label'] for r in rows] == [1, 2]
    assert excluded == {'101'}
    (tmp_path / 'official_split.txt').write_text('../invalid.wav\ttrain\n')
    with pytest.raises(ValueError, match='filename'):
        source_rows(tmp_path)
    (tmp_path / 'official_split.txt').write_text(f'{train}\ttarget\n')
    with pytest.raises(ValueError, match='partition'):
        source_rows(tmp_path)
    (tmp_path / 'official_split.txt').write_text('101_../bad_Al_sc_Meditron\ttrain\n')
    with pytest.raises(ValueError, match='filename'):
        source_rows(tmp_path)


@pytest.mark.parametrize('start,end', [(-1, 1), (1, 1), (2, 1), (0, 3), (float('nan'), 1)])
def test_invalid_boundaries(start, end):
    assert not diagnostics(np.zeros((200, 1)), 100, start, end)['boundary_valid']


def test_native_diagnostics():
    result = diagnostics(np.zeros((200, 1)), 100, 0, 1)
    assert result['native_samples'] == 100
    assert result['silence_fraction'] == 1
    assert 'silence' in result['flags']
    result = diagnostics(np.ones((200, 1)), 100, 0, 1)
    assert result['clip_fraction'] == 1
    assert 'clipping' in result['flags']


def test_oof_alignment_rejects_training_or_target_metadata(tmp_path):
    rows = [dict(patient=str(i), stem=f'{i}_1b1_Al_sc_Meditron', device=0, label=0) for i in range(101, 110)]
    _, val = cv_split([r['patient'] for r in rows], [0]*len(rows), 3, 0)
    args = dict(selection='cv', test_fold='official', holdout_device=None, cv_folds=3, cv_fold=0)
    (tmp_path / 'train_args.json').write_text(json.dumps(args))
    arrays = {k: np.array([rows[i]['label' if k == 'labels' else k] for i in val])
              for k in ('labels', 'patient', 'stem', 'device')}
    arrays.update(epochs=np.array([66]), probs=np.full((1, len(val), 4), .25))
    np.savez(tmp_path / 'val_preds.npz', **arrays)
    actual, probs = align_oof(rows, tmp_path, 66)
    assert np.array_equal(actual, val)
    assert probs.shape == (len(val), 4)
    arrays['patient'][0] = '999'
    np.savez(tmp_path / 'val_preds.npz', **arrays)
    with pytest.raises(ValueError, match='metadata mismatch'):
        align_oof(rows, tmp_path, 66)
    args['holdout_device'] = 'Meditron'
    (tmp_path / 'train_args.json').write_text(json.dumps(args))
    with pytest.raises(ValueError, match='source-only'):
        align_oof(rows, tmp_path, 66)


def test_audit_cache_and_source_exclusion(tmp_path):
    folder, raw = tmp_path / 'processed', tmp_path / 'raw'
    audio = folder / 'audio_test_data'
    audio.mkdir(parents=True)
    raw.mkdir()
    stems = [f'{p}_1b1_Al_sc_Meditron' for p in range(101, 110)]
    (folder / 'official_split.txt').write_text('\n'.join(f'{s}\ttrain' for s in stems) + '\n101_2b1_Al_sc_Meditron\ttest\n')
    for stem in stems:
        sf.write(raw / f'{stem}.wav', np.zeros(1000), 1000)
        (raw / f'{stem}.txt').write_text('0\t1\t0\t0\n')
        for suffix in ('.wav', '.txt'):
            (audio / (stem+suffix)).write_bytes((raw / (stem+suffix)).read_bytes())
    rows, _ = source_rows(folder)
    units = []
    for seed in (0, 1):
        for fold in range(3):
            unit = tmp_path / f's{seed}f{fold}'
            unit.mkdir()
            args = dict(selection='cv', test_fold='official', holdout_device=None, cv_folds=3, cv_fold=fold, seed=seed)
            (unit / 'train_args.json').write_text(json.dumps(args))
            (unit / 'report.json').write_text('{}')
            _, val = cv_split([r['patient'] for r in rows], [0]*len(rows), 3, fold)
            arrays = {k: np.array([rows[i]['label' if k == 'labels' else k] for i in val])
                      for k in ('labels', 'patient', 'stem', 'device')}
            np.savez(unit / 'val_preds.npz', **arrays, epochs=[66], probs=np.full((1, len(val), 4), .25))
            units.append(unit)
    result = audit(folder, raw, units, tmp_path / 'out', 66, limit=3, snippets=True)
    assert result['source_cycles'] == 8
    assert result['excluded_shared_patients'] == ['101']
    assert len(list((tmp_path / 'out/snippets').glob('*.wav'))) == 3
    repeated = audit(folder, raw, units, tmp_path / 'out', 66, limit=3)
    assert repeated['input_fingerprint'] == result['input_fingerprint']
    assert result['confirmed_label_errors'] == 0
