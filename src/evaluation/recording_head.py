"""Frozen learned recording-head probabilities over complete window groups."""
import numpy as np

from src.training.recording_bags import predict_recording


def recording_predictions(images, windows, model, classifier, curves=None):
    """Return binary max-MIL predictions, grouping KAUH filters independently.

    Images and optional spectral gain curves use the same units and ordering as
    predict_images. No cycle probability aggregation substitutes for a missing head.
    """
    head = getattr(classifier, 'module', classifier)
    if not callable(getattr(head, 'recording_logits', None)):
        raise ValueError('Recording predictions require a trained recording head')
    images = np.asarray(images)
    if len(images) != len(windows) or images.ndim not in (3, 4):
        raise ValueError('Window metadata must align with spectrogram images')
    if curves is not None and np.asarray(curves).shape != (len(images), images.shape[-1]):
        raise ValueError('Spectral curves must align with windows and frequency bins')
    groups = ['stem', 'filter'] if 'filter' in windows.columns else ['stem']
    output = {f'recording_{key}': [] for key in ['probs', 'stem', 'patient', 'label']}
    for key in ['filter', 'device']:
        if key in windows.columns:
            output[f'recording_{key}'] = []
    table = windows.reset_index(drop=True)
    for _, rows in table.groupby(groups, sort=True):
        for key in ['patient', 'label', *[k for k in ['filter', 'device'] if k in rows]]:
            if rows[key].nunique(dropna=False) != 1:
                raise ValueError(f'Recording windows disagree on {key}')
        indices = rows.index.to_numpy()
        x = images[indices].copy()
        if x.ndim == 3:
            x = x[:, None]
        if curves is not None:
            x += np.asarray(curves)[indices, None, None, :]
        output['recording_probs'].append(predict_recording(model, classifier, x).cpu().numpy())
        for key in output:
            if key != 'recording_probs':
                output[key].append(rows.iloc[0][key.removeprefix('recording_')])
    return {key: np.asarray(value) for key, value in output.items()}
