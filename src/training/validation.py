"""Source-patient recording predictions from ordered cycle validation outputs."""
import numpy as np
from .dataset import validation_metadata
from .recording_bags import source_indices
from .reporting import per_device_report


def recording_from_cycles(source, probabilities, max_cycles=8):
    patients, stems = validation_metadata(source)
    dataset, indices = source_indices(source)
    labels, devices = np.asarray(dataset.labels)[indices], np.asarray(dataset.devices)[indices]
    p = np.asarray(probabilities)
    if len(p) != len(indices) or max_cycles < 1:
        raise ValueError('Cycle predictions must align with the source validation subset')
    ids = np.unique(stems)
    output, y, dev, pat = [], [], [], []
    for stem in ids:
        rows = np.flatnonzero(stems == stem)
        if len(set(patients[rows])) != 1 or len(set(devices[rows])) != 1:
            raise ValueError('Recording must belong to one patient and device')
        chosen = rows[np.linspace(0, len(rows) - 1, min(len(rows), max_cycles)).round().astype(int)]
        abnormal = float(np.max(1 - p[chosen, 0]))
        output.append([1 - abnormal, abnormal])
        y.append(int(np.any(labels[rows] != 0)))
        dev.append(devices[rows[0]])
        pat.append(patients[rows[0]])
    output, y, dev = np.asarray(output), np.asarray(y), np.asarray(dev)
    pred = output.argmax(1)
    return per_device_report(y, pred, dev, 2), pred, output, y, dev, ids, np.asarray(pat)
