"""Recording-level probabilities from window predictions."""

import numpy as np


def _aggregate(data, columns, agg, binary_labels):
    reducers = {"mean": np.mean, "max": np.max}
    if agg not in reducers:
        raise ValueError(f"Unknown recording aggregation: {agg}")
    stems, inverse = np.unique(data["stem"], return_inverse=True)
    abnormal = np.array(
        [reducers[agg](1 - data["probs"][inverse == k][:, 0]) for k in range(len(stems))]
    )
    first = np.array([np.flatnonzero(inverse == k)[0] for k in range(len(stems))])
    labels = data["label"][first]
    return {
        "p": np.stack([1 - abnormal, abnormal], 1),
        "y": (labels > 0).astype(int) if binary_labels else labels,
        **{key: data[key][first] for key in columns},
    }


def icbhi_recordings(data, agg="mean"):
    return _aggregate(data, ("patient", "device"), agg, binary_labels=False)


def kauh_recordings(data, agg="mean"):
    return _aggregate(data, ("patient", "filter"), agg, binary_labels=True)
