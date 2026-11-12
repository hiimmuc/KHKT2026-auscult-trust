import numpy as np
import pytest

from auscult_trust.evaluation.recordings import icbhi_recordings, kauh_recordings


@pytest.mark.parametrize("agg,probabilities", [("mean", [[0.6, 0.4], [0.3, 0.7]]),
                                              ("max", [[0.6, 0.4], [0.1, 0.9]])])
def test_recordings_aggregate_windows_and_keep_sorted_metadata(agg, probabilities):
    data = {"stem": np.array(["b", "a", "b"]), "patient": np.array([2, 1, 2]),
            "device": np.array([3, 0, 3]), "filter": np.array(["high", "low", "high"]),
            "label": np.array([3, 0, 3]),
            "probs": np.array([[0.5, 0.2, 0.2, 0.1], [0.6, 0.1, 0.1, 0.2], [0.1, 0.3, 0.3, 0.3]])}
    icbhi, kauh = icbhi_recordings(data, agg), kauh_recordings(data, agg)
    for result in (icbhi, kauh):
        np.testing.assert_allclose(result["p"], probabilities)
        np.testing.assert_array_equal(result["patient"], [1, 2])
    np.testing.assert_array_equal(icbhi["y"], [0, 3])
    np.testing.assert_array_equal(icbhi["device"], [0, 3])
    np.testing.assert_array_equal(kauh["y"], [0, 1])
    np.testing.assert_array_equal(kauh["filter"], ["low", "high"])


def test_unknown_recording_aggregation_is_rejected():
    with pytest.raises(ValueError, match="aggregation"):
        icbhi_recordings({}, "median")
