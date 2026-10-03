"""The trivial feature cannot leak adjacent-bin levels or density."""

import numpy as np
import pytest

from labeler.elm import feature, inputs, labels


def test_logistic_fit_separates_bursts_and_regularizes():
    x = np.array([0, 0, 1, 1], dtype=float)
    y = np.array([0, 0, 1, 1])
    theta = feature.fit_logistic(x, y)
    p = feature.logistic_scores(theta, x)
    assert np.all(p[:2] < 0.5) and np.all(p[2:] > 0.5)
    assert 0 < theta[0] < 2  # finite L2-regularized solution, despite separation


def test_burst_feature_is_level_invariant_and_local():
    x = np.zeros((inputs.N_CHANNELS, 1500), dtype=np.float32)
    x[inputs.VALID] = 1
    x[1, 503] = 2
    bins = labels.Bins(
        np.array([0.0]), np.array([1]), np.array(["crowd"]), np.array([0])
    )
    assert feature.bin_feature(x, bins)[0] == 3.0
    x[:3] += 7
    x[6:10] = 100
    x[0, 1200] = 100
    assert feature.bin_feature(x, bins)[0] == 3.0


def test_uncovered_feature_is_rejected():
    x = np.ones((inputs.N_CHANNELS, 1000), dtype=np.float32)
    x[inputs.VALID, 501] = 0
    bins = labels.Bins(
        np.array([0.0]), np.array([1]), np.array(["crowd"]), np.array([0])
    )
    with pytest.raises(ValueError, match="fully covered"):
        feature.bin_feature(x, bins)
