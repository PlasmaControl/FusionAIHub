"""Nonfinite detector outputs cannot masquerade as valid checkpoint metrics."""

import numpy as np
import pytest

from labeler.elm import score


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_scores_are_not_ranked_or_thresholded(bad):
    truth = np.array([0, 1, 0, 1])
    values = np.array([0.1, 0.9, bad, 0.8])
    assert np.isnan(score.roc_auc(truth, values))
    assert np.isnan(score.average_precision(truth, values))
    threshold, f1 = score.best_threshold(truth, values)
    assert np.isnan(threshold) and np.isnan(f1)
