import numpy as np
import pandas as pd

from labeler.confinement.comparison import bin_probabilities, eligible_legacy_shots


def test_legacy_training_and_validation_shots_cannot_be_matched_test():
    rows = pd.DataFrame({"shot": [1, 2, 3, 4], "split": ["test"] * 4})
    training = {"shots": {"train": [1], "val": [2]}}
    assert eligible_legacy_shots(rows, training) == {3, 4}


def test_only_whole_observed_legacy_bins_can_be_matched():
    # First bin starts late; last bin has a missing frame probability.
    probability = np.array([0.2] * 3 + [0.6] * 5 + [0.9] * 4 + [np.nan])
    assert bin_probabilities(probability, first=2) == {50: 0.6}
