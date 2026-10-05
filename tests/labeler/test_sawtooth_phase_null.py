"""The phase test must include its own search for regular subsequences."""

import numpy as np

from labeler.sawtooth.physics import core_relaxation_phases


def test_group_search_does_not_inflate_uniform_refractory_null_rejection():
    # Exact fixed-count/span refractory null, independent of production's
    # shuffled-time generator. Gap grouping must be included in calibration.
    rng = np.random.default_rng(23)
    accepted = 0
    for _ in range(200):
        free = 1.0 - 19 * 0.00515
        times = np.r_[0.0, np.sort(rng.uniform(0.0, free, 18)), free]
        times += np.arange(20) * 0.00515
        accepted += bool(core_relaxation_phases(times, [(-0.1, 1.1)]))
    # At alpha=.05 the expectation is ten rejects. This generous bound catches
    # the old ~27% rejection while permitting ordinary Monte Carlo variation.
    assert accepted <= 24
