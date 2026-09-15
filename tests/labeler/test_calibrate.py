"""Calibration numerics use hand-worked blocks and independent held-out draws."""
import numpy as np
import pytest

from labeler import calibrate
from labeler.validate import binary_metrics


def test_prior_shift_identity_odds_and_clipping():
    p = np.array([-1, 0, .2, .5, 1, 2])
    np.testing.assert_allclose(calibrate.prior_shift(p, from_prevalence=.1,
                                                  to_prevalence=.1),
                               np.clip(p, 1e-6, 1 - 1e-6))
    # Target odds / source odds = 9; sigmoid(log(9)) = .9.
    assert calibrate.prior_shift(.5, from_prevalence=.1,
                                to_prevalence=.5) == pytest.approx(.9)


def test_pava_blocks_interpolation_and_round_trip():
    scores = np.arange(8, dtype=float)
    truth = [0, 1, 1, 0, 0, 1, 0, 1]
    fitted = calibrate.IsotonicMap.fit(scores, truth)
    # Blocks: [0], [1,1,0,0,1,0] (mean 1/2), [1].
    np.testing.assert_allclose(fitted.y, [0, .5, .5, .5, .5, .5, .5, 1])
    assert np.all(np.diff(fitted.y) >= 0)
    np.testing.assert_allclose(fitted.apply([-1, .5, 6.5, 9]), [0, .25, .75, 1])
    restored = calibrate.IsotonicMap.from_dict(fitted.to_dict())
    np.testing.assert_array_equal(restored.apply(scores), fitted.apply(scores))
    assert restored.n_fit == 8 and restored.prevalence_fit == .5


def test_tied_scores_are_pooled_before_pava():
    fitted = calibrate.fit_isotonic([0, 0, 1, 2], [1, 0, 0, 1])
    np.testing.assert_array_equal(fitted.x, [0, 1, 2])
    np.testing.assert_allclose(fitted.y, [1/3, 1/3, 1])


def test_shifted_sample_calibrates_on_held_out_draws():
    rng = np.random.default_rng(19)
    score = rng.uniform(.01, .3, 100000)
    probability = 6 * score / (1 + 5 * score)
    truth = rng.binomial(1, probability)
    fitted = calibrate.fit_isotonic(score[:50000], truth[:50000])
    assert binary_metrics(score[50000:], truth[50000:])["ece"] > .2
    assert binary_metrics(fitted.apply(score[50000:]), truth[50000:])["ece"] < .02


@pytest.mark.parametrize("scores,truth", [([], []), ([np.nan], [0]), ([0], [2]),
                                          ([0, 1], [0])])
def test_invalid_fitting_rows_are_rejected(scores, truth):
    with pytest.raises(ValueError):
        calibrate.fit_isotonic(scores, truth)


@pytest.mark.parametrize("scores, expected", [([.2, .2], [.5, .5, .5]),
                                               ([0., 1.], [0., .2, 1.])])
def test_apply_preserves_nonfinite_inputs(scores, expected):
    fitted = calibrate.fit_isotonic(scores, [0, 1])
    got = fitted.apply([np.nan, np.inf, -np.inf, -1., .2, 2.])
    assert np.isnan(got[:3]).all()
    np.testing.assert_allclose(got[3:], expected)
    assert np.isnan(fitted.apply(np.nan))
