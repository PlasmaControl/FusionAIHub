"""Synthetic checks of upstream shot rules and censoring-weighted ranking."""
import numpy as np
import pytest

from labelmaker.alarm import (
    any_row_call,
    horizon_integral,
    ipcw_auc,
    km_censoring,
    pool_rates,
    shot_alarm,
)
from labelmaker.validate import binary_metrics


@pytest.mark.parametrize(('onset', 'risk', 'verdict'), [
    (1.5, [1, 1], 'TP'), (1.5, [1, 0], 'FN'),
    (None, [0, 1], 'FP'), (None, [1, 0], 'TN'),
])
def test_final_verdict(onset, risk, verdict):
    got = shot_alarm([0, 1], risk, [1, 1], threshold=0.5, onset_s=onset)
    assert got.verdict == verdict
    assert got.alarm == bool(risk[-1])
    assert got.warning_time_s == (1.5 if verdict == 'TP' else None)


def test_warning_and_filtering_in_time_order():
    got = shot_alarm([1, 0, 2, np.nan], [0.5, 0, 0, 1], [1, 1, 0, 1],
                     threshold=0.5, onset_s=1.5)
    assert got.warning_time_s == 0.5
    assert got.n_rows == 2


def test_excursions_use_absolute_reversion_time_and_exclude_final_run():
    got = shot_alarm([0, 1, 1.3, 2, 2.5, 3], [0, 1, 0, 1, 0, 1], [1]*6,
                     threshold=0.5, onset_s=4)
    assert (got.jumps, got.n_excursions) == (1, 2)
    assert got.warning_time_s == 1


def test_any_row_differs_from_final():
    assert any_row_call([0, 1, 0], [1]*3, threshold=0.5)
    assert not shot_alarm([0, 1, 2], [0, 1, 0], [1]*3,
                          threshold=0.5, onset_s=None).alarm
    assert not any_row_call([np.nan, 1], [1, 0], threshold=0.5)


def test_empty_shot_is_not_a_quiet_verdict():
    with pytest.raises(ValueError, match='no valid'):
        shot_alarm([0], [np.nan], [1], threshold=0.5, onset_s=None)


def test_rates_and_horizon_integral():
    got = pool_rates(['TP', 'FN', 'TN', 'FP', 'TN'])
    assert got['fpr'] == pytest.approx(1/3)
    assert got['fnr'] == 0.5
    assert got['counts'] == {'TP': 1, 'FN': 1, 'TN': 2, 'FP': 1}
    assert pool_rates([])['fpr'] is None
    assert horizon_integral([0.25, 0.5, 1], [0.2]*3) == pytest.approx(0.15)


def test_ipcw_without_censoring_equals_plain_auc():
    t = np.array([0.1, 0.2, 0.5, 1, 2, 3])
    score = np.array([0.8, 0.4, 0.4, 0.5, 0.1, 0.2])
    assert ipcw_auc(t, np.ones(6), score, 0.5) == pytest.approx(
        binary_metrics(score, t <= 0.5)['auroc'])


def test_ipcw_six_rows_with_censoring_and_score_tie():
    # G(1)=1, G(3)=4/5 after censor at 2; case weights 1 and 5/4.
    # Controls scores .2, .6, .8; cases .6 and .9 win 1.5 and 3 pairs.
    t = [1, 2, 3, 5, 6, 7]
    e = [1, 0, 1, 1, 0, 1]
    assert ipcw_auc(t, e, [.6, .1, .9, .2, .6, .8], 4) == pytest.approx(7/9)
    np.testing.assert_allclose(km_censoring(t, e)([0, 1, 2, 3]), [1, 1, .8, .8])


def test_km_event_before_censor_at_same_time():
    # Reverse KM removes events from the risk set before tied censorings.
    np.testing.assert_allclose(km_censoring([1, 1, 2], [1, 0, 1])([0, 1, 2]),
                               [1, .5, .5])


@pytest.mark.parametrize(('t', 'e'), [([2, 3], [1, 1]), ([.1, .2], [1, 1]),
                                       ([.1, .2], [0, 0])])
def test_ipcw_missing_cases_or_controls(t, e):
    assert ipcw_auc(t, e, [.2, .8], 1) is None


def test_empty_censoring_sample_has_unit_survival():
    np.testing.assert_array_equal(km_censoring([], [])([0, 1]), [1, 1])
