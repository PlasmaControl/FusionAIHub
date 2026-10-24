"""`labeler.elm.spikes`: reviewed train onsets, the spike loss and the merit costs."""

import numpy as np
import pandas as pd
import pytest
import torch

from labeler.elm import inst, spikes


def spans(rows):
    return pd.DataFrame(rows, columns=["t_start", "t_end", "kind"])


def test_merge_keeps_the_first_of_close_onsets():
    assert spikes.merge([20.0, 10.0, 11.2, 12.9, 13.1]).tolist() == [10.0, 13.1, 20.0]


def test_accepted_onsets_are_the_jump_onsets_inside_his_rows():
    s = spans(
        [
            (0, 100, "absent"),
            (100, 110, "non_crowd"),
            (100, 300, "crowd"),
            (150, 160, "non_crowd"),
        ]
    )
    kept, n = spikes.accepted_onsets(s, [50.0, 101.0, 102.5, 105.0, 130.0, 151.0])
    assert kept.tolist() == [101.0, 105.0, 151.0]
    assert n == {"rows": 2, "rows_without_onset": 0, "onsets_in_rows": 4, "kept": 3}


def test_review_onset_target_marks_onsets_and_guarded_absent_time_only():
    s = spans(
        [
            (0, 100, "absent"),
            (100, 300, "crowd"),
            (150, 160, "non_crowd"),
            (300, 400, "uncertain"),
        ]
    )
    target = spikes.review_onset_target(s, [151.0], int(inst.cell_of(400.0)) + 1)
    c = int(inst.cell_of(151.0))
    assert target.heat[c] == 1.0 and target.known[c] and target.weight[c] == 1.0
    assert target.known[inst.cell_of(50.0)] and not target.known[inst.cell_of(97.0)]
    assert not target.known[inst.cell_of(200.0)]
    assert not target.known[inst.cell_of(350.0)]


def test_spike_loss_rewards_separate_peaks_over_a_plateau():
    anchor = torch.zeros(1, 400, dtype=torch.bool)
    anchor[0, [100, 160]] = True
    spiky = torch.full((1, 400), -6.0)
    spiky[0, [100, 160]] = 6.0
    plateau = torch.full((1, 400), -6.0)
    plateau[0, 90:175] = 6.0
    assert float(spikes.spike_loss(spiky, anchor)) < 0.01
    assert float(spikes.spike_loss(plateau, anchor)) > 5.0


def test_spike_loss_takes_a_peak_a_few_cells_off_and_a_side_beyond_the_crop():
    anchor = torch.zeros(1, 100, dtype=torch.bool)
    anchor[0, 3] = True
    logits = torch.full((1, 100), -6.0, requires_grad=True)
    with torch.no_grad():
        logits[0, 6] = 6.0
    loss = spikes.spike_loss(logits, anchor)
    loss.backward()
    assert float(loss.detach()) < 0.01
    assert torch.isfinite(logits.grad).all()


def test_spike_loss_without_anchors_is_zero():
    logits = torch.zeros(2, 50, requires_grad=True)
    loss = spikes.spike_loss(logits, torch.zeros(2, 50, dtype=torch.bool))
    loss.backward()
    assert float(loss.detach()) == 0.0


def test_merit_gain_uses_the_published_costs():
    assert spikes.GAIN_TP == pytest.approx(5.0)
    assert spikes.COST_FP == pytest.approx(9.5)
    assert spikes.merit_gain(10, 2, 1.0) == pytest.approx(10 * 6.0 - 2 * 10.5)
    assert spikes.GAIN_TP_SMALL == pytest.approx(0.5)
    assert spikes.merit_gain(0, 0, 1.0, tp_small=2) == pytest.approx(2 * 1.5)


def test_prices_by_reported_type():
    assert spikes.prices(2) == pytest.approx((7.0, 4.0, 3.0))
    assert spikes.prices(1) == pytest.approx((12.0, 1.0, 10.0))
    assert spikes.merit_gain(10, 2, 1.0, tp_small=4, reported=2) == pytest.approx(
        10 * 8.0 + 4 * 5.0 - 2 * 4.0
    )


def test_report_type_takes_the_lower_expected_cost():
    # a certain ELM: large pays from P(large) = 3/8 on (5.0 P(large) > 3.0 P(small))
    q = np.array([0.0, 0.37, 0.38, 1.0])
    assert spikes.report_type(np.zeros(4), q, 1 - q).tolist() == [2, 2, 1, 1]
    # a likely false detection stays small even when it would be a large ELM
    assert spikes.report_type([0.5], [0.5], [0.0]).tolist() == [2]
    # an exact tie goes to the first type
    assert spikes.report_type([0.0], [0.375], [0.625]).tolist() == [2]


def test_in_crowd_marks_the_onsets_inside_a_crowd_row():
    s = spans([(0, 100, "absent"), (100, 300, "crowd"), (150, 160, "non_crowd")])
    assert spikes.in_crowd([50.0, 151.0, 300.0], s).tolist() == [False, True, False]


def test_judge_sorts_unmatched_detections():
    s = spans(
        [
            (0, 100, "absent"),
            (100, 300, "crowd"),
            (150, 160, "non_crowd"),
            (200, 210, "non_crowd"),
        ]
    )
    found = [50.0, 98.0, 151.5, 153.5, 205.0, 230.0]
    j = spikes.judge(found, [151.0, 201.0], s)
    assert (j["tp"], j["fn"]) == (1, 1)
    assert (j["fp"], j["fp_absent"], j["fp_near"]) == (2, 1, 1)
    assert (j["empty"], j["other"]) == (1, 2)
    assert j["matched_onsets"].tolist() == [True, False]
    assert j["kind"].tolist() == ["fp", "other", "tp", "fp", "other", "empty"]
