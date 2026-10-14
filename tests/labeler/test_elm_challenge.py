"""The DIII-D ELM detection challenge adapter: submissions and our views of the key."""

import numpy as np
import pytest

from labeler.elm import challenge


def test_a_submission_round_trips_its_spans_and_is_bracketed_by_zero():
    record = challenge.submission(
        7, [(10.0, 12.0), (30.0, 31.0)], (0.0, 100.0), method="x"
    )
    assert record["device"] == "DIII-D" and record["shot"] == 7
    assert record["event_flag"][0] == 0 and record["event_flag"][-1] == 0
    assert record["time"][0] <= 0.0 and record["time"][-1] >= 100.0
    assert set(record["event_type"][1:-1]) == {challenge.DETECTION_ONLY_TYPE}
    assert np.allclose(
        challenge.flag_spans(record),
        [(10.0, 12.0 + challenge.EDGE_MS), (30.0, 31.0 + challenge.EDGE_MS)],
    )


def test_no_detection_is_all_zero_and_overlapping_spans_are_separated():
    empty = challenge.submission(7, np.zeros((0, 2)), (0.0, 50.0), method="none")
    assert empty["event_flag"] == [0, 0] and empty["time"] == [0.0, 50.0]
    apart = challenge.separate([(5.0, 9.0), (1.0, 6.0)])
    assert apart[0, 0] == 1.0 and apart[0, 1] < apart[1, 0] == 5.0
    with pytest.raises(ValueError):
        challenge.separate([(2.0, 1.0)])


def test_resample_brings_a_fast_record_to_the_clock_grid_and_keeps_gaps():
    t = np.arange(0.0, 10.0, 0.02)
    y = t.copy()
    y[(t > 4.0) & (t < 6.0)] = np.nan
    grid, out = challenge.resample(t, y)
    assert np.isclose(np.diff(grid).mean(), challenge.CLOCK_STEP_MS)
    assert np.allclose(out[np.isfinite(out)], grid[np.isfinite(out)])
    assert np.isnan(out[(grid > 4.2) & (grid < 5.8)]).all()
    same, kept = challenge.resample(grid, out)
    assert same is grid and kept is out


CASE = {
    "shot": 1,
    "tstart": 0.0,
    "tend": 100.0,
    "events": [(10.0, 12.0, 1), (40.0, 41.0, 2), (60.0, 62.0, 3), (80.0, 81.0, 5)],
}


def test_onset_match_counts_elms_and_leaves_out_detections_beside_other_events():
    onsets, others = challenge.key_onsets(CASE)
    assert onsets.tolist() == [10.0, 40.0] and others.tolist() == [[60.0, 62.0]]
    found = [(10.5, 11.0), (59.0, 60.0), (80.0, 81.0), (150.0, 151.0)]
    got = challenge.onset_match(CASE, found, 2.0)
    # one ELM found, one missed; 59 ms starts within 2 ms of the uncertain event (left
    # out), the pellet-only 80 ms is a false positive, 150 ms is outside the window
    assert (got["tp"], got["fn"], got["fp"], got["excluded"]) == (1, 1, 1, 1)
    assert got["offsets_ms"] == [0.5]


def test_bins_touched_and_metrics_score_only_the_assessed_bins():
    times = np.arange(0.0, 200.0, 50.0)
    touched = challenge.bins_touched([(10.0, 12.0), (99.0, 101.0)], times)
    assert touched.tolist() == [True, True, True, False]
    assert not challenge.bins_touched(np.zeros((0, 2)), times).any()
    truth = np.array([1.0, np.nan, 0.0, 0.0])
    got = challenge.bin_metrics(truth, touched)
    assert (got["bins"], got["tp"], got["fp"], got["fn"], got["tn"]) == (3, 1, 1, 0, 1)
    assert got["f1"] == pytest.approx(2 / 3)
    assert np.isnan(challenge.bin_metrics_from_counts(0, 0, 0)["f1"])


def test_crowding_gap_and_signed_offsets_of_the_nearest_detection():
    onsets, gap = challenge.nearest_other_gap_ms([30.0, 10.0, 14.0, 100.0])
    assert onsets.tolist() == [10.0, 14.0, 30.0, 100.0]
    assert gap.tolist() == [4.0, 4.0, 16.0, 70.0]
    assert challenge.nearest_other_gap_ms([5.0])[1].tolist() == [1e9]
    got = challenge.nearest_signed_offsets_ms([10.0, 14.0, 100.0], [9.0, 60.0], 20.0)
    # the detection at 9 ms serves both close onsets; 100 ms has none within 20 ms
    assert got[:2].tolist() == [-1.0, -5.0] and np.isnan(got[2])
    assert np.isnan(challenge.nearest_signed_offsets_ms([10.0], [], 20.0)).all()
