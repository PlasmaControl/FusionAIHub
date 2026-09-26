"""Events: observed boundaries only, nearest-first matching within a tolerance."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from labeler.scoring import events
from labeler.scoring.events import boundaries, event_cells, match, within
from labeler.scoring.frames import NOT_OBSERVABLE, PRESENT, UNCERTAIN, Assessment
from labeler.scoring.stats import f1


def test_only_absent_present_changes_are_boundaries():
    read = Assessment(
        (0, 1000),
        ((100, 200, PRESENT), (200, 300, UNCERTAIN), (500, 600, 1), (600, 1000, 1)),
    )
    onsets, ends = boundaries(read)
    assert onsets.tolist() == [100, 500]  # 500-1000 runs to the window's edge
    assert ends.tolist() == []  # 200 leads into uncertain, not absent


def test_a_span_at_the_window_start_has_no_onset():
    onsets, ends = boundaries(Assessment((0, 1000), ((0, 100, PRESENT),)))
    assert onsets.tolist() == [] and ends.tolist() == [100]


def test_nearest_pairs_are_taken_first():
    m = match([0, 10], [6], 10)
    assert m.pairs.tolist() == [[1, 0]]
    assert m.unmatched_reference.tolist() == [0]
    assert m.offsets.tolist() == [4]


def test_counts_of_a_matching():
    m = match([100, 110], [104, 200], 10)
    assert m.pairs.tolist() == [[0, 0]]
    assert event_cells(m).tolist() == [1, 1, 1]  # tp, fp, fn


def test_the_tolerance_is_inclusive():
    assert len(match([0], [2.0], 2).pairs) == 1
    assert len(match([0], [2.0001], 2).pairs) == 0


def test_a_tie_goes_to_the_earlier_estimate():
    m = match([10], [8, 12], 5)
    assert m.pairs.tolist() == [[0, 0]]
    assert m.unmatched_estimate.tolist() == [1]


def test_empty_sides_leave_everything_unmatched():
    m = match([], [1, 2], 5)
    assert len(m.pairs) == 0 and m.unmatched_estimate.tolist() == [0, 1]
    assert event_cells(match([3], [], 5)).tolist() == [0, 0, 1]


def test_points_are_scored_inside_their_window_only():
    assert within([999.9, 1000, 1500, 2000], (1000, 2000)).tolist() == [1000, 1500]


def test_onset_inside_reference_uncertain_is_excluded():
    ref = Assessment((0, 1000), ((400, 500, UNCERTAIN), (500, 800, PRESENT)))
    est = Assessment((0, 1000), ((450, 800, PRESENT),))
    got = events.event_matchings(ref, est, 50, method=True)
    onset = got["onset"]
    assert event_cells(onset).tolist() == [0, 0, 0]
    assert onset.estimate.tolist() == [450.0]
    assert onset.excluded_estimate.tolist() == [0]
    assert event_cells(got["end"]).tolist() == [1, 0, 0]


def test_boundaries_inside_reference_not_observable_are_excluded():
    ref = Assessment((0, 1000), ((0, 300, NOT_OBSERVABLE), (500, 800, PRESENT)))
    est = Assessment((0, 1000), ((100, 200, PRESENT), (500, 800, PRESENT)))
    got = events.event_matchings(ref, est, 10, method=True)
    for matching in got.values():
        assert event_cells(matching).tolist() == [1, 0, 0]
        assert matching.excluded_estimate.tolist() == [0]
        assert matching.pairs.tolist() == [[0, 1]]


def test_boundaries_in_time_only_the_estimate_assessed_are_not_scored():
    ref = Assessment((0, 1000), ((500, 800, PRESENT),))
    est = Assessment((0, 1200), ((500, 800, PRESENT), (1050, 1100, PRESENT)))
    got = events.event_matchings(ref, est, 10, method=True)
    for key, time in [("onset", 500), ("end", 800)]:
        matching = got[key]
        assert event_cells(matching).tolist() == [1, 0, 0]
        assert matching.reference.tolist() == matching.estimate.tolist() == [time]
        assert matching.excluded_estimate.size == 0


@pytest.mark.parametrize("state", [UNCERTAIN, NOT_OBSERVABLE])
def test_method_abstention_is_absent_but_reader_abstention_censors(state):
    ref = Assessment((0, 1000), ((500, 800, PRESENT),))
    est = Assessment((0, 1000), ((400, 500, state), (500, 800, PRESENT)))
    method = events.event_matchings(ref, est, 10, method=True)["onset"]
    assert event_cells(method).tolist() == [1, 0, 0]
    reader = events.event_matchings(ref, est, 10, method=False)["onset"]
    assert event_cells(reader).tolist() == [0, 0, 0]
    assert reader.excluded_reference.tolist() == [0]
    assert reader.unmatched_reference.size == 0


def test_method_abstention_never_excludes_a_missed_reference_boundary():
    ref = Assessment((0, 1000), ((500, 800, PRESENT),))
    est = Assessment((0, 1000), ((400, 900, UNCERTAIN),))
    for matching in events.event_matchings(ref, est, 10, method=True).values():
        assert event_cells(matching).tolist() == [0, 0, 1]
        assert matching.excluded_reference.size == 0


def test_matching_precedes_exclusion_beside_an_uncertain_span():
    ref = Assessment((0, 1000), ((400, 490, UNCERTAIN), (500, 800, PRESENT)))
    est = Assessment((0, 1000), ((495, 800, PRESENT),))
    got = events.event_matchings(ref, est, 10, method=False)["onset"]
    assert event_cells(got).tolist() == [1, 0, 0]
    assert got.offsets.tolist() == [5]
    assert got.excluded_reference.size == got.excluded_estimate.size == 0


@pytest.mark.parametrize("onset, excluded", [(390, True), (510, True), (511, False)])
def test_exclusion_uses_distance_to_the_closed_span(onset, excluded):
    ref = Assessment((0, 1000), ((400, 500, UNCERTAIN),))
    est = Assessment((0, 1000), ((onset, 800, PRESENT),))
    got = events.event_matchings(ref, est, 10, method=True)["onset"]
    assert got.excluded_estimate.size == int(excluded)
    assert got.unmatched_estimate.size == int(not excluded)


def test_exclusion_has_the_same_float_slack_as_matching():
    ref = Assessment((0, 1000), ((400, 500, UNCERTAIN),))
    est = Assessment((0, 1000), ((510, 800, PRESENT),))
    got = events.event_matchings(ref, est, 10 - 5e-10, method=True)["onset"]
    assert got.excluded_estimate.tolist() == [0]


def test_reader_swapping_swaps_errors_and_excluded_boundaries():
    a = Assessment(
        (0, 2000), ((100, 200, 1), (400, 500, 2), (500, 600, 1), (800, 900, 1))
    )
    b = Assessment(
        (0, 2000),
        ((105, 205, 1), (450, 600, 1), (750, 800, 2), (800, 900, 1), (1200, 1300, 1)),
    )
    ab = events.event_matchings(a, b, 10, method=False)
    ba = events.event_matchings(b, a, 10, method=False)
    for key in ab:
        assert np.array_equal(event_cells(ab[key]), event_cells(ba[key])[[0, 2, 1]])
        assert np.array_equal(ab[key].excluded_reference, ba[key].excluded_estimate)
        assert np.array_equal(ab[key].excluded_estimate, ba[key].excluded_reference)
    assert ab["onset"].excluded_reference.size == 1
    assert ab["onset"].excluded_estimate.size == 1
    assert ab["onset"].unmatched_estimate.size == 1


def test_one_reader_using_uncertain_at_onsets_does_not_change_onset_f1():
    spans = ((100, 200, 1), (400, 500, 1), (700, 800, 1))
    a = Assessment((0, 1000), spans)
    b = Assessment((0, 1000), (*spans, (350, 400, 2), (650, 700, 2)))
    got = events.event_matchings(a, b, 10, method=False)["onset"]
    assert f1(event_cells(got)) == 1.0
    assert got.excluded_reference.tolist() == [1, 2]
    assert got.reference.tolist() == [100, 400, 700]
    assert got.estimate.tolist() == [100]


def test_boundaries_on_both_common_window_edges_are_not_scored():
    ref = Assessment((0, 1000), ((400, 600, PRESENT),))
    est = Assessment((400, 600), ((400, 600, PRESENT),))
    for matching in events.event_matchings(ref, est, 10, method=True).values():
        assert matching.reference.size == matching.estimate.size == 0
        assert event_cells(matching).tolist() == [0, 0, 0]


def test_disjoint_windows_score_no_boundaries():
    ref = Assessment((0, 100), ((20, 80, 1),))
    est = Assessment((200, 300), ((220, 280, 1),))
    for matching in events.event_matchings(ref, est, 1000, method=True).values():
        assert matching.reference.size == matching.estimate.size == 0


def test_d19_disruption_points_take_inclusive_end_slack():
    from labeler.events.catalog.check import DISRUPTION_TIMING_TOLERANCE_MS

    times = [7.999, 8, 5260, 5260.53, 5262, 5262.001]
    got = within(times, (8, 5260), end_slack_ms=DISRUPTION_TIMING_TOLERANCE_MS)
    assert got.tolist() == [8, 5260, 5260.53, 5262]
    assert within(times, (8, 5260)).tolist() == [8]


def test_nearest_first_is_not_maximum_cardinality():
    got = match([0, 10], [6, 14], 6)
    assert got.pairs.tolist() == [[1, 0]]  # two pairs could fit the tolerance
    assert event_cells(got).tolist() == [1, 1, 1]


def test_match_records_given_times_and_has_no_exclusions():
    got = match([10, 0], [6, 14], 6)
    assert got.reference.tolist() == [10.0, 0.0]
    assert got.estimate.tolist() == [6.0, 14.0]
    assert got.reference.dtype == got.estimate.dtype == np.dtype(float)
    assert got.excluded_reference.dtype.kind == got.excluded_estimate.dtype.kind == "i"
    assert got.excluded_reference.size == got.excluded_estimate.size == 0


def test_scoring_imports_do_not_load_heavy_packages():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import labeler.scoring.frames; "
                "import labeler.scoring.events; import labeler.scoring.stats; "
                "assert not {'torch', 'pandas', 'toksearch', 'toksearch_d3d'} "
                "& sys.modules.keys()"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
