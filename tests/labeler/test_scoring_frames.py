"""Frames: states by precedence, counts against a reference, shot presence."""

from __future__ import annotations

import numpy as np
import pytest

from labeler.events.review.labels import normalise
from labeler.scoring.frames import (
    ABSENT,
    NOT_OBSERVABLE,
    OUTSIDE,
    PRESENT,
    UNCERTAIN,
    Assessment,
    agreement_frames,
    frame_counts,
    frame_grid,
    frame_states,
    shot_presence,
)


def test_a_frame_takes_the_highest_state_that_touches_it():
    read = Assessment((0, 60), ((15, 25, PRESENT), (30, 31, UNCERTAIN)))
    first, n = frame_grid(read)
    assert (first, n) == (0, 6)
    assert frame_states(read, first, n).tolist() == [
        ABSENT,
        PRESENT,
        PRESENT,
        UNCERTAIN,
        ABSENT,
        ABSENT,
    ]


def test_not_observable_outranks_everything():
    read = Assessment((0, 20), ((0, 12, PRESENT), (12, 14, NOT_OBSERVABLE)))
    assert frame_states(read, 0, 2).tolist() == [PRESENT, NOT_OBSERVABLE]


def test_only_frames_the_whole_window_covers_belong_to_it():
    read = Assessment((5, 95))
    assert frame_grid(read) == (1, 8)
    states = frame_states(read, 0, 10)
    assert states[0] == OUTSIDE and states[9] == OUTSIDE
    assert (states[1:9] == ABSENT).all()


def test_frame_counts_by_hand():
    # Frames 0-9. Reference: present 2-5, uncertain 8-9. Estimate: present 4-8.
    reference = Assessment((0, 100), ((20, 60, PRESENT), (80, 100, UNCERTAIN)))
    estimate = Assessment((0, 100), ((40, 90, PRESENT),))
    counts = frame_counts(reference, estimate)
    assert (counts.tp, counts.fp, counts.fn, counts.tn) == (2, 2, 2, 2)
    assert counts.excluded == 2
    assert counts.method_uncertain == counts.method_unobserved == 0
    assert counts.cells().tolist() == [2, 2, 2, 2]


def test_a_method_saying_uncertain_counts_as_not_present():
    reference = Assessment((0, 100), ((20, 60, PRESENT),))
    estimate = Assessment((0, 100), ((20, 60, UNCERTAIN), (60, 70, NOT_OBSERVABLE)))
    counts = frame_counts(reference, estimate)
    assert (counts.tp, counts.fn, counts.fp) == (0, 4, 0)
    assert counts.method_uncertain == 4
    assert counts.method_unobserved == 1


def test_only_frames_both_windows_cover_are_counted():
    reference = Assessment((0, 100), ((0, 100, PRESENT),))
    estimate = Assessment((50, 200), ((50, 200, PRESENT),))
    counts = frame_counts(reference, estimate)
    assert counts.tp == 5 and counts.fp == counts.fn == counts.tn == 0


def test_agreement_drops_frames_any_reader_did_not_call():
    a = Assessment((0, 40), ((0, 20, PRESENT), (30, 40, UNCERTAIN)))
    b = Assessment((0, 40), ((10, 30, PRESENT),))
    assert agreement_frames(a, b).tolist() == [[1, 0], [1, 1], [0, 1]]


def test_shot_presence():
    assert shot_presence(Assessment((0, 100), ((10, 20, PRESENT),))) == 1
    assert shot_presence(Assessment((0, 100))) == 0
    assert shot_presence(Assessment((0, 100), ((10, 20, UNCERTAIN),))) is None
    assert shot_presence(Assessment((0, 100), ((0, 100, NOT_OBSERVABLE),))) is None
    assert shot_presence(Assessment((0, 100), ((0, 50, NOT_OBSERVABLE),))) == 0


@pytest.mark.parametrize(
    "window, spans",
    [
        ((10, 10), ()),
        ((0, 100), ((10, 30, 1), (20, 40, 1))),
        ((0, 100), ((90, 110, 1),)),
        ((0, 100), ((10, 20, 4),)),
        ((0, 100), ((20, 20, 1),)),
    ],
)
def test_malformed_assessments_are_refused(window, spans):
    with pytest.raises(ValueError):
        Assessment(window, spans)


def test_from_a_review_label_and_from_tiling_rows():
    label = normalise((0, 100), [(10, 20, 1), (40, 50, 2)])
    assert Assessment.from_label(label) == Assessment(
        (0, 100), ((10, 20, 1), (40, 50, 2))
    )
    rows = [(0, 10, 0), (10, 20, 1), (20, 100, 0)]
    assert Assessment.from_rows(rows) == Assessment((0, 100), ((10, 20, 1),))


def test_runs_fill_gaps_and_merge_neighbours():
    read = Assessment((0, 100), ((10, 20, 1), (20, 30, 1), (50, 60, 2)))
    assert read.runs() == [
        (0, 10, 0),
        (10, 30, 1),
        (30, 50, 0),
        (50, 60, 2),
        (60, 100, 0),
    ]
    assert np.array_equal(frame_states(read, 0, 10), [0, 1, 1, 0, 0, 2, 0, 0, 0, 0])
