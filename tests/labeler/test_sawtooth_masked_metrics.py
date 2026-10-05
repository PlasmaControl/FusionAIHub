"""Masked span matching preserves the original annotation identities."""

import numpy as np

from labeler.sawtooth.metrics import masked_interval_cells


def test_missing_support_does_not_multiply_reference_spans():
    cells = masked_interval_cells(
        [(0.0, 1.0)],
        [(0.0, 0.3), (0.6, 1.0)],
        [(0.0, 0.3), (0.6, 1.0)],
        0.1,
    )
    assert cells.tolist() == [1, 1, 0]


def test_multiple_supported_pieces_of_one_estimate_match_one_annotation():
    cells = masked_interval_cells(
        [(0.0, 0.3), (0.6, 1.0)],
        [(0.0, 1.0)],
        [(0.0, 0.3), (0.6, 1.0)],
        0.1,
    )
    assert cells.tolist() == [1, 0, 1]


def test_zero_support_spans_are_excluded_from_all_denominators():
    cells = masked_interval_cells(
        [(0.0, 1.0), (2.0, 3.0)],
        [(0.0, 1.0), (4.0, 5.0)],
        [(0.0, 1.0)],
        0.1,
    )
    assert cells.tolist() == [1, 0, 0]


def test_iou_uses_supported_duration_and_ignores_missing_gap():
    cells = masked_interval_cells(
        [(0.0, 1.0)],
        [(0.0, 10.0)],
        [(0.0, 1.0)],
        0.9,
    )
    assert cells.tolist() == [1, 0, 0]


def test_overlapping_support_is_merged_before_measuring_iou():
    # Union support is [0, 1]; duplicating [0, 0.2] must not inflate overlap.
    cells = masked_interval_cells(
        [(0.0, 1.0)],
        [(0.0, 0.2)],
        [(0.0, 1.0), (0.0, 0.2)],
        0.25,
    )
    assert cells.tolist() == [0, 1, 1]


def test_no_observable_support_is_no_assessment():
    assert np.array_equal(
        masked_interval_cells([(0.0, 1.0)], [(0.0, 1.0)], []), [0, 0, 0]
    )
