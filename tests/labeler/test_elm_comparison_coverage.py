"""Every method is compared only on bins fully supported by its outputs."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from labeler.elm import compare, dsm, labels, methods


def _bins(starts):
    n = len(starts)
    return labels.Bins(
        np.asarray(starts, float),
        np.zeros(n, np.int8),
        np.full(n, "absent", object),
        np.zeros(n, int),
    )


def _scores():
    usable = np.ones(len(dsm.ROW_T_MS), bool)
    row = dsm.Rows(1, np.zeros((len(usable), 60)), usable, usable.copy(), (), {}, ())
    return compare.DsmScores(
        {1: row},
        {1: np.ones((len(usable), 4))},
        {compare.NAME["detect"]: {1: np.full(len(usable), 0.5)}},
        {compare.NAME["dsm"]: {1: 0.5}, compare.NAME["detect"]: {1: 0.5}},
        {1: row},
    )


def test_common_masks_require_full_analysed_bin_not_a_detection_touch():
    bins = _bins([100, 150, 200, 250, 300, 350])
    cover = methods.cover_frame([100, 200, 350], [180, 300, 500])
    masks = compare.bin_support(1, bins, cover, np.ones(600), _scores())

    kept = methods.restrict_bins(bins, np.logical_and.reduce(list(masks.values())))

    assert kept.t0.tolist() == [100, 200, 250, 350]
    assert not masks["analysed_time"][1]  # 30 ms covered still leaves unknown time
    assert not masks["analysed_time"][4]  # no analysed sample; never a negative


def test_repaired_detector_usability_and_finite_scores_narrow_common_bins():
    bins = _bins([100, 150, 200, 250])
    dscores = _scores()
    repaired_usable = dscores.rows[1].usable.copy()
    repaired_usable[dsm.row_index(bins, 0)[2]] = False
    dscores.detection_rows = {1: replace(dscores.rows[1], usable=repaired_usable)}
    dscores.scores[compare.NAME["detect"]][1][dsm.row_index(bins, 0)[3]] = np.nan
    masks = compare.bin_support(
        1, bins, methods.cover_frame([0], [500]), np.ones(600), dscores
    )

    kept = methods.restrict_bins(bins, np.logical_and.reduce(list(masks.values())))

    assert kept.t0.tolist() == [100, 150]
    np.testing.assert_array_equal(masks[compare.NAME["detect"]], [1, 1, 0, 0])


def test_common_parts_intersect_repaired_rows_and_diagnostic_coverage():
    bins = _bins([100, 150, 200, 250])
    dscores = _scores()
    repaired_usable = dscores.rows[1].usable.copy()
    repaired_usable[dsm.row_index(bins, 0)[2]] = False
    dscores.detection_rows = {1: replace(dscores.rows[1], usable=repaired_usable)}
    spans = pd.DataFrame({"t_start": [0], "t_end": [500], "kind": ["absent"]})
    oof = SimpleNamespace(trace=lambda shot: np.ones((2, 600)), threshold={1: 0.5})

    parts, kept = compare.shot_parts(
        1, spans, bins, methods.cover_frame([100], [280]), oof, dscores, {}, {}
    )

    assert kept.t0.tolist() == [100, 150]
    assert all(len(part.truth) == 2 for part in parts.values())


def test_trace_gaps_and_out_of_range_rows_remain_unsupported():
    bins = _bins([-100, 100, 150, 5950, 6000])
    trace = np.ones(6100)
    trace[150 + 50] = np.nan  # the 150 ms bin, on a grid starting at -50 ms
    masks = compare.bin_support(
        1, bins, methods.cover_frame([-100], [6100]), trace, _scores()
    )

    assert masks[compare.NAME["ours"]].tolist() == [False, True, False, True, True]
    assert not masks[compare.NAME["detect"]][-2:].any()


def test_missing_repaired_input_audit_makes_detection_support_unknown():
    dscores = _scores()
    dscores.detection_rows = {}

    support = compare.bin_support(
        1,
        _bins([100, 150]),
        methods.cover_frame([0], [500]),
        np.ones(600),
        dscores,
    )

    assert not support[compare.NAME["detect"]].any()


def test_saved_detector_requires_a_repaired_input_coverage_audit(tmp_path):
    dscores = _scores()
    dscores.save(tmp_path)

    with pytest.raises(ValueError, match="repaired detection rows"):
        compare.DsmScores.load(
            tmp_path, dscores.rows, variants=[compare.NAME["detect"]]
        )


def test_saved_detector_automatically_loads_repaired_input_coverage(tmp_path):
    dscores = _scores()
    dscores.save(tmp_path)
    cache = tmp_path / "repaired_raw_rows"
    cache.mkdir()
    repaired = replace(dscores.rows[1], usable=dscores.rows[1].usable.copy())
    repaired.usable[6] = False
    dsm.save_rows(repaired, cache / "1.npz")

    loaded = compare.DsmScores.load(
        tmp_path, dscores.rows, variants=[compare.NAME["detect"]]
    )

    assert 1 in loaded.detection_rows
    assert not loaded.detection_rows[1].usable[6]


def test_float32_timestamp_roundoff_does_not_create_missing_coverage():
    bins = _bins([100, 150, 200])
    cover = methods.cover_frame([100 + 1e-5, 200 + 1e-3], [200 - 1e-5, 250])

    supported = compare.covered_bin_mask(bins, cover)

    assert supported.tolist() == [True, True, False]
