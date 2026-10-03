"""TokEye's mask -> filtered mask -> components -> label tags
(`paper.mode_tags`), on small drawn masks: no model or store is read."""

from __future__ import annotations

import numpy as np

from labeler.paper import mode_tags as mt

ROWS = 512


def _grid(n_cols):
    """The 512 rows' frequencies (0.5 kHz apart) and `n_cols` columns' times (1 ms)."""
    return np.arange(ROWS) * 0.5, np.arange(n_cols, dtype=float)


def test_a_mode_is_coherent_and_not_transient():
    coh = np.array([[0.1, 0.3, 0.3, 0.2]])
    tra = np.array([[0.0, 0.0, 0.3, 0.1]])
    assert mt.mode_mask(coh, tra, 0.2).tolist() == [[False, True, False, True]]


def test_pickup_goes_and_a_mode_crossing_it_stays_whole():
    lit = np.zeros((ROWS, 20), bool)
    lit[100, :] = True  # a line of pickup, one row
    lit[90:111, 5:8] = True  # a mode across it, rows 90-110
    share = np.zeros(ROWS)
    share[100] = 0.9
    out = mt.bridge_pickup(lit, share)
    assert not out[100, 10:].any()  # the pickup line is gone
    assert out[100, 5:8].all()  # the mode's pixels on it stay
    assert out[95, 5:8].all()


def test_pickup_on_the_edge_rows_is_dropped():
    lit = np.ones((ROWS, 4), bool)
    share = np.zeros(ROWS)
    share[0] = share[ROWS - 1] = 1.0
    out = mt.bridge_pickup(lit, share)
    assert not out[0].any() and not out[-1].any()
    assert out[1:-1].all()


def test_clean_drops_small_objects_and_fills_small_holes():
    lit = np.zeros((ROWS, 40), bool)
    lit[10:14, 5:30] = True  # 100 pixels
    lit[11, 12] = False  # a one-pixel hole
    lit[200, 3] = True  # one pixel of salt
    out = mt.clean(lit, min_size=30, hole_area=10)
    assert out[11, 12]
    assert not out[200, 3]
    assert out[10:14, 5:30].all()


def test_filtered_runs_the_steps_in_order():
    coh = np.zeros((ROWS, 40))
    tra = np.zeros((ROWS, 40))
    coh[50:54, 5:35] = 0.9
    tra[50:54, 20:22] = 0.9  # a burst across the mode
    share = np.zeros(ROWS)
    out = mt.filtered(coh, tra, share, min_size=10, threshold=0.2)
    assert out[50:54, 5:20].all() and out[50:54, 22:35].all()
    assert not out[50:54, 20:22].any()


def test_blobs_are_in_the_shots_units():
    f, t = _grid(60)
    lit = np.zeros((ROWS, 60), bool)
    lit[20:24, 10:30] = True  # 10.0-11.5 kHz, 10-29 ms
    (blob,) = mt.blobs(lit, t, f)
    assert (blob.t0_ms, blob.t1_ms) == (10.0, 29.0)
    assert (blob.f0_khz, blob.f1_khz) == (10.0, 11.5)
    assert blob.f_khz == 10.75
    assert blob.n_pix == 80


def test_a_blob_is_tagged_by_the_label_over_it_in_its_band():
    f, t = _grid(100)
    lit = np.zeros((ROWS, 100), bool)
    lit[200:204, 10:40] = True  # ~100 kHz, 10-39 ms
    lit[20:24, 10:40] = True  # ~10 kHz, 10-39 ms
    found = mt.blobs(lit, t, f)
    spans = {mt.AE: [(0.0, 50.0)], mt.NTM: [(0.0, 50.0)]}
    tagged = {
        round(b.f_khz): b.tags
        for b in mt.tag_blobs(found, spans, t, f, n_map=np.ones(lit.shape))
    }
    assert tagged == {101: (mt.AE,), 11: (mt.NTM,)}  # AE only above 60 kHz


def test_sawtooth_is_not_a_rotating_mode_tag():
    f, t = _grid(100)
    lit = np.zeros((ROWS, 100), bool)
    lit[20:24, 10:40] = True
    (blob,) = mt.tag_blobs(
        mt.blobs(lit, t, f),
        {mt.SAWTOOTH: [(0.0, 100.0)], mt.NTM: [(5.0, 60.0)]},
        t,
        f,
        n_map=np.ones(lit.shape),
    )
    assert blob.tags == (mt.NTM,)


def test_a_short_overlap_tags_only_the_pixels_inside_the_label():
    f, t = _grid(100)
    lit = np.zeros((ROWS, 100), bool)
    lit[20:24, 0:40] = True  # 0-39 ms
    spans = {mt.NTM: [(30.0, 80.0)]}
    found = mt.tag_blobs(mt.blobs(lit, t, f), spans, t, f, np.ones(lit.shape))
    assert found[0].tags == (mt.NTM,)
    mask = mt.tag_mask(
        found, mt.NTM, lit.shape, t, f, spans[mt.NTM], n_map=np.ones(lit.shape)
    )
    assert mask.sum() == 40
    assert not mask[:, :30].any()


def test_a_blob_of_one_column_is_tagged_when_its_time_is_in_a_span():
    f, t = _grid(100)
    lit = np.zeros((ROWS, 100), bool)
    lit[20:24, 50] = True
    (blob,) = mt.tag_blobs(
        mt.blobs(lit, t, f), {mt.NTM: [(40.0, 60.0)]}, t, f, np.ones(lit.shape)
    )
    assert blob.tags == (mt.NTM,)
    (blob,) = mt.tag_blobs(
        mt.blobs(lit, t, f), {mt.NTM: [(0.0, 10.0)]}, t, f, np.ones(lit.shape)
    )
    assert blob.tags == ()


def test_spans_are_merged_where_they_touch():
    assert mt.union([(5.0, 9.0), (0.0, 5.0), (20.0, 30.0), (25.0, 27.0)]) == [
        (0.0, 9.0),
        (20.0, 30.0),
    ]
    assert mt.overlap_ms(4.0, 25.0, [(0.0, 9.0), (20.0, 30.0)]) == 10.0


def test_projection_clips_absent_gaps_and_both_sides_of_60_khz():
    f, t = _grid(20)
    lit = np.zeros((ROWS, 20), bool)
    lit[110:124, :] = True  # 55-61.5 kHz: crosses both fold and tag boundary
    spans = {mt.AE: [(2.0, 8.0), (12.0, 18.0)], mt.NTM: [(4.0, 16.0)]}
    found = mt.tag_blobs(mt.blobs(lit, t, f), spans, t, f, np.ones(lit.shape))
    ae = mt.tag_mask(found, mt.AE, lit.shape, t, f, spans[mt.AE])
    ntm = mt.tag_mask(
        found, mt.NTM, lit.shape, t, f, spans[mt.NTM], n_map=np.ones(lit.shape)
    )
    assert ae.sum() == 48  # four rows >=60, twelve present columns
    assert ntm.sum() == 120  # ten rows 55-59.5, twelve present columns
    assert not ae[f < 60].any()
    assert not ae[:, 8:12].any()
    assert not ae[:, 18:].any()  # end is exclusive
    assert not ntm[f >= 60].any()
    assert not ntm[:, :4].any()


def test_ntm_requires_dominant_n_one_or_two_and_measured_support():
    f, t = _grid(20)
    lit = np.zeros((ROWS, 20), bool)
    for row in (10, 30, 50, 70):
        lit[row : row + 3, :] = True
    n = np.full(lit.shape, np.nan)
    n[10:13] = 1
    n[30:33] = 2
    n[50:53] = 3
    n[50, :3] = 1  # a small n=1 minority must not admit n=3
    found = mt.tag_blobs(mt.blobs(lit, t, f), {mt.NTM: [(0, 20)]}, t, f, n)
    assert [b.tags for b in found] == [(mt.NTM,), (mt.NTM,), (), ()]


def test_persistent_physical_rows_are_kept_at_fifteen_and_thirty_khz():
    lit = np.zeros((ROWS, 20), bool)
    lit[30, :13] = True  # 65%: real n=2 near 15 kHz
    lit[60, :11] = True  # 55%: its ~30 kHz harmonic
    out = mt.bridge_pickup(lit, lit.mean(axis=1))
    assert out[30, :13].all()
    assert out[60, :11].all()


def test_n_sampling_decodes_brightness_and_does_not_extrapolate():
    codes = np.array([[2, 0], [5, 4]])  # ns=[1,2], level 0 means unknown
    sampled = mt.sample_n_map(
        codes,
        [1, 2],
        [0.0, 2.0],
        [0.0, 10.0],
        np.array([0.0, 2.0, 4.0]),
        np.array([0.0, 10.0, 55.0]),
    )
    assert sampled[0, 0] == 1
    assert np.isnan(sampled[0, 1])
    assert sampled[1, :2].tolist() == [2, 1]
    assert np.isnan(sampled[2]).all()
    assert np.isnan(sampled[:, 2]).all()


def test_tied_n_one_and_three_are_ambiguous_not_a_tearing_mode():
    f, t = _grid(4)
    lit = np.zeros((ROWS, 4), bool)
    lit[10:13] = True
    n = np.full(lit.shape, np.nan)
    n[10:13, :2] = 1
    n[10:13, 2:] = 3
    (blob,) = mt.tag_blobs(mt.blobs(lit, t, f), {mt.NTM: [(0, 4)]}, t, f, n)
    assert blob.dominant_n is None
    assert blob.tags == ()


def test_ntm_projection_stops_at_unmeasured_pixels_inside_a_component():
    f, t = _grid(4)
    lit = np.zeros((ROWS, 4), bool)
    lit[50:80] = True  # one component, 25–39.5 kHz
    n = np.full(lit.shape, np.nan)
    n[50:61, :2] = 1  # n-map coverage through 30 kHz, with a temporal gap
    spans = {mt.NTM: [(0, 4)]}
    found = mt.tag_blobs(mt.blobs(lit, t, f), spans, t, f, n)
    shown = mt.tag_mask(found, mt.NTM, lit.shape, t, f, spans[mt.NTM], n_map=n)
    assert shown.sum() == 22
    assert not shown[f > 30].any()
    assert not shown[:, 2:].any()
