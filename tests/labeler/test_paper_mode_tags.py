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
    tagged = {round(b.f_khz): b.tags for b in mt.tag_blobs(found, spans)}
    assert tagged == {101: (mt.AE,), 11: (mt.NTM,)}  # AE only above 60 kHz


def test_two_labels_in_one_band_both_tag_the_blob():
    f, t = _grid(100)
    lit = np.zeros((ROWS, 100), bool)
    lit[20:24, 10:40] = True
    (blob,) = mt.tag_blobs(
        mt.blobs(lit, t, f),
        {mt.SAWTOOTH: [(0.0, 100.0)], mt.NTM: [(5.0, 60.0)]},
    )
    assert blob.tags == (mt.NTM, mt.SAWTOOTH)


def test_a_label_over_under_half_the_blob_does_not_tag_it():
    f, t = _grid(100)
    lit = np.zeros((ROWS, 100), bool)
    lit[20:24, 0:40] = True  # 0-39 ms
    (blob,) = mt.tag_blobs(mt.blobs(lit, t, f), {mt.NTM: [(30.0, 80.0)]})
    assert blob.tags == ()  # 10 of 39 ms
    (blob,) = mt.tag_blobs(mt.blobs(lit, t, f), {mt.NTM: [(15.0, 80.0)]})
    assert blob.tags == (mt.NTM,)  # 24 of 39 ms


def test_a_blob_of_one_column_is_tagged_when_its_time_is_in_a_span():
    f, t = _grid(100)
    lit = np.zeros((ROWS, 100), bool)
    lit[20:24, 50] = True
    (blob,) = mt.tag_blobs(mt.blobs(lit, t, f), {mt.NTM: [(40.0, 60.0)]})
    assert blob.tags == (mt.NTM,)
    (blob,) = mt.tag_blobs(mt.blobs(lit, t, f), {mt.NTM: [(0.0, 10.0)]})
    assert blob.tags == ()


def test_spans_are_merged_where_they_touch():
    assert mt.union([(5.0, 9.0), (0.0, 5.0), (20.0, 30.0), (25.0, 27.0)]) == [
        (0.0, 9.0),
        (20.0, 30.0),
    ]
    assert mt.overlap_ms(4.0, 25.0, [(0.0, 9.0), (20.0, 30.0)]) == 10.0
