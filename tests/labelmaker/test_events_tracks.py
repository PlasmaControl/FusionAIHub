"""Coherent mask -> components -> tracks -> descriptors -> events.

Everything here is measured against a mask that was DRAWN, not one that was
found: `synth_mask` (in `conftest.py`) puts five things on the pipeline's own
zoom grid - a pickup line, an EHO fundamental with a hole in it and its two
harmonics, a mode in two pieces, a fishbone chirping at a known rate - and
every expected value below is arithmetic on the rows and columns they were
drawn at. A descriptor that drifts is a descriptor that stops matching the
number in this file.

The merge geometry is tested twice over: once on hand-built boxes, whose
answer can be read off the page, and once against `_merge_literal`, the
literal port of the reference, on 200 random component sets - the fast
path's early break has to be an optimisation and nothing else.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from labelmaker.events import masks, schema, tracks

from .conftest import (
    SYNTH_BIN_KHZ,
    SYNTH_CHIRP_KHZ_PER_MS,
    SYNTH_CHIRP_N,
    SYNTH_CHIRP_ROWS,
    SYNTH_EHO_COLS,
    SYNTH_EHO_KHZ,
    SYNTH_HARMONIC_COLS,
    SYNTH_HOP_S,
    SYNTH_PROB,
    SYNTH_SPLIT_COLS,
    SYNTH_T,
    _synth_band,
)

#: `merge` returns tracks in ascending start time, so the seven tracks of
#: the synthetic channel are these seven, in this order. Pinned by
#: `test_the_synthetic_channel_is_seven_tracks_in_time_order`.
PICKUP, EHO, H2, H3, SPLIT_A, SPLIT_B, FISHBONE = range(7)


def _mask(*boxes, n_cols=200):
    """A `(512, n_cols)` mask lit over each half-open `(r0, r1, c0, c1)`."""
    mask = np.zeros((masks.N_BINS, n_cols), dtype=bool)
    for r0, r1, c0, c1 in boxes:
        mask[r0:r1, c0:c1] = True
    return mask


def _comp(r0, r1, c0, c1, label=1):
    """One Component covering a solid box, with its pixel lists."""
    rows, cols = np.mgrid[r0:r1, c0:c1]
    return tracks.Component(
        row0=r0, row1=r1, col0=c0, col1=c1,
        n_pix=(r1 - r0) * (c1 - c0),
        rows=rows.ravel(), cols=cols.ravel(), label=label,
    )


def _boxes(groups):
    """`(row0, row1, col0, col1)` of each group, for comparing partitions."""
    return [
        (
            min(c.row0 for c in g), max(c.row1 for c in g),
            min(c.col0 for c in g), max(c.col1 for c in g),
        )
        for g in groups
    ]


@pytest.fixture
def synth(synth_mask):
    """The synthetic channel's `(tracks, prob, raw_logpow, freq_khz, t_s)`."""
    prob, raw, freq_khz, t_s = synth_mask()
    comps = tracks.components(prob >= masks.PROB_THRESHOLD)
    got = [
        tracks.descriptors(
            group, prob=prob, raw_logpow=raw, freq_khz=freq_khz, t_s=t_s
        )
        for group in tracks.merge(comps)
    ]
    return got, prob, raw, freq_khz, t_s


# --------------------------------------------------------------- constants

def test_the_thresholds_are_ours_and_not_the_reference_detectors():
    # raddet's merge geometry, this pipeline's numbers: 200 pixels is a
    # blob worth a row in the events table, 40 columns is 41 ms of zoom
    # pass, and the 95th percentile is a confidence that one hot pixel
    # cannot set.
    assert tracks.MIN_AREA == 200
    assert tracks.MERGE_GAP_COLS == 40
    assert tracks.FREQ_OVERLAP == 0.5
    assert tracks.CONF_PCT == 95
    assert tracks.PICKUP_ROW_FRACTION == 0.8
    assert tracks.HARMONIC_ORDERS == (2, 3, 4, 5)
    assert tracks.HARMONIC_TOL == 0.06
    assert tracks.MIN_OVERLAP_S == 0.02
    assert tracks.IOU_MIN == 0.2


def test_the_source_and_phenomena_are_ones_the_events_table_knows():
    assert tracks.SOURCE in schema.KNOWN_SOURCES
    assert tracks.PHENOMENON == "coherent_mode"
    assert tracks.PICKUP_PHENOMENON == "pickup"


# -------------------------------------------------------------- components

def test_a_solid_box_is_one_component_with_half_open_extents():
    comps = tracks.components(_mask((100, 110, 20, 60)))
    assert len(comps) == 1
    got = comps[0]
    assert (got.row0, got.row1, got.col0, got.col1) == (100, 110, 20, 60)
    assert got.n_pix == 10 * 40 == got.rows.size == got.cols.size
    assert got.rows.min() == 100 and got.rows.max() == 109
    assert got.cols.min() == 20 and got.cols.max() == 59


def test_components_are_eight_connected():
    # A mode drifting a row per column is one mode, not one per column.
    mask = np.zeros((masks.N_BINS, 20), dtype=bool)
    for c in range(20):
        mask[100 + c, c] = True
    comps = tracks.components(mask, min_area=1)
    assert len(comps) == 1 and comps[0].n_pix == 20


def test_a_blob_below_the_minimum_area_is_not_a_component():
    mask = _mask((100, 110, 0, 40), (300, 302, 0, 2))
    assert len(tracks.components(mask)) == 1              # 400 px, not 4
    assert len(tracks.components(mask, min_area=1)) == 2


def test_an_empty_mask_has_no_components():
    assert tracks.components(np.zeros((masks.N_BINS, 50), dtype=bool)) == []


def test_components_refuses_a_mask_that_is_not_the_spectrogram():
    with pytest.raises(ValueError, match="512"):
        tracks.components(np.zeros((348, 50), dtype=bool))
    with pytest.raises(ValueError, match="512"):
        tracks.components(np.zeros(512, dtype=bool))


def test_the_salt_of_the_synthetic_channel_vanishes(synth_mask):
    # 400 single pixels at 51-61 kHz: every one of them is a component and
    # not one of them is a track.
    prob, _, _, _ = synth_mask()
    mask = prob >= masks.PROB_THRESHOLD
    assert len(tracks.components(mask, min_area=1)) > 400
    assert len(tracks.components(mask)) == 8
    assert not any(c.row0 >= 400 for c in tracks.components(mask))


# ------------------------------------------------------------------- merge

def test_two_boxes_close_in_time_and_overlapping_in_frequency_merge():
    comps = [_comp(100, 110, 0, 40, 1), _comp(100, 110, 60, 100, 2)]
    got = tracks.merge(comps)
    assert len(got) == 1 and got[0] == comps
    assert _boxes(got) == [(100, 110, 0, 100)]


def test_a_time_gap_wider_than_the_limit_leaves_two_tracks():
    # 40 columns is the limit and it is inclusive: 41 is two tracks.
    near = [_comp(100, 110, 0, 40, 1), _comp(100, 110, 80, 120, 2)]
    far = [_comp(100, 110, 0, 40, 1), _comp(100, 110, 81, 120, 2)]
    assert len(tracks.merge(near)) == 1
    assert len(tracks.merge(far)) == 2


def test_a_frequency_overlap_below_half_the_smaller_extent_does_not_merge():
    # The smaller extent is 10 rows; 5 of them shared merges, 4 does not.
    half = [_comp(100, 110, 0, 40, 1), _comp(105, 130, 50, 90, 2)]
    less = [_comp(100, 110, 0, 40, 1), _comp(106, 130, 50, 90, 2)]
    assert len(tracks.merge(half)) == 1
    assert len(tracks.merge(less)) == 2


def test_overlapping_in_time_is_not_the_same_as_being_the_same_mode():
    # Two bands 200 rows apart and simultaneous stay two tracks.
    comps = [_comp(100, 110, 0, 100, 1), _comp(300, 310, 0, 100, 2)]
    assert len(tracks.merge(comps)) == 2


def test_merging_nothing_gives_nothing():
    assert tracks.merge([]) == []
    assert tracks._merge_literal([]) == []


def test_a_lone_component_is_a_track_of_one():
    comps = [_comp(100, 110, 0, 40)]
    assert tracks.merge(comps) == [comps]


def test_tracks_come_back_in_ascending_start_column():
    comps = [
        _comp(100, 110, 500, 540, 1),
        _comp(300, 310, 0, 40, 2),
        _comp(200, 210, 250, 290, 3),
    ]
    got = tracks.merge(comps)
    assert [g[0].col0 for g in got] == [0, 250, 500]


@pytest.mark.parametrize("seed", range(200))
def test_the_fast_merge_is_the_literal_port(seed):
    # The production merge sorts, filters and breaks out of the scan early;
    # the reference does none of that. The partition has to be identical,
    # component for component, or the early break is not an optimisation.
    rng = np.random.default_rng(seed)
    comps = []
    for label in range(1, int(rng.integers(2, 13)) + 1):
        r0 = int(rng.integers(0, 60))
        c0 = int(rng.integers(0, 120))
        comps.append(
            _comp(r0, r0 + int(rng.integers(1, 13)),
                  c0, c0 + int(rng.integers(1, 13)), label)
        )
    assert tracks.merge(comps) == tracks._merge_literal(comps)


def test_the_minimum_area_is_applied_before_the_merge_not_after():
    # A blob too small to be a track must not be able to bridge two that
    # are: the 60-column gap here is only crossable through it.
    mask = _mask((100, 110, 0, 40), (100, 110, 60, 62), (100, 110, 100, 140))
    assert len(tracks.merge(tracks.components(mask))) == 2
    assert len(tracks.merge(tracks.components(mask, min_area=1))) == 1


# ------------------------------------------------------------- descriptors

def test_the_synthetic_channel_is_seven_tracks_in_time_order(synth):
    got, _, _, freq_khz, _ = synth
    assert len(got) == 7
    assert [round(t.f_centroid_khz, 3) for t in got] == [
        round(freq_khz[_synth_band(freq_khz, khz)[0] + 1], 3)
        for khz in (1.95, 8.0, 16.0, 24.0, 44.0, 44.0)
    ] + [round(got[FISHBONE].f_centroid_khz, 3)]
    assert [t.t0_s for t in got] == sorted(t.t0_s for t in got)


def test_the_eho_fundamental_is_the_box_it_was_drawn_in(synth):
    got, _, _, freq_khz, t_s = synth
    eho = got[EHO]
    r0, r1 = _synth_band(freq_khz, SYNTH_EHO_KHZ)
    assert (eho.t0_s, eho.t1_s) == (t_s[200], t_s[1199])
    assert (eho.f0_khz, eho.f1_khz) == (freq_khz[r0], freq_khz[r1 - 1])
    assert eho.f_centroid_khz == pytest.approx(freq_khz[r0 + 1])
    assert eho.bandwidth_khz == pytest.approx(2 * SYNTH_BIN_KHZ)
    assert eho.duration_ms == pytest.approx(999 * SYNTH_HOP_S * 1e3)
    assert eho.n_pix == 3 * (400 + 580)
    assert eho.n_components == 2
    assert eho.mean_prob == pytest.approx(SYNTH_PROB["eho"])
    assert eho.conf == pytest.approx(SYNTH_PROB["eho"])


def test_the_twenty_column_hole_costs_duty_and_not_a_second_track(synth):
    got, _, _, _, _ = synth
    (a0, a1), (b0, b1) = SYNTH_EHO_COLS
    assert got[EHO].n_components == 2                      # two components
    assert got[EHO].duty == pytest.approx(
        ((a1 - a0) + (b1 - b0)) / (b1 - a0)                 # 980 of 1000
    )


def test_the_sixty_column_gap_is_two_tracks(synth):
    got, _, _, _, t_s = synth
    (a0, a1), (b0, b1) = SYNTH_SPLIT_COLS
    assert (got[SPLIT_A].t0_s, got[SPLIT_A].t1_s) == (t_s[a0], t_s[a1 - 1])
    assert (got[SPLIT_B].t0_s, got[SPLIT_B].t1_s) == (t_s[b0], t_s[b1 - 1])
    assert got[SPLIT_A].duty == got[SPLIT_B].duty == 1.0
    assert got[SPLIT_A].n_components == got[SPLIT_B].n_components == 1


def test_a_solid_mode_has_a_duty_of_one(synth):
    got, _, _, _, _ = synth
    assert got[H2].duty == 1.0 and got[H3].duty == 1.0
    assert got[H2].n_pix == 3 * (SYNTH_HARMONIC_COLS[1] - SYNTH_HARMONIC_COLS[0])


def test_a_mode_that_does_not_chirp_has_no_slope_and_no_fit(synth):
    got, _, _, _, _ = synth
    assert got[EHO].chirp_khz_per_ms == 0.0
    assert got[EHO].chirp_r2 == 0.0        # a flat centroid explains nothing


def test_the_fishbone_recovers_the_chirp_it_was_drawn_with(synth):
    got, _, _, _, _ = synth
    fish = got[FISHBONE]
    assert fish.chirp_khz_per_ms == pytest.approx(SYNTH_CHIRP_KHZ_PER_MS, abs=0.05)
    assert fish.chirp_r2 > 0.95
    assert fish.n_components == 1
    assert fish.n_pix == SYNTH_CHIRP_N * SYNTH_CHIRP_ROWS
    assert fish.duty == 1.0
    assert fish.duration_ms == pytest.approx(
        (SYNTH_CHIRP_N - 1) * SYNTH_HOP_S * 1e3
    )
    assert fish.bandwidth_khz == pytest.approx(
        (SYNTH_CHIRP_ROWS - 1) * SYNTH_BIN_KHZ
    )


def test_a_row_lit_for_the_whole_record_is_pickup_not_plasma(synth):
    got, _, _, _, _ = synth
    assert got[PICKUP].row_lit_fraction == 1.0
    assert got[PICKUP].pickup is True
    assert got[PICKUP].n_pix == 3 * SYNTH_T


def test_a_mode_that_comes_and_goes_is_not_pickup(synth):
    got, _, _, _, _ = synth
    assert got[EHO].row_lit_fraction == pytest.approx(980 / SYNTH_T)
    assert got[EHO].pickup is False
    assert not any(t.pickup for t in got if t is not got[PICKUP])


def test_the_centroid_is_weighted_by_the_probability():
    # Two rows, one of them ten times as probable: the centroid sits a
    # tenth of a bin from the confident row, not between them.
    prob = np.zeros((masks.N_BINS, 60), dtype=np.float32)
    prob[100, :] = 0.9
    prob[101, :] = 0.09
    freq = masks.freq_axis_khz(5e5, 4)
    t_s = masks.col_times_s(60, 5e5, 4, 0.0)
    group = tracks.components(prob >= 0.05, min_area=1)
    got = tracks.descriptors(
        group[:1], prob=prob, raw_logpow=np.ones_like(prob), freq_khz=freq, t_s=t_s
    )
    assert got.f_centroid_khz == pytest.approx(
        (0.9 * freq[100] + 0.09 * freq[101]) / 0.99
    )


def test_the_per_column_centroid_is_weighted_by_the_power():
    # The chirp is read off the ridge, not off the middle of the mask: the
    # power says where the mode is inside a band the network lit whole.
    prob = np.zeros((masks.N_BINS, 40), dtype=np.float32)
    prob[100:110, :] = 0.9
    raw = np.full_like(prob, 0.01)
    for c in range(40):
        raw[100 + c // 4, c] = 10.0        # the ridge climbs a row every 4
    freq = masks.freq_axis_khz(5e5, 4)
    t_s = masks.col_times_s(40, 5e5, 4, 0.0)
    group = tracks.merge(tracks.components(prob >= 0.5))[0]
    got = tracks.descriptors(
        group, prob=prob, raw_logpow=raw, freq_khz=freq, t_s=t_s
    )
    # A bin every four columns of 1.024 ms, upwards.
    assert got.chirp_khz_per_ms == pytest.approx(
        SYNTH_BIN_KHZ / (4 * SYNTH_HOP_S * 1e3), rel=0.05
    )
    assert got.chirp_r2 > 0.9


def test_the_confidence_is_the_ninety_fifth_percentile_of_the_lit_pixels():
    prob = np.zeros((masks.N_BINS, 100), dtype=np.float32)
    prob[100, :] = 0.4
    prob[100, :5] = 1.0                    # 5 of 100 pixels at one
    freq = masks.freq_axis_khz(5e5, 4)
    t_s = masks.col_times_s(100, 5e5, 4, 0.0)
    group = tracks.components(prob >= 0.2, min_area=1)
    got = tracks.descriptors(
        group[:1], prob=prob, raw_logpow=np.ones_like(prob), freq_khz=freq, t_s=t_s
    )
    assert got.conf == pytest.approx(float(np.percentile(prob[100, :], 95)))
    assert got.mean_prob == pytest.approx(0.4 * 0.95 + 1.0 * 0.05)


def test_a_track_too_short_to_fit_reports_no_fit():
    prob = np.zeros((masks.N_BINS, 10), dtype=np.float32)
    prob[100:200, 0:2] = 0.9               # two columns, 200 pixels
    freq = masks.freq_axis_khz(5e5, 4)
    t_s = masks.col_times_s(10, 5e5, 4, 0.0)
    group = tracks.merge(tracks.components(prob >= 0.2))[0]
    got = tracks.descriptors(
        group, prob=prob, raw_logpow=np.ones_like(prob), freq_khz=freq, t_s=t_s
    )
    assert got.chirp_r2 == 0.0
    assert got.duration_ms == pytest.approx(SYNTH_HOP_S * 1e3)


def test_every_descriptor_of_every_synthetic_track_is_finite(synth):
    got, _, _, _, _ = synth
    for track in got:
        for name, value in tracks.as_attrs(track).items():
            assert isinstance(value, (bool, int, float, str)), name
            if isinstance(value, float):
                assert np.isfinite(value), name


def test_descriptors_refuses_axes_that_do_not_match_the_maps():
    prob = np.zeros((masks.N_BINS, 20), dtype=np.float32)
    prob[100:110, :] = 0.9
    group = tracks.components(prob >= 0.2, min_area=1)
    freq = masks.freq_axis_khz(5e5, 4)
    t_s = masks.col_times_s(20, 5e5, 4, 0.0)
    with pytest.raises(ValueError):
        tracks.descriptors(
            group[:1], prob=prob, raw_logpow=np.ones((masks.N_BINS, 19), np.float32),
            freq_khz=freq, t_s=t_s,
        )
    with pytest.raises(ValueError):
        tracks.descriptors(
            group[:1], prob=prob, raw_logpow=np.ones_like(prob),
            freq_khz=freq[:100], t_s=t_s,
        )
    with pytest.raises(ValueError):
        tracks.descriptors(
            group[:1], prob=prob, raw_logpow=np.ones_like(prob),
            freq_khz=freq, t_s=t_s[:5],
        )


def test_a_track_needs_a_component():
    with pytest.raises(ValueError):
        tracks.descriptors(
            [], prob=np.zeros((masks.N_BINS, 4), np.float32),
            raw_logpow=np.zeros((masks.N_BINS, 4), np.float32),
            freq_khz=masks.freq_axis_khz(5e5, 4),
            t_s=masks.col_times_s(4, 5e5, 4, 0.0),
        )


# --------------------------------------------------------------- harmonics

def test_the_fundamental_finds_its_second_and_third_harmonics(synth):
    got, _, _, _, _ = synth
    assert tracks.harmonics(got)[EHO] == [H2, H3]


def test_a_partner_that_never_coexists_is_not_a_harmonic(synth):
    # The fishbone's centroid is within tolerance of twice the
    # fundamental's, and it is excluded for one reason only: the two are
    # never on the spectrogram at the same time.
    got, _, _, _, _ = synth
    assert FISHBONE not in tracks.harmonics(got)[EHO]
    assert FISHBONE in tracks.harmonics(got, min_overlap_s=-10.0)[EHO]


def test_a_centroid_outside_the_tolerance_is_not_a_harmonic(synth):
    got, _, _, _, _ = synth
    assert tracks.harmonics(got, tol=0.001)[EHO] == []
    assert SPLIT_A not in tracks.harmonics(got)[EHO]      # 44 kHz is nobody's


def test_n_harmonics_counts_the_partners_of_every_track(synth):
    got, _, _, _, _ = synth
    counts = tracks.n_harmonics(got)
    assert len(counts) == len(got)
    assert counts[EHO] == 2
    assert counts[H2] == counts[H3] == counts[FISHBONE] == 0
    assert counts[SPLIT_A] == counts[SPLIT_B] == 0
    # The pickup line at 1.95 kHz has the 8 kHz fundamental as a fourth
    # harmonic to within 3%; a class-agnostic count says so.
    assert counts[PICKUP] == 1


def test_harmonics_of_nothing():
    assert tracks.harmonics([]) == {}
    assert tracks.n_harmonics([]) == []


# ------------------------------------------------------------ cooccurrence

def _track(t0, t1, f0, f1):
    return tracks.Track(
        t0_s=t0, t1_s=t1, f0_khz=f0, f1_khz=f1,
        f_centroid_khz=(f0 + f1) / 2, chirp_khz_per_ms=0.0, chirp_r2=0.0,
        bandwidth_khz=f1 - f0, duration_ms=(t1 - t0) * 1e3, duty=1.0,
        mean_prob=0.9, conf=0.9, n_pix=1000, n_components=1,
        row_lit_fraction=0.1, pickup=False,
    )


def test_two_channels_seeing_the_same_box_co_occur():
    by_channel = {
        "mhr_00_zoom": [_track(1.0, 1.1, 8.0, 9.0)],
        "ece_08_zoom": [_track(1.0, 1.1, 8.0, 9.0)],
    }
    assert tracks.cooccurrence(by_channel) == [
        {("ece_08_zoom", 0), ("mhr_00_zoom", 0)}
    ]


def test_boxes_that_barely_overlap_do_not_co_occur():
    by_channel = {
        "a": [_track(1.0, 1.1, 8.0, 9.0)],
        "b": [_track(1.09, 1.2, 8.0, 9.0)],       # IoU 0.01/0.21
    }
    assert tracks.cooccurrence(by_channel) == []


def test_cooccurrence_is_transitive():
    # a-b overlap and b-c overlap; a-c do not, and all three are one group.
    by_channel = {
        "a": [_track(1.00, 1.20, 8.0, 9.0)],
        "b": [_track(1.05, 1.25, 8.0, 9.0)],
        "c": [_track(1.10, 1.30, 8.0, 9.0)],
    }
    got = tracks.cooccurrence(by_channel)
    assert got == [{("a", 0), ("b", 0), ("c", 0)}]


def test_two_tracks_on_one_channel_are_not_a_coincidence():
    # The same channel seeing the same mode twice is one detector, not two.
    by_channel = {"a": [_track(1.0, 1.1, 8.0, 9.0), _track(1.0, 1.1, 8.0, 9.0)]}
    assert tracks.cooccurrence(by_channel) == []


def test_cooccurrence_of_nothing():
    assert tracks.cooccurrence({}) == []
    assert tracks.cooccurrence({"a": []}) == []


def test_degenerate_boxes_that_merely_touch_do_not_co_occur():
    # Two one-row tracks on the same row, sharing 1 ms of 100. Both boxes
    # have zero area, so the union is zero - which used to score 1.0 and
    # group them whatever `iou_min` said.
    by_channel = {
        "a": [_track(1.0, 1.1, 8.0, 8.0)],
        "b": [_track(1.099, 1.199, 8.0, 8.0)],
    }
    assert tracks.cooccurrence(by_channel) == []


def test_point_tracks_on_adjacent_bands_do_not_co_occur():
    # Two one-column tracks at the same instant whose bands touch at 9 kHz
    # and share nothing. Zero union again, and no coincidence.
    by_channel = {
        "a": [_track(1.0, 1.0, 8.0, 9.0)],
        "b": [_track(1.0, 1.0, 9.0, 10.0)],
    }
    assert tracks.cooccurrence(by_channel) == []


def test_genuinely_coincident_degenerate_boxes_still_co_occur():
    # The fix must not cost the case it is guarding: a degenerate box that
    # really is another channel's box is still one event seen twice.
    for a, b in (
        (_track(1.0, 1.0, 8.0, 9.0), _track(1.0, 1.0, 8.0, 9.0)),   # points
        (_track(1.0, 1.1, 8.0, 8.0), _track(1.0, 1.1, 8.0, 8.0)),   # one row
        (_track(1.0, 1.0, 8.0, 8.0), _track(1.0, 1.0, 8.0, 8.0)),   # a pixel
    ):
        assert tracks.cooccurrence({"a": [a], "b": [b]}) == [
            {("a", 0), ("b", 0)}
        ]


def test_the_iou_of_a_zero_union_pair_is_a_one_dimensional_one():
    # One-row tracks: the frequency axis has no extent to share, so the
    # score is the overlap fraction of the time axis alone.
    a = _track(1.0, 1.1, 8.0, 8.0)
    assert tracks._iou(a, _track(1.05, 1.15, 8.0, 8.0)) == pytest.approx(
        0.05 / 0.15
    )
    # Point tracks: the time axis is the degenerate one, frequency scores.
    c = _track(1.0, 1.0, 8.0, 9.0)
    assert tracks._iou(c, _track(1.0, 1.0, 8.5, 9.5)) == pytest.approx(0.5 / 1.5)
    # The degenerate axis has to agree: a point 200 ms later is elsewhere.
    assert tracks._iou(c, _track(1.2, 1.2, 8.0, 9.0)) == 0.0
    # Both axes degenerate - a single pixel - matches itself and nothing else.
    e = _track(1.0, 1.0, 8.0, 8.0)
    assert tracks._iou(e, _track(1.0, 1.0, 8.0, 8.0)) == 1.0
    assert tracks._iou(e, _track(1.0, 1.0, 9.0, 9.0)) == 0.0
    # A row crossing a column has no axis on which both boxes have extent;
    # there is no overlap fraction to take, and touching is not matching.
    assert tracks._iou(c, _track(0.9, 1.1, 8.5, 8.5)) == 0.0


# -------------------------------------------------------------- the events

def _events(got):
    return tracks.tracks_to_events(
        got, shot=198658, diag="mhr", channel=0, pass_name="zoom",
        t_cov=(0.0, 6.5), unet_sha256="b" * 64,
    )


def test_a_track_becomes_one_class_agnostic_event(synth):
    got, _, _, _, _ = synth
    events = _events(got)
    assert len(events) == len(got)
    event = events[EHO]
    assert event.shot == 198658
    assert event.source == "tokeye_track"
    assert event.evidence_kind == "detector"
    assert event.phenomenon == "coherent_mode"
    assert (event.t0_s, event.t1_s) == (got[EHO].t0_s, got[EHO].t1_s)
    assert (event.f0_khz, event.f1_khz) == (got[EHO].f0_khz, got[EHO].f1_khz)
    assert event.confidence == got[EHO].conf
    assert (event.diag, event.channel, event.pass_name) == ("mhr", 0, "zoom")
    assert (event.t_cov0_s, event.t_cov1_s) == (0.0, 6.5)


def test_a_pickup_track_is_not_called_a_mode(synth):
    got, _, _, _, _ = synth
    events = _events(got)
    assert events[PICKUP].phenomenon == "pickup"
    assert [e.phenomenon for e in events].count("pickup") == 1


def test_the_attrs_carry_the_whole_track_and_survive_json(synth):
    got, _, _, _, _ = synth
    attrs = _events(got)[EHO].attrs
    assert set(attrs) == set(tracks.as_attrs(got[EHO])) | {
        "n_harmonics", "unet_sha256"
    }
    assert attrs["n_components"] == 2
    assert attrs["n_harmonics"] == 2
    assert attrs["unet_sha256"] == "b" * 64
    assert attrs["pickup"] is False
    assert json.loads(json.dumps(attrs, allow_nan=False))["duty"] == attrs["duty"]


def test_the_events_go_into_the_shots_table(synth, tmp_path):
    got, _, _, _, _ = synth
    path = tmp_path / "198658_events.parquet"
    schema.write_events(path, 198658, _events(got), run_id="t")
    back = schema.read_events(path, source="tokeye_track")
    assert len(back) == 7
    assert set(back["phenomenon"]) == {"coherent_mode", "pickup"}
    assert json.loads(back["attrs"].iloc[0])["unet_sha256"] == "b" * 64


def test_events_of_no_tracks():
    assert _events([]) == []


# -------------------------------------------------------- tracks_for_block

def test_tracks_for_block_reads_a_stored_block(synth_mask, tmp_path):
    prob, raw, freq_khz, t_s = synth_mask()
    block = masks.MaskBlock(
        diag="mhr", channel=0, pass_name="zoom",
        coh=prob, tra=np.zeros_like(prob), raw_logpow=raw, t_s=t_s,
        meta={"fs_hz": 5e5, "decim": 4, "n_cols": prob.shape[1]},
    )
    path = tmp_path / "198658_masks.npz"
    masks.write_masks(
        path, 198658, [masks.block_arrays(block, unet_sha256="c" * 64)]
    )
    got = tracks.tracks_for_block("mhr_00_zoom", path)
    assert len(got) == 7
    assert [t.n_components for t in got] == [1, 2, 1, 1, 1, 1, 1]
    assert got[PICKUP].pickup is True
    # Only the boolean mask survives the packing, so there are no
    # probabilities to be confident with: unknown, not certain.
    assert math.isnan(got[EHO].mean_prob) and math.isnan(got[EHO].conf)
    assert got[EHO].f_centroid_khz == pytest.approx(
        freq_khz[_synth_band(freq_khz, SYNTH_EHO_KHZ)[0] + 1]
    )
    assert got[FISHBONE].chirp_khz_per_ms == pytest.approx(
        SYNTH_CHIRP_KHZ_PER_MS, abs=0.05
    )


def test_a_stored_blocks_unknown_confidence_survives_to_the_events_table(
    synth_mask, tmp_path
):
    # A track read back from a packed mask has no probability behind it, so
    # `conf` is NaN, `as_attrs` writes it as JSON null, and the event row
    # carries the NaN rather than a 1.0 nobody measured.
    prob, raw, _, t_s = synth_mask()
    block = masks.MaskBlock(
        diag="mhr", channel=0, pass_name="zoom",
        coh=prob, tra=np.zeros_like(prob), raw_logpow=raw, t_s=t_s,
        meta={"fs_hz": 5e5, "decim": 4, "n_cols": prob.shape[1]},
    )
    path = tmp_path / "198658_masks.npz"
    masks.write_masks(
        path, 198658, [masks.block_arrays(block, unet_sha256="c" * 64)]
    )
    got = tracks.tracks_for_block("mhr_00_zoom", path)
    attrs = tracks.as_attrs(got[EHO])
    assert attrs["conf"] is None and attrs["mean_prob"] is None
    assert attrs["duty"] == pytest.approx(got[EHO].duty)
    assert json.loads(json.dumps(attrs, allow_nan=False))["conf"] is None
    event = _events(got)[EHO]
    assert math.isnan(event.confidence)
    assert event.attrs["conf"] is None


def test_tracks_for_block_needs_a_block_that_is_there(tmp_path):
    prob = np.zeros((masks.N_BINS, 600), dtype=np.float32)
    block = masks.MaskBlock(
        diag="mhr", channel=0, pass_name="zoom", coh=prob, tra=prob,
        raw_logpow=prob, t_s=masks.col_times_s(600, 5e5, 4, 0.0),
        meta={"fs_hz": 5e5, "decim": 4},
    )
    path = tmp_path / "198658_masks.npz"
    masks.write_masks(
        path, 198658, [masks.block_arrays(block, unet_sha256="c" * 64)]
    )
    assert tracks.tracks_for_block("mhr_00_zoom", path) == []
    with pytest.raises(KeyError):
        tracks.tracks_for_block("ece_08_zoom", path)
