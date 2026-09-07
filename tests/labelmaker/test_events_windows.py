"""The 46 window features, and the grid they are computed on.

Every number here is drawn rather than measured: a rectangular lit region is
a lit fraction that can be written down, a ladder of log-power whose sixteen
band means are 0..15 makes a band average arithmetic, and the event families
are hand-written rows through the real `write_events`. So a failure names a
definition that changed, not a fixture that drifted.

The fixture's column grid is 1 ms, coarser than either real pass (0.256 ms
wide, 1.024 ms zoom), so a 1 s record is 1,001 columns instead of 3,907 -
but `fs_hz` and `decim` in each block's meta are the production ones,
because the band indices `band_logpow_{lo,mid,hi}` average over are derived
from them and would otherwise be a different set of bands.
"""
from __future__ import annotations

import numpy as np
import pytest

from labelmaker.config import Paths
from labelmaker.events import masks, schema, windows

SHOT = 198658
SHA = "0" * 64
RUN = "test-windows"

FS_HZ = 5.0e5
COL_S = 0.001
DECIM = {"wide": 1, "zoom": masks.ZOOM_DECIM}
N_ROWS = 512


# ------------------------------------------------------------- the fixtures

def _t_grid(t0: float = 0.0, t1: float = 1.0, step: float = COL_S) -> np.ndarray:
    n = round((t1 - t0) / step) + 1
    return t0 + step * np.arange(n, dtype=np.float64)


def _zeros(t_s) -> np.ndarray:
    return np.zeros((N_ROWS, np.asarray(t_s).size), dtype=np.float32)


def _lit(t_s, rows, cols=None, value: float = 1.0) -> np.ndarray:
    """A rectangle of lit pixels: half-open rows, half-open columns."""
    out = _zeros(t_s)
    c0, c1 = (0, np.asarray(t_s).size) if cols is None else cols
    out[rows[0]:rows[1], c0:c1] = value
    return out


def _band_ladder(t_s) -> np.ndarray:
    """Log-power whose sixteen 32-row band means are exactly 0, 1, .. 15."""
    per = N_ROWS // masks.N_BANDS
    rows = np.repeat(np.arange(masks.N_BANDS, dtype=np.float32), per)
    return np.repeat(rows[:, None], np.asarray(t_s).size, axis=1)


def _block(diag, channel, pass_name, t_s, *, coh=None, tra=None, raw_logpow=None):
    t_s = np.asarray(t_s, dtype=np.float64)
    block = masks.MaskBlock(
        diag=diag,
        channel=int(channel),
        pass_name=pass_name,
        coh=_zeros(t_s) if coh is None else np.asarray(coh, dtype=np.float32),
        tra=_zeros(t_s) if tra is None else np.asarray(tra, dtype=np.float32),
        raw_logpow=(
            _zeros(t_s) if raw_logpow is None
            else np.asarray(raw_logpow, dtype=np.float32)
        ),
        t_s=t_s,
        meta={
            "fs_hz": FS_HZ,
            "decim": DECIM[pass_name],
            "n_cols": int(t_s.size),
            "hop_s": COL_S,
            "freq_khz_per_bin": FS_HZ / DECIM[pass_name] / 1e3 / 1024,
        },
    )
    return masks.block_arrays(block, unet_sha256=SHA)


def _paths(tmp_path) -> Paths:
    paths = Paths(root=tmp_path / "root", corpus=tmp_path / "corpus")
    paths.mkdirs()
    return paths


def _write_masks(paths, blocks):
    return masks.write_masks(paths.masks_file(SHOT), SHOT, list(blocks))


def _write_events(paths, events):
    schema.write_events(paths.events_file(SHOT), SHOT, list(events), run_id=RUN)
    return schema.read_events(paths.events_file(SHOT))


def _elm(t: float, diag: str = "mhr") -> schema.Event:
    return schema.Event(
        shot=SHOT, source="tokeye_transient", phenomenon="elm",
        t0_s=t, t1_s=t, diag=diag, channel=0, pass_name="wide",
        t_cov0_s=0.0, t_cov1_s=10.0,
    )


def _interval(phenomenon: str, t0: float, t1: float, *, source: str,
              diag: str = "mhr", kind: str = "heuristic") -> schema.Event:
    return schema.Event(
        shot=SHOT, source=source, evidence_kind=kind, phenomenon=phenomenon,
        t0_s=t0, t1_s=t1, diag=diag, channel=0,
        t_cov0_s=0.0, t_cov1_s=10.0,
    )


def _track(t0, t1, *, diag="mhr", f_centroid=8.0, chirp=0.0, harmonics=0,
           conf=0.9, phenomenon="coherent_mode") -> schema.Event:
    return schema.Event(
        shot=SHOT, source="tokeye_track", phenomenon=phenomenon,
        t0_s=t0, t1_s=t1, f0_khz=f_centroid - 1.0, f1_khz=f_centroid + 1.0,
        confidence=conf, diag=diag, channel=0, pass_name="zoom",
        attrs={
            "f_centroid_khz": float(f_centroid),
            "chirp_khz_per_ms": float(chirp),
            "n_harmonics": int(harmonics),
            "duration_ms": float((t1 - t0) * 1e3),
        },
        t_cov0_s=0.0, t_cov1_s=10.0,
    )


def _f(vec, name: str) -> float:
    return float(np.asarray(vec)[windows.FEATURE_NAMES.index(name)])


# ------------------------------------------------------------ FEATURE_NAMES

def test_feature_names_are_the_frozen_forty_six():
    assert windows.FEATURE_NAMES == (
        "mhr_coh_lit_frac_wide",
        "mhr_coh_lit_frac_zoom",
        "mhr_tra_col_act_mean_wide",
        "mhr_tra_col_act_max_wide",
        "mhr_band_logpow_lo",
        "mhr_band_logpow_mid",
        "mhr_band_logpow_hi",
        "mhr_n_tracks",
        "mhr_track_f_centroid_khz",
        "mhr_track_abs_chirp_max",
        "mhr_track_n_harmonics_max",
        "mhr_track_duration_max_ms",
        "co2_coh_lit_frac_wide",
        "co2_coh_lit_frac_zoom",
        "co2_tra_col_act_mean_wide",
        "co2_tra_col_act_max_wide",
        "co2_band_logpow_lo",
        "co2_band_logpow_mid",
        "co2_band_logpow_hi",
        "co2_n_tracks",
        "co2_track_f_centroid_khz",
        "co2_track_abs_chirp_max",
        "co2_track_n_harmonics_max",
        "co2_track_duration_max_ms",
        "ece_coh_lit_frac_wide",
        "ece_coh_lit_frac_zoom",
        "ece_tra_col_act_mean_wide",
        "ece_tra_col_act_max_wide",
        "ece_band_logpow_lo",
        "ece_band_logpow_mid",
        "ece_band_logpow_hi",
        "ece_n_tracks",
        "ece_track_f_centroid_khz",
        "ece_track_abs_chirp_max",
        "ece_track_n_harmonics_max",
        "ece_track_duration_max_ms",
        "elm_rate_hz",
        "elm_free_frac",
        "time_since_last_elm_s",
        "n_sawtooth",
        "sawtooth_period_ms",
        "lh_recent",
        "pickup_flag",
        "cov_frac_mhr",
        "cov_frac_co2",
        "cov_frac_ece",
    )
    assert len(windows.FEATURE_NAMES) == 46
    assert len(set(windows.FEATURE_NAMES)) == 46
    assert windows.N_FEATURES == 46


def test_no_feature_carries_actuator_or_equilibrium_context():
    """Plan section 7: the diagnostic classifiers must not see the actuators.

    A prior that says "EHO happens when the beams are on and the plasma is
    in H-mode" would be trained on, and then evaluated against, the very
    conditions the annotator used to pick the windows. The 46 are therefore
    diagnostics-only: masks, tracks and the transient/heuristic event
    families, and nothing from `pinj`, `ech`, the RMP coils or EFIT.
    """
    forbidden = (
        "pinj", "tinj", "ech", "rmp", "gas", "ip", "bt", "betan", "kappa",
        "q95", "qmin", "li", "aminor", "volume", "efit", "zipfit", "nbi",
    )
    for name in windows.FEATURE_NAMES:
        parts = set(name.split("_"))
        assert not (parts & set(forbidden)), name


# -------------------------------------------------------------- window_grid

def test_the_window_and_stride_are_the_specified_ones():
    assert windows.WINDOW_S == 0.34
    assert windows.STRIDE_S == 0.17


def test_window_grid_tiles_the_coverage_and_truncates_the_last_window():
    centres, starts, ends = windows.window_grid(0.0, 1.0)
    np.testing.assert_allclose(starts, [0.0, 0.17, 0.34, 0.51, 0.68])
    np.testing.assert_allclose(ends, [0.34, 0.51, 0.68, 0.85, 1.0])
    np.testing.assert_allclose(centres, 0.5 * (starts + ends))
    assert ends[-1] == 1.0


def test_a_coverage_of_exactly_one_width_is_one_window():
    centres, starts, ends = windows.window_grid(1.0, 1.34)
    assert centres.size == 1
    np.testing.assert_allclose([starts[0], ends[0]], [1.0, 1.34])


def test_coverage_shorter_than_the_width_is_one_truncated_window():
    centres, starts, ends = windows.window_grid(2.0, 2.2)
    assert centres.size == 1
    np.testing.assert_allclose([starts[0], ends[0], centres[0]], [2.0, 2.2, 2.1])


def test_empty_coverage_is_no_windows_at_all():
    centres, starts, ends = windows.window_grid(1.0, 1.0)
    assert centres.size == starts.size == ends.size == 0


def test_a_custom_width_and_stride_are_honoured():
    _, starts, ends = windows.window_grid(0.0, 0.5, width_s=0.2, stride_s=0.1)
    np.testing.assert_allclose(starts, [0.0, 0.1, 0.2, 0.3])
    np.testing.assert_allclose(ends, [0.2, 0.3, 0.4, 0.5])


# ------------------------------------------------------------ band indices

def test_the_band_indices_are_the_documented_ones():
    """16 bands of 32 bins; a band's centre bin decides which pass band it is.

    Wide: 0.48828 kHz/bin, so band j spans bins 32j+1..32j+32 and is centred
    on 8.06 + 15.625j kHz. Zoom: 0.12207 kHz/bin, centred on 2.01 + 3.906j.
    """
    zoom_lo = windows.band_indices(FS_HZ, masks.ZOOM_DECIM, *windows.BAND_LO_KHZ)
    zoom_mid = windows.band_indices(FS_HZ, masks.ZOOM_DECIM, *windows.BAND_MID_KHZ)
    wide_hi = windows.band_indices(FS_HZ, 1, *windows.BAND_HI_KHZ)
    assert list(zoom_lo) == [0, 1, 2, 3, 4]
    assert list(zoom_mid) == [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    assert list(wide_hi) == [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    assert windows.BAND_LO_KHZ == (2.0, 20.0)
    assert windows.BAND_MID_KHZ == (20.0, 60.0)
    assert windows.BAND_HI_KHZ == (100.0, 250.0)


# ------------------------------------------------------- the mask families

@pytest.fixture
def masked(tmp_path):
    """A masks file: mhr on both passes, ece wide only, no co2 at all.

    * mhr wide: rows 100:110 lit in every column - 10 of 512 rows;
    * mhr zoom: rows 0:64 lit in every column - 64 of 512;
    * mhr wide transient: rows 0:128 lit in every column (activity 0.25),
      and rows 0:256 in column 100 alone (0.5), so the mean and the max of
      the window at 0.0 differ;
    * mhr band power: the 0..15 ladder on both passes;
    * ece wide: rows 0:256 lit in columns 0:100 only.
    """
    paths = _paths(tmp_path)
    t = _t_grid(0.0, 1.0)
    tra = _lit(t, (0, 128))
    tra[0:256, 100] = 1.0
    _write_masks(paths, [
        _block("mhr", 0, "wide", t, coh=_lit(t, (100, 110)), tra=tra,
               raw_logpow=_band_ladder(t)),
        _block("mhr", 0, "zoom", t, coh=_lit(t, (0, 64)),
               raw_logpow=_band_ladder(t)),
        _block("ece", 0, "wide", t, coh=_lit(t, (0, 256), cols=(0, 100))),
    ])
    _write_events(paths, [])
    return paths, t


def _blocks_and_cov(paths):
    blocks = windows.blocks_from_masks(paths.masks_file(SHOT))
    return blocks, windows.coverage_from_blocks(blocks)


def test_the_lit_fraction_is_the_rectangle_that_was_drawn(masked):
    paths, _ = masked
    blocks, cov = _blocks_and_cov(paths)
    events = schema.read_events(paths.events_file(SHOT))
    vec = windows.window_features((0.0, 0.34), blocks=blocks, events=events, cov=cov)
    assert _f(vec, "mhr_coh_lit_frac_wide") == pytest.approx(10 / 512)
    assert _f(vec, "mhr_coh_lit_frac_zoom") == pytest.approx(64 / 512)
    # ece has no zoom block at all, and its wide rectangle stops at column
    # 100, which is inside this window.
    assert _f(vec, "ece_coh_lit_frac_zoom") == 0.0
    assert _f(vec, "ece_coh_lit_frac_wide") > 0.0


def test_a_window_past_the_lit_columns_is_lit_nowhere(masked):
    paths, _ = masked
    blocks, cov = _blocks_and_cov(paths)
    events = schema.read_events(paths.events_file(SHOT))
    vec = windows.window_features((0.5, 0.84), blocks=blocks, events=events, cov=cov)
    assert _f(vec, "ece_coh_lit_frac_wide") == 0.0
    assert _f(vec, "mhr_coh_lit_frac_wide") == pytest.approx(10 / 512)


def test_the_column_activity_mean_and_max_come_from_col_act(masked):
    paths, t = masked
    blocks, cov = _blocks_and_cov(paths)
    events = schema.read_events(paths.events_file(SHOT))
    vec = windows.window_features((0.0, 0.34), blocks=blocks, events=events, cov=cov)
    n = int(((t >= 0.0) & (t < 0.34)).sum())
    assert _f(vec, "mhr_tra_col_act_max_wide") == pytest.approx(0.5)
    assert _f(vec, "mhr_tra_col_act_mean_wide") == pytest.approx(
        (0.25 * (n - 1) + 0.5) / n
    )
    assert _f(vec, "ece_tra_col_act_mean_wide") == 0.0


def test_the_band_powers_average_the_stored_bands_of_their_pass(masked):
    paths, _ = masked
    blocks, cov = _blocks_and_cov(paths)
    events = schema.read_events(paths.events_file(SHOT))
    vec = windows.window_features((0.0, 0.34), blocks=blocks, events=events, cov=cov)
    assert _f(vec, "mhr_band_logpow_lo") == pytest.approx(np.mean([0, 1, 2, 3, 4]))
    assert _f(vec, "mhr_band_logpow_mid") == pytest.approx(np.mean(range(5, 15)))
    assert _f(vec, "mhr_band_logpow_hi") == pytest.approx(np.mean(range(6, 16)))
    # ece has no zoom block, so the two zoom bands are 0 while the wide one
    # is the floor the fixture drew (a flat zero log-power).
    assert _f(vec, "ece_band_logpow_lo") == 0.0
    assert _f(vec, "ece_band_logpow_hi") == 0.0


def test_coverage_fractions_and_an_absent_diagnostic(masked):
    paths, _ = masked
    blocks, cov = _blocks_and_cov(paths)
    events = schema.read_events(paths.events_file(SHOT))
    vec = windows.window_features((0.0, 0.34), blocks=blocks, events=events, cov=cov)
    assert _f(vec, "cov_frac_mhr") == pytest.approx(1.0)
    assert _f(vec, "cov_frac_ece") == pytest.approx(1.0)
    assert _f(vec, "cov_frac_co2") == 0.0
    for name in windows.FEATURE_NAMES:
        if name.startswith("co2_"):
            assert _f(vec, name) == 0.0, name


def test_a_window_half_outside_the_coverage_is_half_covered(masked):
    paths, _ = masked
    blocks, cov = _blocks_and_cov(paths)
    events = schema.read_events(paths.events_file(SHOT))
    vec = windows.window_features((0.83, 1.17), blocks=blocks, events=events, cov=cov)
    assert _f(vec, "cov_frac_mhr") == pytest.approx(0.5)


# ------------------------------------------------------ the track families

@pytest.fixture
def track_events(tmp_path):
    """Four tokeye_track rows, three of them on `mhr`."""
    paths = _paths(tmp_path)
    return paths, _write_events(paths, [
        _track(0.05, 0.15, f_centroid=10.0, chirp=0.2, harmonics=2, conf=0.8),
        _track(0.20, 0.30, f_centroid=20.0, chirp=-0.9, harmonics=1, conf=0.2),
        _track(0.60, 0.70, f_centroid=99.0, chirp=5.0, harmonics=5, conf=1.0),
        _track(0.05, 0.30, diag="co2", f_centroid=44.0, chirp=0.0,
               harmonics=0, conf=0.5),
    ])


def test_track_statistics_come_from_the_overlapping_rows(track_events):
    _, events = track_events
    cov = {"mhr": (0.0, 1.0), "co2": (0.0, 1.0)}
    vec = windows.window_features((0.0, 0.34), blocks={}, events=events, cov=cov)
    assert _f(vec, "mhr_n_tracks") == 2.0
    assert _f(vec, "mhr_track_abs_chirp_max") == pytest.approx(0.9)
    assert _f(vec, "mhr_track_n_harmonics_max") == 2.0
    assert _f(vec, "mhr_track_duration_max_ms") == pytest.approx(100.0)
    assert _f(vec, "co2_n_tracks") == 1.0
    assert _f(vec, "co2_track_duration_max_ms") == pytest.approx(250.0)


def test_the_track_centroid_is_confidence_weighted(track_events):
    _, events = track_events
    cov = {"mhr": (0.0, 1.0)}
    vec = windows.window_features((0.0, 0.34), blocks={}, events=events, cov=cov)
    assert _f(vec, "mhr_track_f_centroid_khz") == pytest.approx(
        (0.8 * 10.0 + 0.2 * 20.0) / (0.8 + 0.2)
    )


def test_a_window_with_no_track_reports_zeros(track_events):
    _, events = track_events
    cov = {"mhr": (0.0, 1.0)}
    vec = windows.window_features((0.35, 0.55), blocks={}, events=events, cov=cov)
    for suffix in ("n_tracks", "track_f_centroid_khz", "track_abs_chirp_max",
                   "track_n_harmonics_max", "track_duration_max_ms"):
        assert _f(vec, f"mhr_{suffix}") == 0.0, suffix


# ------------------------------------------------------- the event families

def test_the_elm_rate_is_the_count_over_the_window_width(tmp_path):
    paths = _paths(tmp_path)
    events = _write_events(paths, [_elm(t) for t in (0.05, 0.15, 0.25, 0.40)])
    vec = windows.window_features(
        (0.0, 0.34), blocks={}, events=events, cov={"mhr": (0.0, 1.0)}
    )
    assert _f(vec, "elm_rate_hz") == pytest.approx(3 / 0.34)


def test_the_elm_free_fraction_is_the_overlap_with_the_quiet_intervals(tmp_path):
    paths = _paths(tmp_path)
    events = _write_events(paths, [
        _interval("elm_free", 0.20, 0.40, source="elm_clock"),
        _interval("elm_free", 0.90, 1.00, source="elm_clock"),
    ])
    vec = windows.window_features(
        (0.0, 0.34), blocks={}, events=events, cov={"mhr": (0.0, 1.0)}
    )
    assert _f(vec, "elm_free_frac") == pytest.approx(0.14 / 0.34)


def test_the_time_since_the_last_elm_is_capped_at_one_second(tmp_path):
    paths = _paths(tmp_path)
    events = _write_events(paths, [_elm(0.10), _elm(3.00)])
    near = windows.window_features(
        (0.0, 0.34), blocks={}, events=events, cov={"mhr": (0.0, 5.0)}
    )
    far = windows.window_features(
        (2.0, 2.34), blocks={}, events=events, cov={"mhr": (0.0, 5.0)}
    )
    before_any = windows.window_features(
        (0.0, 0.04), blocks={}, events=events, cov={"mhr": (0.0, 5.0)}
    )
    assert _f(near, "time_since_last_elm_s") == pytest.approx(0.17 - 0.10)
    assert _f(far, "time_since_last_elm_s") == pytest.approx(1.0)
    assert _f(before_any, "time_since_last_elm_s") == pytest.approx(1.0)


def test_the_sawtooth_count_and_the_median_crash_interval(tmp_path):
    paths = _paths(tmp_path)
    # 0.10, 0.18, 0.28, 0.32: intervals 80, 100, 40 ms, median 80 ms.
    events = _write_events(paths, [
        _interval("sawtooth", t, t, source="ece_sawtooth", diag="ece")
        for t in (0.10, 0.18, 0.28, 0.32)
    ])
    vec = windows.window_features(
        (0.0, 0.34), blocks={}, events=events, cov={"ece": (0.0, 1.0)}
    )
    assert _f(vec, "n_sawtooth") == 4.0
    assert _f(vec, "sawtooth_period_ms") == pytest.approx(80.0)


def test_one_sawtooth_near_the_centre_has_no_period(tmp_path):
    paths = _paths(tmp_path)
    events = _write_events(paths, [
        _interval("sawtooth", 0.10, 0.10, source="ece_sawtooth", diag="ece"),
        # more than 0.5 s from the centre at 0.17, so it does not pair up
        _interval("sawtooth", 0.90, 0.90, source="ece_sawtooth", diag="ece"),
    ])
    vec = windows.window_features(
        (0.0, 0.34), blocks={}, events=events, cov={"ece": (0.0, 1.0)}
    )
    assert _f(vec, "n_sawtooth") == 1.0
    assert _f(vec, "sawtooth_period_ms") == 0.0


def test_lh_recent_is_the_half_second_before_the_centre_half_open(tmp_path):
    """`(centre - 0.5, centre]` - inclusive at the centre, exclusive behind.

    The four windows are chosen so that every bound is exact in binary and
    the two boundary cases are the assertions, not a rounding accident: the
    L-H transition sits at exactly 1.0 s, and the centres are 1.0 (on it),
    1.25 (a quarter second after), 1.5 (exactly half a second after) and
    0.625 (before it happened).
    """
    paths = _paths(tmp_path)
    events = _write_events(paths, [
        _interval("lh_transition", 1.0, 1.0, source="dalpha_lh"),
    ])

    def at(t0, t1):
        return windows.window_features(
            (t0, t1), blocks={}, events=events, cov={"mhr": (0.0, 5.0)}
        )

    assert _f(at(0.75, 1.25), "lh_recent") == 1.0    # centre 1.0, exactly on it
    assert _f(at(1.0, 1.5), "lh_recent") == 1.0      # centre 1.25, inside
    assert _f(at(1.25, 1.75), "lh_recent") == 0.0    # centre 1.5, exactly 0.5 s
    assert _f(at(0.5, 0.75), "lh_recent") == 0.0     # centre 0.625, still ahead


def test_the_pickup_flag_is_any_overlapping_pickup_row(tmp_path):
    paths = _paths(tmp_path)
    events = _write_events(paths, [
        _track(0.30, 0.90, phenomenon="pickup", f_centroid=1.95, conf=0.95),
    ])
    over = windows.window_features(
        (0.0, 0.34), blocks={}, events=events, cov={"mhr": (0.0, 1.0)}
    )
    clear = windows.window_features(
        (0.90, 1.00), blocks={}, events=events, cov={"mhr": (0.0, 1.0)}
    )
    assert _f(over, "pickup_flag") == 1.0
    assert _f(clear, "pickup_flag") == 0.0


def test_an_empty_events_table_leaves_every_event_feature_at_zero(tmp_path):
    paths = _paths(tmp_path)
    events = _write_events(paths, [])
    vec = windows.window_features(
        (0.0, 0.34), blocks={}, events=events, cov={"mhr": (0.0, 1.0)}
    )
    for name in ("elm_rate_hz", "elm_free_frac", "n_sawtooth",
                 "sawtooth_period_ms", "lh_recent", "pickup_flag"):
        assert _f(vec, name) == 0.0, name
    # "no ELM yet" is the cap, not zero: zero would read as "one just now".
    assert _f(vec, "time_since_last_elm_s") == pytest.approx(1.0)
    assert vec.shape == (46,)
    assert vec.dtype == np.float32


# ---------------------------------------------------- shot_window_features

def test_a_shot_becomes_a_grid_of_windows_and_a_validity_mask(masked):
    paths, _ = masked
    centres, x, valid = windows.shot_window_features(SHOT, paths)
    assert centres.size == 5
    assert x.shape == (46, 5)
    assert x.dtype == np.float32
    assert valid.shape == (5,)
    np.testing.assert_allclose(centres, [0.17, 0.34, 0.51, 0.68, 0.84], atol=1e-9)
    assert valid.dtype == np.bool_
    assert valid.all()
    lit = x[windows.FEATURE_NAMES.index("mhr_coh_lit_frac_wide")]
    np.testing.assert_allclose(lit, 10 / 512, rtol=1e-6)


def test_the_grid_spans_the_widest_diagnostic_and_the_others_thin_out(tmp_path):
    """Coverage is per diagnostic; the grid is their union.

    One diagnostic alone gives a grid inside its own coverage, where
    `cov_frac` is 1 throughout. Add a second, much longer one and the grid
    grows to the union - so the short diagnostic's `cov_frac` falls to 0 at
    the end while the window stays valid on the strength of the long one.
    """
    paths = _paths(tmp_path)
    t = _t_grid(0.0, 0.40)
    _write_masks(paths, [_block("mhr", 0, "wide", t, coh=_lit(t, (0, 8)))])
    _write_events(paths, [])
    centres, x, valid = windows.shot_window_features(SHOT, paths)
    assert centres.size == 2
    assert valid.tolist() == [True, True]
    cov = x[windows.FEATURE_NAMES.index("cov_frac_mhr")]
    np.testing.assert_allclose(cov, [1.0, 1.0], atol=1e-6)

    # Now the same record with a second, much longer diagnostic: `ece` runs
    # to 2 s, so the grid is long and every window past 0.40 s has no `mhr`
    # in it at all - which is exactly the half-covered case.
    t_long = _t_grid(0.0, 2.0)
    _write_masks(paths, [_block("ece", 0, "wide", t_long, coh=_lit(t_long, (0, 8)))])
    centres, x, valid = windows.shot_window_features(SHOT, paths)
    mhr = x[windows.FEATURE_NAMES.index("cov_frac_mhr")]
    assert mhr[0] == pytest.approx(1.0)
    assert mhr[-1] == pytest.approx(0.0)
    assert valid.all()          # `ece` covers all of them


def test_no_diagnostic_covering_half_a_window_makes_it_invalid(tmp_path):
    paths = _paths(tmp_path)
    # One diagnostic, 0.5 s of coverage, and a window grid that runs past it
    # only because a second diagnostic covers a sliver at the end.
    t_short = _t_grid(0.0, 0.5)
    t_sliver = _t_grid(1.30, 1.34)
    _write_masks(paths, [
        _block("mhr", 0, "wide", t_short, coh=_lit(t_short, (0, 8))),
        _block("ece", 0, "wide", t_sliver, coh=_lit(t_sliver, (0, 8))),
    ])
    _write_events(paths, [])
    _, x, valid = windows.shot_window_features(SHOT, paths)
    assert not valid.all()
    covered = np.maximum(
        x[windows.FEATURE_NAMES.index("cov_frac_mhr")],
        x[windows.FEATURE_NAMES.index("cov_frac_ece")],
    )
    np.testing.assert_array_equal(valid, covered >= 0.5)


def test_a_missing_masks_file_names_the_file_it_wanted(tmp_path):
    paths = _paths(tmp_path)
    _write_events(paths, [])
    with pytest.raises(FileNotFoundError, match=str(paths.masks_file(SHOT))):
        windows.shot_window_features(SHOT, paths)


def test_a_missing_events_file_names_the_file_it_wanted(tmp_path):
    paths = _paths(tmp_path)
    t = _t_grid(0.0, 1.0)
    _write_masks(paths, [_block("mhr", 0, "wide", t, coh=_lit(t, (0, 8)))])
    with pytest.raises(FileNotFoundError, match=str(paths.events_file(SHOT))):
        windows.shot_window_features(SHOT, paths)


def test_the_stored_blocks_come_back_with_their_diagnostic_and_pass(masked):
    paths, t = masked
    blocks = windows.blocks_from_masks(paths.masks_file(SHOT))
    assert sorted(blocks) == ["ece", "mhr"]
    assert sorted(b.pass_name for b in blocks["mhr"]) == ["wide", "zoom"]
    one = blocks["mhr"][0]
    assert one.diag == "mhr" and one.channel == 0
    assert one.t_s.size == t.size
    assert one.band_logpow.shape == (masks.N_BANDS, t.size)
    cov = windows.coverage_from_blocks(blocks)
    assert cov["mhr"] == pytest.approx((0.0, 1.0))
    assert "co2" not in cov
