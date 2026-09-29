"""Round three, Task 2.6: sub-frame features from a review store's rows."""

from __future__ import annotations

import dataclasses

import h5py
import numpy as np
import pytest

from labeler import frames
from labeler.ae.xpower.data import window_frames
from labeler.events.panels._shared import CLIPPED
from labeler.events.panels.neoclassical_tearing_mode import N_COLOURS, Z_DB
from labeler.events.review import rows
from labeler.frames.features import StaleStore, features, find_role, level_for
from labeler.scoring.frames import FRAME_MS

from . import frames_tree
from .frames_tree import (
    DT_MS,
    ECE_PHASE_MS,
    ELM,
    EVENT_MS,
    HMODE,
    HMODE_DROP_MS,
    HMODE_FS,
    NTM_BIN,
    NTM_BINS,
    NTM_DB,
    NTM_N,
    SAWTOOTH_MS,
    SHOTS,
    STORE_T0_MS,
    WINDOW,
)

#: Each spec's feature channels: a trace's minima then its maxima (one of each
#: when pooled), an image's groups, a modes row's n, an optional role's presence.
WIDTHS = {"elm_frames": 2, "hmode_frames": 5, "ntm_frames": 42, "sawtooth_frames": 41}
#: `verify.mode_bytes`' codes for the tearing-mode panel's 10 n: 25 levels, 0-24.
K = len(N_COLOURS)
TOP = 256 // K - 1
MODES = {
    "n": sorted(N_COLOURS),
    "levels": 256 // K,
    "colours": [N_COLOURS[v] for v in sorted(N_COLOURS)],
}
#: The n row's title before d9fb57d: n values, no modes meta (D65).
OLD_N_TITLE = "toroidal mode number n (MPI66M probes)"


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    return frames_tree.build(tmp_path_factory.mktemp("frames"))


def _store(tree, method):
    spec = frames.SPECS[method]
    return frames.store_path(tree, spec, SHOTS[spec.store_event])


def _features(tree, method, window=WINDOW):
    return features(_store(tree, method), frames.SPECS[method], window)


def _starts(spec, window=WINDOW) -> np.ndarray:
    """The window's sub-frames' start times, ms."""
    first, n = window_frames(window)
    subs = n * round(FRAME_MS / spec.sub_ms)
    return first * FRAME_MS + np.arange(subs) * spec.sub_ms


def _during(spec, span=EVENT_MS) -> np.ndarray:
    """Whether each of the window's sub-frames lies inside `span`."""
    starts = _starts(spec)
    return (starts >= span[0]) & (starts + spec.sub_ms <= span[1])


def _ntm_store(path, codes, *, title="toroidal n, MPI66M probes", modes=MODES):
    """A small tearing-mode store: 5 ms columns from 0 ms, 0.1 kHz bins, a dark
    power row and an n row holding `codes`."""
    axis = {
        "y0": 0.0,
        "dy": 0.1,
        "y_units": "kHz",
        "z_lo": Z_DB[0],
        "z_hi": Z_DB[1],
        "z_units": "",
    }
    power = np.zeros(codes.shape, dtype=np.uint8)
    built = [
        rows.ImageRow("p0", "MPI66M322D power", power, **axis),
        rows.ImageRow("p1", title, codes, **axis, modes=modes),
    ]
    rows.write(path, rows.Grid(0.0, 5.0, codes.shape[1]), built)


def _hmode_rows() -> tuple[rows.Grid, list]:
    """The tree's H-mode rows on their grid, in memory."""
    dt = DT_MS[HMODE]
    grid = rows.Grid(STORE_T0_MS, dt, round(frames_tree.STORE_MS / dt))
    t = grid.t0_ms + (np.arange(grid.n) + 0.5) * dt
    return grid, frames_tree.STORE_ROWS[HMODE](t, np.random.default_rng(1))


def test_level_for_takes_the_coarsest_level_with_at_least_four_columns_a_sub_frame():
    assert level_for({1: 0.256, 8: 2.05, 64: 16.4}, 2) == 1
    assert level_for({1: 0.256, 8: 2.05, 64: 16.4}, 10) == 8
    assert level_for({1: 2.56, 8: 20.48, 64: 163.84}, 2) == 1  # none fits: the finest
    # The tree's stores, as the real ones (M7): the ELM store's 0.05 ms columns
    # at level 8 are 0.4 ms, five to a 2 ms sub-frame; the H-mode store's 0.1 ms
    # at level 8 are 0.8 ms, 12 or 13 to 10 ms (level 64's 6.4 ms would give 1 or 2).
    want = {"elm_frames": 8, "hmode_frames": 8, "ntm_frames": 1, "sawtooth_frames": 8}
    for method, spec in frames.SPECS.items():
        by_level = {lv: lv * DT_MS[spec.store_event] for lv in rows.LEVELS}
        assert level_for(by_level, spec.sub_ms) == want[method], method
    assert frames.SPECS["elm_frames"].sub_ms / (8 * DT_MS[ELM]) == pytest.approx(5)
    # The legacy ELM stores Task 2.8 reads: 0.1 ms columns stay at level 1 for
    # 2 ms sub-frames (level 8's 0.8 ms would give 2 or 3).
    assert level_for({lv: lv * 0.1 for lv in rows.LEVELS}, 2.0) == 1


def test_each_spec_s_features_cover_the_window_s_sub_frames(tree):
    _, n = window_frames(WINDOW)
    for method, spec in frames.SPECS.items():
        x, observed = _features(tree, method)
        assert x.dtype == np.float32 and x.shape == (WIDTHS[method], len(_starts(spec)))
        assert x.shape[1] == n * FRAME_MS / spec.sub_ms
        assert np.isfinite(x).all() and x.min() >= 0 and x.max() <= 1, method
        assert observed.dtype == bool and observed.shape == (n,) and observed.all()
    # Past the store's end (2100 ms) no column falls: 0, and the frames unobserved.
    x, observed = _features(tree, "sawtooth_frames", (100, 2300))
    assert observed.shape == (220,)
    assert observed[:200].all() and not observed[200:].any()
    assert x[:, :1000].any() and not x[:, 1000:].any()


def test_an_image_row_gives_its_groups_in_zero_to_one(tree):
    spec = frames.SPECS["ntm_frames"]
    power = _features(tree, "ntm_frames")[0][: spec.roles[0].groups]
    assert power.shape[0] == 32 and power.min() >= 0 and power.max() <= 1
    # 308 bins in 32 groups: the line's bin 102 is group 10's.
    groups = np.array_split(np.arange(NTM_BINS), 32)
    (line,) = [g for g, bins in enumerate(groups) if NTM_BIN in bins]
    lo, hi = Z_DB
    lit = np.rint((NTM_DB - lo) * 255 / (hi - lo)) / 255  # the line's byte, 0-1
    during = _during(spec)
    assert power[line, during] == pytest.approx(lit)
    assert power[line, ~during].max() < 0.2
    assert np.delete(power, line, axis=0).max() < 0.2


def test_a_trace_s_max_and_min_share_one_range(tree):
    """D64: one affine map takes both extremes, so a sub-frame's spread survives."""
    spec = frames.SPECS["elm_frames"]
    path = _store(tree, "elm_frames")
    x, _ = features(path, spec, WINDOW)
    name = find_role(rows.meta(path)["rows"], spec.roles[0])["name"]
    with h5py.File(path, "r") as f:
        fine = f["rows"][name]["1"][:, 0]  # each 0.05 ms column's minimum, maximum
    per = round(spec.sub_ms / DT_MS[ELM])
    i0 = round((WINDOW[0] - STORE_T0_MS) / DT_MS[ELM])
    block = fine[:, i0 : i0 + per * x.shape[1]].reshape(2, x.shape[1], per)
    raw = np.concatenate([block[0].min(axis=1), block[1].max(axis=1)])
    scaled = np.concatenate([x[0], x[1]])
    slope, intercept = np.polyfit(raw, scaled, 1)
    assert slope > 0
    np.testing.assert_allclose(slope * raw + intercept, scaled, atol=1e-5)
    # So the minima stay flat while the maxima carry the ELMs.
    assert np.ptp(x[0]) < 0.05 and np.ptp(x[1]) > 0.2


def test_the_modes_decode_each_n_s_highest_level(tmp_path, tree):
    spec = frames.SPECS["ntm_frames"]
    codes = np.zeros((400, 40), dtype=np.uint8)
    codes[50, 4] = 7 * K + 3
    codes[55, 4] = 5 * K + 3  # the same n, lower: its highest level counts
    codes[60, 4] = 2 * K + 5
    codes[350, 4] = 20 * K + 1  # 35 kHz, above the band
    codes[70, 7] = TOP * K + 9
    _ntm_store(tmp_path / "modes.h5", codes)
    x, observed = features(tmp_path / "modes.h5", spec, (0, 200))
    assert x.shape == (42, 40) and observed.all()
    want = np.zeros((K, 40))
    want[3, 4], want[5, 4], want[9, 7] = 7 / TOP, 2 / TOP, 1.0
    np.testing.assert_allclose(x[32:], want, rtol=1e-6)
    # The tree's line, coded by `mode_bytes`: its n at its level over EVENT_MS.
    x, _ = _features(tree, "ntm_frames")
    level = np.rint((NTM_DB - Z_DB[0]) / (Z_DB[1] - Z_DB[0]) * TOP)
    line = x[32 + sorted(N_COLOURS).index(NTM_N)]
    during = _during(spec)
    assert line[during] == pytest.approx(level / TOP)
    assert line[~during].max() < 0.2


def test_a_missing_optional_role_is_zeros_and_a_presence_of_0(tmp_path, tree):
    x, observed = _features(tree, "sawtooth_frames")  # the tree has no SXR row
    assert not x[32:].any() and observed.all()
    # The H-mode rows with and without NBI: the same D-alpha either way, and NBI's
    # values and a presence of 1, or zeros and a presence of 0.
    spec = frames.SPECS["hmode_frames"]
    grid, built = _hmode_rows()
    rows.write(tmp_path / "with.h5", grid, built)
    rows.write(tmp_path / "without.h5", grid, [r for r in built if r.name != "p2"])
    x, _ = features(tmp_path / "with.h5", spec, WINDOW)
    y, observed = features(tmp_path / "without.h5", spec, WINDOW)
    assert (x[4] == 1).all() and x[2:4].any()
    np.testing.assert_array_equal(y[:2], x[:2])
    assert not y[2:].any() and observed.all()


def test_an_old_n_row_raises_stale_store(tmp_path):
    spec = frames.SPECS["ntm_frames"]
    codes = np.zeros((400, 40), dtype=np.uint8)
    assert issubclass(StaleStore, ValueError)
    _ntm_store(tmp_path / "old.h5", codes, title=OLD_N_TITLE, modes=None)
    with pytest.raises(StaleStore, match="toroidal mode number n"):
        features(tmp_path / "old.h5", spec, (0, 200))
    # The new title without the modes meta is as stale (D55).
    _ntm_store(tmp_path / "bare.h5", codes, modes=None)
    with pytest.raises(StaleStore):
        features(tmp_path / "bare.h5", spec, (0, 200))


def test_a_row_of_the_wrong_kind_raises(tree):
    """A role reads only a row of its own kind, never a trace as an image, an
    image as a trace, or a modes row's codes as an image's scale."""
    wrong = (
        ("elm_frames", frames.Role("power", "D-alpha FS", "image"), "trace"),
        ("ntm_frames", frames.Role("dalpha", "MPI66M322D power", "trace"), "image"),
        ("ntm_frames", frames.Role("power", "toroidal n", "image"), "modes"),
    )
    for method, role, kind in wrong:
        spec = dataclasses.replace(frames.SPECS["ntm_frames"], roles=(role,))
        with pytest.raises(ValueError, match=f"is {kind}, the role {role.kind}"):
            features(_store(tree, method), spec, WINDOW)


def test_a_frame_is_observed_when_half_its_sub_frames_are(tmp_path):
    """Four 2.5 ms sub-frames a frame: two with a value observe it, one does not."""
    spec = dataclasses.replace(frames.SPECS["elm_frames"], sub_ms=2.5)
    grid = rows.Grid(0.0, 0.5, 400)  # 0-200 ms: five columns a sub-frame, level 1
    t = (np.arange(grid.n) + 0.5) * grid.dt_ms
    y = 1.0 + 0.1 * np.sin(2 * np.pi * t / 7.0)
    sub = np.floor(t / spec.sub_ms)
    y[np.isin(sub, (12, 13))] = np.nan  # frame 3 keeps 2 of its 4 sub-frames
    y[np.isin(sub, (20, 21, 22))] = np.nan  # frame 5 keeps 1
    values = np.stack([y, y])[:, None].astype(np.float32)  # (2, 1, n): min, max
    row = rows.TraceRow("p0", "D-alpha FS01", values)
    rows.write(tmp_path / "gaps.h5", grid, [row])
    x, observed = features(tmp_path / "gaps.h5", spec, (0, 200))
    assert x.shape == (2, 80)
    assert observed.tolist() == [frame != 5 for frame in range(20)]
    assert not x[:, [12, 13, 20, 21, 22]].any() and x[:, [14, 15, 23]].all()


def test_the_clipped_suffix_still_matches_by_prefix(tree):
    spec = frames.SPECS["elm_frames"]
    described = [
        {"name": "p1", "title": "D-alpha PCPHD03"},
        {"name": "p2", "title": "D-alpha FS01, the ELM spans' channel" + CLIPPED},
        {"name": "p3", "title": "D-alpha FS03" + CLIPPED},
    ]
    assert find_role(described, spec.roles[0])["name"] == "p2"  # the first
    assert find_role(described[:1], spec.roles[0]) is None
    # On the tree the role reads the clipped FS01 row: its maxima carry the ELMs.
    x, _ = _features(tree, "elm_frames")
    elms = _during(spec) & (_starts(spec) % 20 == 0)  # the tree's ELM every 20 ms
    assert x[1, elms].min() > x[1, ~elms].max() + 0.2
    # A store with no such row: the required role is missing.
    with pytest.raises(ValueError, match="D-alpha FS"):
        features(_store(tree, "sawtooth_frames"), spec, WINDOW)


def test_the_four_ece_roles_give_four_blocks_in_the_spec_s_order(tree):
    """Roles share a name: each ECE role is its own block, in the spec's order."""
    spec = frames.SPECS["sawtooth_frames"]
    assert [role.name for role in spec.roles].count("ece") == 4
    x, _ = _features(tree, "sawtooth_frames")
    per = round(SAWTOOTH_MS / spec.sub_ms)
    crashes = []
    for i in range(4):
        block = x[8 * i : 8 * i + 8]  # its 4 channels' minima, then their maxima
        spread = (block[4:] - block[:4]).mean(axis=0)[_during(spec)]
        crashes.append(np.unique(spread.reshape(-1, per).argmax(axis=1)).tolist())
    # Each row's crash is in its own sub-frame of every 20 ms period.
    assert crashes == [[int(phase // spec.sub_ms)] for phase in ECE_PHASE_MS]


def test_the_pooled_filterscopes_are_their_channels_mean(tree):
    spec = frames.SPECS["hmode_frames"]
    x, _ = _features(tree, "hmode_frames")
    each = dataclasses.replace(spec.roles[0], pooled=False, channels=len(HMODE_FS))
    unpooled = dataclasses.replace(spec, roles=(each,))
    y, _ = features(_store(tree, "hmode_frames"), unpooled, WINDOW)
    c = len(HMODE_FS)
    assert y.shape == (2 * c, x.shape[1])
    np.testing.assert_allclose(x[0], y[:c].mean(axis=0), atol=1e-6)
    np.testing.assert_allclose(x[1], y[c:].mean(axis=0), atol=1e-6)
    # Each channel's minimum falls through the middle of its range in the
    # sub-frame of its own L-H drop (M1a), so no channel stands in for another.
    starts = _starts(spec)
    for channel, drop in enumerate(HMODE_DROP_MS):
        first = starts[np.argmax(y[channel] < 0.5)]
        assert first == spec.sub_ms * (drop // spec.sub_ms), (HMODE_FS[channel], first)
    # An unpooled role takes the row's channel count as it is.
    four = dataclasses.replace(spec, roles=(dataclasses.replace(each, channels=4),))
    with pytest.raises(ValueError, match="7 channels"):
        features(_store(tree, "hmode_frames"), four, WINDOW)


def test_a_pool_of_eight_filterscopes_is_the_mean_of_those_with_a_range(tmp_path):
    """Eight channels, as 405 of the 450 real rows have: a dark (flat) one and a
    dead (nan) one are left out of the mean, the six others averaged."""
    spec = frames.SPECS["hmode_frames"]
    dt = DT_MS[HMODE]
    grid = rows.Grid(STORE_T0_MS, dt, round(frames_tree.STORE_MS / dt))
    t = grid.t0_ms + (np.arange(grid.n) + 0.5) * dt
    drops = EVENT_MS[0] + 5.0 + 10.0 * np.arange(8)
    in_h = (t >= drops[:, None]) & (t < EVENT_MS[1])
    gain = 1.0 + 0.1 * np.arange(8)[:, None]
    noise = 0.05 * np.random.default_rng(2).standard_normal((8, grid.n))
    dalpha = gain * np.where(in_h, 0.5, 2.0) + noise
    dalpha[0] = 0.0  # dark, as FS01 on 186561
    dalpha[5] = np.nan  # dead
    legend = [f"FS{c:02d}" for c in range(1, 9)]
    values = np.stack([dalpha, dalpha]).astype(np.float32)
    row = rows.TraceRow("p0", "D-alpha filterscopes", values, legend=legend)
    rows.write(tmp_path / "eight.h5", grid, [row])
    x, observed = features(tmp_path / "eight.h5", spec, WINDOW)
    each = dataclasses.replace(spec.roles[0], pooled=False, channels=8)
    unpooled = dataclasses.replace(spec, roles=(each,))
    y, _ = features(tmp_path / "eight.h5", unpooled, WINDOW)
    live = [1, 2, 3, 4, 6, 7]
    np.testing.assert_allclose(x[0], y[:8][live].mean(axis=0), atol=1e-6)
    np.testing.assert_allclose(x[1], y[8:][live].mean(axis=0), atol=1e-6)
    assert not y[[0, 5, 8, 13]].any()  # no range: 0 on their own, and left out
    assert x[0].min() < 0.5 < x[0].max()  # the drops come through the mean
    assert not x[2:].any() and observed.all()  # no NBI row: zeros, presence 0


def test_rows_built_in_memory_give_the_store_s_features(tmp_path):
    """D48: a population shot's rows as `panel_rows.build` returns them."""
    spec = frames.SPECS["hmode_frames"]
    grid, built = _hmode_rows()
    rows.write(tmp_path / "store.h5", grid, built)
    for window in (WINDOW, (1500, 2300)):
        stored = features(tmp_path / "store.h5", spec, window)
        in_memory = features((grid, built, {}), spec, window)
        for a, b in zip(stored, in_memory, strict=True):
            np.testing.assert_array_equal(a, b)
