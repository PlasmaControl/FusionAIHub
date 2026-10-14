"""The MHD mode method (`labeler.mhd`) and its eight suggestion tables (`mhd_modes`).

The pieces run on synthetic inputs: a Mirnov n = 2 mode (also on a record that is NaN
over part of the window), rules 1-4 and the harmonic names, the resonance statuses, the
equilibrium tree a shot is labelled with (EFIT02, else the EFIT02ER/EFIT01 fallback), and
the rows the eight events take from a shot's labelled tracks. The frozen label model and
resonance gate run on five of dev's case tracks stored in `data/mhd_frozen_golden.npz`
(made by `data/make_mhd_frozen_golden.py`). Nothing is read from the corpus or fetched.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from labeler.events import candidates, suggestions
from labeler.mhd import (
    equilibrium,
    freeze,
    pipeline,
    resonance,
    rows,
    rules,
    tracks,
    views,
)
from labeler.mhd import labelmodel as LM
from labeler.mhd.labelmodel import NAMES

from . import editor_tree as tree

SHOT = 201250
WINDOW = (0.0, 1000.0)

# --- the Mirnov tracks and rules 1-4 ---------------------------------------------


def test_a_rotating_n2_mode_is_found_as_n2_in_every_frame():
    rng = np.random.default_rng(0)
    t = np.arange(0, 40, tracks.DT)  # ms
    tor_idx = [tracks.ROWS.index(r) for r in tracks.TOR]
    phi = np.deg2rad([tracks.TOR[r] for r in tracks.TOR])
    M = 0.05 * rng.standard_normal((len(tracks.ROWS), len(t)))
    for i, p in zip(tor_idx, phi):
        M[i] += np.cos(2 * np.pi * 10.0 * t - 2 * p)
    grid = np.arange(10, 30, 4.0)
    pk = [p for p in tracks.peaks(M, 0.0, grid, tor_idx, phi) if abs(p[1] - 10) < 0.5]
    assert len(pk) == len(grid)
    assert all(p[4] == 2 and p[5] > tracks.N_SCORE for p in pk)
    ids = tracks.link(pk)
    assert len(set(ids)) == 1
    assert tracks.track_n(np.array([p[4] for p in pk])) == 2


def _n2_record(t_ms):
    """Every corpus `mirnov` row read at 500 kHz, a 10 kHz n = 2 mode on the toroidal ones."""
    rng = np.random.default_rng(1)
    y = 0.05 * rng.standard_normal((max(tracks.ROWS) + 1, len(t_ms)))
    for r, deg in tracks.TOR.items():
        y[r] += np.cos(2 * np.pi * 10.0 * t_ms - 2 * np.deg2rad(deg))
    return y


@pytest.mark.parametrize("nan_ms", [(60.0, 200.0), (50.0, 60.0)], ids=["tail", "gap"])
def test_a_record_that_is_nan_over_part_of_the_window_is_used_where_it_is_finite(
    tmp_path, nan_ms
):
    # dev judged a probe on the whole read (finite over 99% of it): every probe of these
    # records was dead and the shot was not observable
    p = tree.paths(tmp_path)
    t = tree.times(0.0, 120.0, 500e3)
    y = _n2_record(t)
    y[:, (t >= nan_ms[0]) & (t < nan_ms[1])] = np.nan
    tree.write(tracks.corpus_path(p, SHOT), {"mirnov": (t, y)})
    df, reason, info = tracks.frames(SHOT, p, (0.0, 120.0), None)
    assert reason == "ok" and info["probes"] == len(tracks.TOR)
    lo, hi = info["mirnov_ms"]
    reach = tracks.FIR_HALF / 500.0  # ms
    if nan_ms[1] > t[-1]:
        assert lo == 0.0 and nan_ms[0] - 2 * reach < hi < nan_ms[0]
        assert info["mirnov_gaps"] == []
    else:
        assert hi > 119.9 and len(info["mirnov_gaps"]) == 1
        a, b = info["mirnov_gaps"][0]
        assert (
            nan_ms[0] - 2 * reach < a < nan_ms[0]
            and nan_ms[1] < b < nan_ms[1] + 2 * reach
        )
    # no frame's window touches a NaN sample, and every other frame is found as n = 2
    near = (df.t_ms + tracks.WIN / 2 > nan_ms[0] - reach) & (
        df.t_ms - tracks.WIN / 2 < nan_ms[1] + reach
    )
    assert not near.any()
    grid = np.arange(10.0, 110.0, tracks.STEP)
    clear = (grid + tracks.WIN / 2 < nan_ms[0] - reach) | (
        grid - tracks.WIN / 2 > nan_ms[1] + reach
    )
    found = df[(df.f0 - 10.0).abs() < 0.5]
    assert sorted(found.t_ms) == list(grid[clear]) and set(found.n_track) == {2}


def _track(**kw):
    base = {"n": 1, "frames": 30, "fiqr": 0.5, "f": 10.0, "harm": False}
    base |= {"qmin": 1.5, "q95": 4.0, "par": None}
    return SimpleNamespace(**(base | kw))


def test_rules_1_to_4_keep_the_surfaces_inside_q_min_and_q95():
    # m/1 within q_min - 0.2 .. q95 + 0.2: 2/1 and 3/1 (4/1 is out of scope)
    assert rules.shortlist(_track()) == ("2/1", "3/1")
    assert rules.shortlist(_track(par="odd")) == ("3/1",)
    assert rules.shortlist(_track(qmin=np.nan)) == ("none",)
    assert rules.shortlist(_track(harm=True)) == ("harmonic",)


def test_a_fixed_frequency_line_is_gated_out_before_the_surfaces():
    line = _track(frames=rules.LINE_FRAMES, fiqr=0.002 * 10.0)
    assert rules.is_line(line) and rules.shortlist(line) == ("line",)
    assert not rules.is_line(_track(frames=rules.LINE_FRAMES - 1, fiqr=0.0))


def test_a_harmonic_takes_k_times_its_fundamentals_m_and_n():
    assert rules.harmonic_name(2, "2/1") == "4/2"
    assert rules.harmonic_name(3, "2/1") == "6/3"
    assert rules.harmonic_name(3, "3/2") is None
    assert rules.harmonic_name(2, "") is None


# --- the resonance status -------------------------------------------------------------


def test_the_resonance_statuses():
    j21, j31 = NAMES.index("2/1"), NAMES.index("3/1")
    k = len(NAMES)
    P = np.zeros((5, k))
    P[:4, j21] = 0.95
    P[:4, j31] = 0.05
    P[4, [j21, j31]] = 0.5
    R = np.full((5, k), 5.0)
    R[[0, 1, 4], j21] = 0.5
    R[4, j31] = 0.5
    R[3] = np.nan
    A = np.full((5, k), -0.5)
    A[1, j21] = 0.3
    law = {"s_f": 1.0}
    nan = np.full(5, np.nan)
    label, status, reason, _, _, p_res = resonance.decide(P, R, A, nan, nan, law)
    assert status.tolist() == [
        "m/n",
        "q_min above",
        "not resonant",
        "no rotation",
        "uncertain",
    ]
    assert label.tolist() == ["2/1", "", "", "", ""]
    assert reason[2] == "no surface"
    assert np.allclose(p_res[[0, 2, 4]], [0.95, 0.0, 1.0])


# --- the eight events' rows ------------------------------------------------------------


def test_spans_pad_merge_and_drop_short_runs():
    # 98-114 and 138-154 (+-2 ms): 24 ms apart, so not merged, and each under 20 ms
    t = [100, 104, 108, 112, 140, 144, 148, 152]
    assert rows.spans(t, np.full(len(t), 0.5)) == []
    assert rows.spans(np.arange(100, 200, 4.0), np.full(25, 0.7)) == [(98, 198, 0.7)]
    # 98-118 and 134-158: 16 ms apart, merged; the span keeps the largest value
    t = np.r_[np.arange(100, 120, 4.0), np.arange(136, 160, 4.0)]
    value = np.where(t == 140, 0.9, 0.5)
    assert rows.spans(t, value) == [(98, 158, 0.9)]


def _shot(*, status="ok", eq_tree="efit02", extra=(), gaps=()):
    """Tracks over 0-1000 ms: 2/1 (100-300 ms), a 2/1-or-3/1 track (400-480), a line
    (500-700), a non-resonant n = 1 (600-640) and an n = 2 harmonic (800-900)."""
    spec = [
        (1, 1, (100, 300), {"status": "m/n", "label": "2/1", "p_res": 0.97}),
        (2, 1, (400, 480), {"status": "uncertain", "set90": "2/1|3/1"}),
        (3, 1, (500, 700), {"line": True, "why": "line", "modelled": False}),
        (4, 1, (600, 640), {"status": "not resonant", "p_res": 0.0}),
        (5, 2, (800, 900), {"harm": True, "why": "harmonic", "modelled": False}),
        *extra,
    ]
    T, fr = [], []
    for fid, n, (a, b), kw in spec:
        row = {c: np.nan for c in pipeline.TRACK_COLUMNS}
        row |= {"shot": SHOT, "family_id": fid, "n": n, "t0": a, "t1": b, "coh": 0.9}
        row |= {"line": False, "harm": False, "modelled": True, "why": ""}
        row |= {"status": "", "label": "", "set90": "", "p_2/1": 0.6, "p_3/1": 0.3}
        T.append(row | kw)
        fr += [(SHOT, fid, t, 10.0, n) for t in np.arange(a, b + 1, 4.0)]
    info = {
        "window": list(WINDOW),
        "window_source": "test",
        "status": status,
        "mirnov_ms": list(WINDOW),
        "mirnov_gaps": [list(g) for g in gaps],
        "inputs": {"tree": eq_tree, "rotation": True},
        "eq_t": list(np.arange(0, 1000, 20.0)) if eq_tree else [],
        "tracks": len(T),
        "modelled": 3,
    }
    frames = pd.DataFrame(fr, columns=["shot", "family_id", "t_ms", "f0", "n"])
    return pipeline.Shot(SHOT, info, pd.DataFrame(T)[pipeline.TRACK_COLUMNS], frames)


def _state(cells, t_ms):
    return int(cells.state[int(t_ms) - cells.lo])


def test_parents_take_every_n_track_but_lines_and_children_stay_inside():
    c = rows.shot_cells(_shot())
    n1, n2 = c["mhd_n1"], c["mhd_n2"]
    assert [_state(n1, t) for t in (99, 200, 450, 550, 620, 750)] == [1, 1, 1, 0, 1, 0]
    assert _state(n2, 850) == rows.PRESENT and _state(c["mhd_n3"], 850) == rows.ABSENT
    m2, m3 = c["mhd_n1.m2"], c["mhd_n1.m3"]
    assert [_state(m2, t) for t in (200, 450, 620, 750)] == [1, 2, 0, 0]
    assert m2.conf[200] == 0.97 and m2.conf[450] == 0.6
    assert [_state(m3, t) for t in (200, 450)] == [0, 2] and m3.conf[450] == 0.3
    for event, name in rows.CHILDREN.items():
        parent = c[event.split(".")[0]]
        child = c[event]
        assert not np.any(
            np.isin(child.state, (1, 2)) & (parent.state != rows.PRESENT)
        ), name


def test_rows_tile_the_window_with_blank_confidence_off_present_and_uncertain():
    crash = [(900, 950)]  # crash-only present time: n = 1 and 1/1, no confidence
    for event, cells in rows.shot_cells(_shot(), crash=crash).items():
        got = cells.rows(SHOT)
        assert got[0][2] == WINDOW[0] and got[-1][3] == WINDOW[1], event
        assert all(a[3] == b[2] for a, b in itertools.pairwise(got)), event
        assert all(r[4] == "" for r in got if r[1] in (0, 3)), event
        blank = [r[1:4] for r in got if r[1] in (1, 2) and r[4] == ""]
        want = [[1, 900, 950]] if event in ("mhd_n1", "mhd_n1.m1") else []
        assert blank == want, event


def test_no_mirnov_is_not_observable_for_all_eight_events():
    c = rows.shot_cells(_shot(status="no mirnov"))
    assert set(c) == set(rows.EVENTS)
    assert all((cells.state == rows.UNSEEN).all() for cells in c.values())


def test_a_mirnov_gap_is_not_observable_for_the_parent_and_its_children():
    c = rows.shot_cells(_shot(gaps=[(150.5, 260.0)]))
    assert [_state(c["mhd_n1"], t) for t in (120, 200, 280)] == [1, 3, 1]
    assert [_state(c["mhd_n1.m2"], t) for t in (120, 200, 280)] == [1, 3, 1]
    assert _state(c["mhd_n2"], 200) == rows.UNSEEN


def test_without_any_tree_the_children_are_not_observable_inside_the_parent():
    c = rows.shot_cells(_shot(eq_tree=None))
    m2 = c["mhd_n1.m2"]
    assert [_state(m2, t) for t in (200, 450, 750)] == [3, 3, 0]
    assert _state(c["mhd_n1"], 200) == rows.PRESENT
    # a fallback tree labels the children as EFIT02 does
    for eq_tree in ("efit02er", "efit01"):
        m2 = rows.shot_cells(_shot(eq_tree=eq_tree))["mhd_n1.m2"]
        assert [_state(m2, t) for t in (200, 450, 750)] == [1, 2, 0], eq_tree


def test_on_efit01_a_q_min_above_track_is_uncertain_for_its_class_never_absent():
    # EFIT01's q_min reads high: a 3/1 track called "q_min above" may be a resonant 3/1.
    # Its 90% set is left empty here, so only the EFIT01 rule can make it uncertain.
    above = (6, 1, (900, 980), {"status": "q_min above", "cand": "3/1", "set90": ""})
    for eq_tree, want in (("efit01", rows.UNCERTAIN), ("efit02", rows.ABSENT)):
        c = rows.shot_cells(_shot(eq_tree=eq_tree, extra=[above]))
        assert _state(c["mhd_n1"], 940) == rows.PRESENT
        assert _state(c["mhd_n1.m3"], 940) == want, eq_tree
        assert _state(c["mhd_n1.m2"], 940) == rows.ABSENT, eq_tree


def test_mhd_n1_unites_its_tracks_with_crash_and_m1_takes_the_crash_unclipped():
    one = (6, 1, (300, 340), {"status": "m/n", "label": "1/1", "p_res": 0.95})
    crash = [(420, 460), (900, 950)]
    c = rows.shot_cells(_shot(extra=[one]), crash=crash, no_q1=[(0, 330)])
    n1, m1 = c["mhd_n1"], c["mhd_n1.m1"]
    # 900-950 has no n = 1 track: the crash alone makes it n = 1 present
    assert [_state(n1, t) for t in (430, 920, 975)] == [1, 1, 0]
    assert np.isnan(n1.conf[920]) and rows.clipped_ms(n1, crash) == 0.0
    assert [_state(m1, t) for t in (310, 335, 430, 920)] == [0, 1, 1, 1]
    assert _state(c["mhd_n2"], 920) == rows.ABSENT
    # a crash is seen on ECE, so it is n = 1 present without the Mirnov too
    blind = rows.shot_cells(_shot(status="no mirnov"), crash=crash)
    assert _state(blind["mhd_n1"], 920) == rows.PRESENT
    assert _state(blind["mhd_n1.m1"], 920) == rows.PRESENT
    assert _state(blind["mhd_n1.m2"], 920) == rows.UNSEEN  # no Mirnov to tell a 2/1
    assert _state(blind["mhd_n2"], 920) == rows.UNSEEN
    assert _state(c["mhd_n1.m2"], 920) == rows.ABSENT  # the Mirnov saw no 2/1 there
    # the crash is 1/1 evidence only: 2/1 and 3/1 are the same without it, with no tree too
    for eq_tree in ("efit02", None):
        got = _shot(extra=[one], eq_tree=eq_tree)
        a, b = rows.shot_cells(got, crash=crash), rows.shot_cells(got)
        for event in ("mhd_n1.m2", "mhd_n1.m3"):
            assert (a[event].state == b[event].state).all()
        assert _state(a["mhd_n1.m2"], 920) == rows.ABSENT


def test_a_crash_over_a_resonant_1_1_track_keeps_the_tracks_confidence():
    one = (6, 1, (300, 340), {"status": "m/n", "label": "1/1", "p_res": 0.95})
    m1 = rows.shot_cells(_shot(extra=[one]), crash=[(320, 380)])["mhd_n1.m1"]
    assert [_state(m1, t) for t in (310, 330, 360)] == [1, 1, 1]
    assert m1.conf[310] == m1.conf[330] == 0.95 and np.isnan(m1.conf[360])


def test_the_n1_union_takes_the_span_rule_merges_and_drops_short_crash_runs():
    # n = 1 tracks: 98-302, 398-482, 598-642 (the line 500-700 does not count)
    crash = [(310, 330), (650, 660), (700, 712), (720, 735), (900, 910)]
    c = rows.shot_cells(_shot(), crash=crash, no_q1=[])
    n1, m1 = c["mhd_n1"], c["mhd_n1.m1"]
    # 302-310, 642-650 and 712-720 are gaps of 8 ms: present, with no confidence
    assert [_state(n1, t) for t in (305, 340, 645, 680, 715, 905)] == [1, 0, 1, 0, 1, 0]
    assert np.isnan(n1.conf[305]) and n1.conf[200] == 0.9
    # 900-910 alone is under 20 ms: dropped from n = 1, so from m1 too
    assert rows.clipped_ms(n1, crash) == rows.clipped_ms(m1, crash) == 10.0
    assert [_state(m1, t) for t in (305, 320, 705, 715, 905)] == [0, 1, 1, 0, 0]
    for event in rows.CHILDREN:
        parent = n1 if event == "mhd_n1.m1" else c[event.split(".")[0]]
        assert not np.any(np.isin(c[event].state, (1, 2)) & (parent.state != 1)), event
    # the n = 2 and n = 3 parents, and 2/1 and 3/1, are the same without the crash
    plain = rows.shot_cells(_shot(), no_q1=[])
    for event in ("mhd_n2", "mhd_n3", "mhd_n1.m2", "mhd_n1.m3"):
        assert (c[event].state == plain[event].state).all(), event


def test_the_union_rule_keeps_track_time_and_not_observable_gaps():
    c = rows.Cells((0, 100))
    c.state[10:15] = rows.PRESENT  # a 5 ms piece of a track span a Mirnov gap cut
    c.state[40:50] = c.state[55:65] = rows.PRESENT  # 5 ms apart: one span
    c.state[52:54] = rows.UNSEEN  # a Mirnov gap inside the merge stays so
    c.state[90:95] = rows.PRESENT  # a 5 ms crash run alone: back to the base
    keep = np.zeros(100, bool)
    keep[10:15] = True
    base = np.full(100, rows.ABSENT)
    base[92:] = rows.UNSEEN
    rows.merged_union(c, keep=keep, base=base)
    assert (c.state[10:15] == rows.PRESENT).all()
    assert list(c.state[38:67]) == [0, 0] + [1] * 12 + [3, 3] + [1] * 11 + [0, 0]
    assert list(c.state[88:97]) == [0, 0, 0, 0, 3, 3, 3, 0, 0]


def test_no_q1_time_is_absent_for_m1_only_where_m1_is_observable():
    one = (6, 1, (300, 340), {"status": "m/n", "label": "1/1", "p_res": 0.95})
    c = rows.shot_cells(_shot(extra=[one], gaps=[(150.5, 260.0)]), no_q1=[(0, 1000)])
    m1 = c["mhd_n1.m1"]
    assert _state(m1, 200) == rows.UNSEEN and _state(m1, 320) == rows.ABSENT
    blind = rows.shot_cells(_shot(status="no mirnov"), no_q1=[(0, 1000)])
    assert (blind["mhd_n1.m1"].state == rows.UNSEEN).all()


# --- the equilibrium a shot is labelled with ------------------------------------------


def _eq_file(path, n_slices, tree=None):
    nr = 9
    d = {
        "times": np.arange(n_slices) * 20.0 + 100.0,
        "qpsi": np.tile(np.linspace(1.1, 4.0, nr), (n_slices, 1)),
        "fpol": np.full((n_slices, nr), 3.4),
        "psirz": np.zeros((n_slices, 5, 5), np.float32),
        "r": np.linspace(1.0, 2.4, 5),
        "z": np.linspace(-1.0, 1.0, 5),
        "ssimag": np.zeros(n_slices),
        "ssibry": np.ones(n_slices),
        "rmaxis": np.full(n_slices, 1.7),
    }
    if tree:
        d["tree"] = np.array(tree)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **d)


def test_the_equilibrium_is_efit02_else_the_fallback_tree_each_with_two_slices(
    tmp_path,
):
    p = tree.paths(tmp_path)
    assert (
        equilibrium.equilibrium(p, SHOT) is None and pipeline._eq_key(p, SHOT) is None
    )
    _eq_file(equilibrium.fallback_path(p, SHOT), 3, "EFIT01")
    got = equilibrium.equilibrium(p, SHOT)
    assert got.tree == "efit01" and got.path == str(equilibrium.fallback_path(p, SHOT))
    # a one-slice EFIT02 is no tree: the fallback stays
    _eq_file(equilibrium.efit02_path(p, SHOT), 1)
    assert equilibrium.equilibrium(p, SHOT).tree == "efit01"
    assert pipeline._eq_key(p, SHOT)[0] == str(equilibrium.fallback_path(p, SHOT))
    _eq_file(equilibrium.efit02_path(p, SHOT), 4)
    got = equilibrium.equilibrium(p, SHOT)
    assert got.tree == "efit02" and len(got.t) == 4
    # the cache key names the file used, so the shot reruns when it changes
    path, _, size, used = pipeline._eq_key(p, SHOT)
    assert path == str(equilibrium.efit02_path(p, SHOT)) and used == "efit02"
    assert size == equilibrium.efit02_path(p, SHOT).stat().st_size
    assert "efit02er" in equilibrium.TRUSTED and "efit01" not in equilibrium.TRUSTED


def test_mhd_modes_writes_all_eight_tables_from_one_pipeline_run(tmp_path, monkeypatch):
    p = tree.paths(tmp_path)
    calls = []
    monkeypatch.setattr(pipeline, "window", lambda shot, paths: (WINDOW, "test"))

    def run_shot(shot, paths, win, source, *, force=False):
        calls.append(shot)
        return _shot()

    monkeypatch.setattr(pipeline, "run_shot", run_shot)
    monkeypatch.setattr(candidates, "_no_q1", lambda shot, paths: None)
    assert candidates.MHD_EVENTS == rows.EVENTS
    assert candidates.EVENTS["mhd_modes"] == rows.EVENTS
    # without the catalog cohort the blind shots cannot be kept out: refused
    with pytest.raises(FileNotFoundError, match="cohort.csv"):
        candidates.run_mhd_all(p, shots=[SHOT])
    assert calls == []
    tree.cohort(p, [tree.queue_row(SHOT + 1, 1, blind=True)])
    metas = candidates.run_mhd_all(p, shots=[SHOT, SHOT + 1])
    assert calls == [SHOT] and set(metas) == set(rows.EVENTS)
    version = candidates.METHOD_VERSION["mhd_modes"]
    for event in metas:
        path = candidates.table_path(p, event, "mhd_modes", version)
        frame = pd.read_csv(path, keep_default_na=False)
        assert frame.t_start.min() == 0 and frame.t_end.max() == 1000, event
        on_disk = json.loads(path.with_suffix(".meta.json").read_text())
        assert on_disk["version"] == version and on_disk["freeze"]["fit_shots"] > 0
        assert sum(on_disk["totals_s"].values()) == pytest.approx(1.0)
    m1 = json.loads(
        candidates.table_path(p, "mhd_n1.m1", "mhd_modes", version)
        .with_suffix(".meta.json")
        .read_text()
    )
    assert m1["crash"].startswith(f"no crash {candidates.MHD_CRASH[1]} table")
    assert metas["mhd_n1.m2"]["status_counts"] == {"m/n": 1, "uncertain": 1}
    assert metas["mhd_n1"]["trees"] == {"efit02": 1}
    assert metas["mhd_n1"]["per_shot"][str(SHOT)]["tree"] == "efit02"
    # a written version is never overwritten without --refresh
    with pytest.raises(FileExistsError, match="--refresh"):
        candidates.run_mhd_all(p, shots=[SHOT])
    assert calls == [SHOT]
    # with the crash table: its time outside mhd_n1 and m1 (a run under 20 ms) is named,
    # and a shot whose q gate fails is named in m1's meta
    crash = candidates.table_path(p, "mhd_n1.m1", *candidates.MHD_CRASH)
    crash.parent.mkdir(parents=True, exist_ok=True)
    suggestions.write_table(
        crash,
        [
            [SHOT, 0, 0, 420, ""],
            [SHOT, 1, 420, 460, ""],
            [SHOT, 0, 460, 900, ""],
            [SHOT, 1, 900, 910, ""],
            [SHOT, 0, 910, 1000, ""],
        ],
        {"event": "mhd_n1.m1", "method": "crash"},
    )

    def no_gate(shot, paths):
        raise RuntimeError("an unreadable gate")

    monkeypatch.setattr(candidates, "_no_q1", no_gate)
    metas = candidates.run_mhd_all(p, shots=[SHOT], refresh=True)
    assert calls == [SHOT, SHOT]
    assert metas["mhd_n1"]["crash"]["outside_parent_ms"] == 10.0
    assert metas["mhd_n1.m1"]["crash"]["not_in_m1_ms"] == 10.0
    assert metas["mhd_n1.m1"]["q_gate"]["errors"] == {
        str(SHOT): "RuntimeError: an unreadable gate"
    }


def test_cached_rows_skip_a_stale_shot_and_keep_a_shots_skip_reason(
    tmp_path, monkeypatch
):
    p = tree.paths(tmp_path)
    got = _shot()
    got.info["key"] = {"window": list(WINDOW), "equilibrium": None}
    monkeypatch.setattr(pipeline, "window", lambda shot, paths: (WINDOW, "test"))
    monkeypatch.setattr(pipeline, "load", lambda shot, paths: got)
    monkeypatch.setattr(candidates, "_no_q1", lambda shot, paths: [])
    now = {"window": list(WINDOW), "equilibrium": ["f.npz", 1, 2, "efit01"]}
    monkeypatch.setattr(pipeline, "key", lambda shot, paths, win: now)
    _, out, why, *_ = candidates._mhd_job((SHOT, p, False, True))
    assert out is None and why == "stale cache: the key differs in equilibrium"
    now = dict(got.info["key"])
    assert candidates._mhd_job((SHOT, p, False, True))[1] is got
    monkeypatch.setattr(pipeline, "window", lambda shot, paths: (None, "no Ip"))
    assert candidates._mhd_job((SHOT, p, False, True))[2] == "no window: no Ip"


def test_only_numpys_all_nan_warnings_are_silenced():
    import warnings

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        with tracks.quiet_all_nan():
            nan = np.full((2, 3), np.nan)
            for reduce in (np.nanmean, np.nanmedian, np.nanstd, np.nanmax):
                assert np.isnan(reduce(nan, axis=1)).all()
            np.divide(np.ones(1), np.zeros(1))
    assert [str(w.message) for w in seen] == ["divide by zero encountered in divide"]


def test_the_freeze_cli_refits_only_with_write(monkeypatch):
    monkeypatch.setattr(freeze, "fit", lambda paths: pytest.fail("refit"))
    for argv in ([], ["--check-only", "--write"]):
        with pytest.raises(SystemExit) as exit_:
            freeze.main(argv)
        assert exit_.value.code == 2


# --- the frozen label model and resonance gate on dev's case tracks ---------------------

GOLDEN = Path(__file__).parent / "data" / "mhd_frozen_golden.npz"
GOLDEN_SHOT = 199607


class _Npz(dict):
    """What `equilibrium.Equilibrium` reads from an opened npz."""

    @property
    def files(self):
        return list(self)


def _golden():
    """The stored tracks (five picked, one per status, and the anchors they overlap) with
    their views, frames, EFIT02 slices and ZIPFIT times, and the expected numbers."""
    with np.load(GOLDEN) as d:
        g = {k: d[k] for k in d.files}
    tab = pd.DataFrame({c[4:]: v for c, v in g.items() if c.startswith("tab_")}).astype(
        {"par": object, "anchor4": object, "list4": object}
    )
    for c in ("par", "anchor4", "list4"):
        tab[c] = tab[c].where(tab[c] != "", np.nan)
    tab["harm"] = tab.harm == "True"
    F = {k[2:]: v for k, v in g.items() if k.startswith("F_")}
    tab = pd.concat([tab, views.locators(F)], axis=1)
    frames = pd.DataFrame(
        {c[7:]: v for c, v in g.items() if c.startswith("frames_")}
    ).assign(shot=GOLDEN_SHOT)
    eq = equilibrium.Equilibrium(
        _Npz({k[3:]: v for k, v in g.items() if k.startswith("eq_")})
    )
    rot_u = (g["rotu_w"], g["rotu_rho"], g["rotu_t"])
    rot_r = (
        [
            (g[f"rotr{s}_w"], g[f"rotr{s}_rho"], g[f"rotr{s}_t"])
            for s in range(int(g["rotr_n"]))
        ],
        float(g["rotr_sign"]),
    )
    return g, tab, F, frames, {GOLDEN_SHOT: eq}, rot_u, rot_r


def test_the_ported_label_model_gives_devs_own_case_posterior():
    # dev's case fit (EM on the corpus without the case shots), refit by the port
    g, tab, F, _, eqs, rot_u, _ = _golden()
    P = LM.posterior(
        tab, F, eqs, {GOLDEN_SHOT: rot_u}, json.loads(str(g["dev_params"]))
    )
    np.testing.assert_allclose(P[g["at"]], g["dev_P"], rtol=0, atol=1e-9)


def test_the_frozen_model_labels_dev_case_tracks_as_it_did_at_the_freeze():
    g, tab, F, frames, eqs, rot_u, rot_r = _golden()
    frozen = freeze.load()
    P = LM.posterior(tab, F, eqs, {GOLDEN_SHOT: rot_u}, frozen["params"])
    sl = resonance.slices(tab, frames, eqs, {GOLDEN_SHOT: rot_r})
    R, A, fn, top = resonance.distances(sl, tab, frozen["law"])
    label, status, _, cand, p, p_res = resonance.decide(P, R, A, fn, top, frozen["law"])
    mp, pmax, s90 = LM.credible(P, np.array([NAMES] * len(tab)))
    at = g["at"]
    for got, key in ((P, "P"), (R, "R"), (p, "p"), (p_res, "p_res"), (pmax, "pmax")):
        np.testing.assert_allclose(got[at], g[f"want_{key}"], rtol=0, atol=1e-9)
    assert list(np.array(s90)[at]) == list(g["want_set90"])
    # one track per status, labelled as dev labelled it
    assert list(status[at]) == list(resonance.STATUSES) == list(g["dev_status"])
    assert list(label[at]) == list(g["dev_label"]) and list(cand[at]) == list(
        g["dev_cand"]
    )
    assert list(mp[at]) == list(g["dev_map"])
    # dev's numbers are rounded to 4 decimals; its case fit left out the 153 corpus tracks
    # of three case shots that freeze.json was fit on, hence the looser posterior bound
    np.testing.assert_allclose(R[at], g["dev_R"], rtol=0, atol=6e-5)
    np.testing.assert_allclose(pmax[at], g["dev_pmax"], rtol=0, atol=5e-3)
