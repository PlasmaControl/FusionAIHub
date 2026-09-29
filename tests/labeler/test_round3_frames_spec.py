"""Round three, Task 2.5: `labeler.frames`' specs, paths and per-bin targets."""

from __future__ import annotations

import dataclasses
import json

import h5py
import numpy as np
import pandas as pd
import pytest

from labeler import frames
from labeler.events import rosters
from labeler.events.interval_tables import write_label_grid
from labeler.events.panels._shared import CLIPPED, robust_limits
from labeler.events.review import labels
from labeler.frames import targets
from labeler.frames.targets import ABSENT, PRESENT_T, UNCERTAIN_T, UNKNOWN

from . import editor_tree, frames_tree
from .frames_tree import (
    DT_MS,
    ECE_PHASE_MS,
    ECE_RISE_KEV,
    ELM,
    EVENT_MS,
    HMODE,
    HMODE_DROP_MS,
    HMODE_FS,
    IP_SHOT,
    IP_SHOT_HULL,
    IP_WINDOW,
    LEGACY,
    LEGACY_HULL,
    LMODE,
    NTM,
    POPULATION_ONLY,
    POPULATION_WINDOW,
    SAWTOOTH,
    SHOTS,
    WINDOW,
    grid_path,
)

CATEGORIES = {"0": "absent", "1": "present"}
BARS = {
    "elm_frames": {
        "E1": (("f1", ">=", 0.80), ("precision", ">=", 0.70), ("recall", ">=", 0.70)),
        "E2": (("lo(f1 - elm_onsets)", ">=", -0.03),),
        "E3": (("lo(f1 - always)", ">", 0.0),),
    },
    "hmode_frames": {
        "H1": (("f1(H)", ">=", 0.95), ("f1(L)", ">=", 0.70)),
        "H2": (("lo(f1(H) - always)", ">", 0.0),),
    },
    "ntm_frames": {
        "N1": (("f1", ">=", 0.70), ("precision", ">=", 0.60), ("recall", ">=", 0.60)),
        "N2": (("lo(f1 - always)", ">", 0.0),),
    },
    "sawtooth_frames": {
        "S1": (("f1", ">=", 0.85),),
        "S2": (("lo(f1 - always)", ">", 0.0),),
    },
}
#: `Role`'s fields in order, the five older ones first, so a positional `Role`
#: means what it did.
ROLE_FIELDS = ("name", "title", "kind", "optional", "groups", "channels", "pooled")
#: Each spec but its bar: event, roles (`ROLE_FIELDS`), target, sub_ms, bin_ms,
#: pool, required groups, baselines.
SPECS = {
    "elm_frames": (
        ELM,
        (("dalpha", "D-alpha FS", "trace", False, 1, 1, False),),
        "hiro_onsets",
        2.0,
        50.0,
        "max",
        ("filterscopes",),
        ("elm_onsets", "elm_clock", "always"),
    ),
    "hmode_frames": (
        HMODE,
        (
            ("dalpha", "D-alpha filterscopes", "trace", False, 1, 1, True),
            ("nbi", "NBI power", "trace", True, 1, 1, False),
        ),
        "jalal_butt_hl",
        10.0,
        50.0,
        "mean",
        ("filterscopes",),
        ("dalpha_lh", "always"),
    ),
    "ntm_frames": (
        NTM,
        (
            ("power", "MPI66M322D power", "image", False, 32, 1, False),
            ("modes", "toroidal n, MPI66M probes", "modes", False, 10, 1, False),
        ),
        "tearing_archive",
        5.0,
        50.0,
        "mean",
        ("mirnov",),
        ("always",),
    ),
    "sawtooth_frames": (
        SAWTOOTH,
        (
            ("ece", "ECE Te, ch 20-23", "trace", False, 1, 4, False),
            ("ece", "ECE Te, ch 24-27", "trace", False, 1, 4, False),
            ("ece", "ECE Te, ch 28-31", "trace", False, 1, 4, False),
            ("ece", "ECE Te, ch 32-35", "trace", False, 1, 4, False),
            ("sxr", "SXR", "trace", True, 1, 4, False),
        ),
        "ece_sawtooth_v2",
        2.0,
        10.0,
        "mean",
        ("ece",),
        ("always",),
    ),
}


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    return frames_tree.build(tmp_path_factory.mktemp("frames"))


def _roster(shots) -> pd.DataFrame:
    return pd.DataFrame(
        [[shot, "unverified", "false", "", "", ""] for shot in shots],
        columns=list(rosters.ROSTER_COLUMNS),
    )


def _by_start(starts, states) -> dict[float, int]:
    return dict(zip(np.asarray(starts).tolist(), np.asarray(states).tolist()))


def _fine(tree, method) -> tuple[np.ndarray, dict]:
    """A tree store's column centres and its trace rows' level-1 maxima by title."""
    spec = frames.SPECS[method]
    with h5py.File(frames.store_path(tree, spec, SHOTS[spec.store_event]), "r") as f:
        t0, dt, n = (f.attrs[key] for key in ("t0_ms", "dt_ms", "n"))
        metas = {name: json.loads(f["rows"][name].attrs["meta"]) for name in f["rows"]}
        maxima = {
            meta["title"]: f["rows"][name]["1"][1]
            for name, meta in metas.items()
            if meta["kind"] == "trace"
        }
    return t0 + (np.arange(n) + 0.5) * dt, maxima


def _titled(values: dict, prefix: str) -> np.ndarray:
    (found,) = [v for title, v in values.items() if title.startswith(prefix)]
    return found


def test_the_specs_are_the_four_events_with_their_bars():
    assert tuple(f.name for f in dataclasses.fields(frames.Role)) == ROLE_FIELDS
    assert list(frames.SPECS) == list(SPECS)
    for method, spec in frames.SPECS.items():
        roles = tuple(dataclasses.astuple(r) for r in spec.roles)
        assert (
            spec.event,
            roles,
            spec.target,
            spec.sub_ms,
            spec.bin_ms,
            spec.pool,
            spec.required_groups,
            spec.baselines,
        ) == SPECS[method]
        assert (spec.method, spec.store_event) == (method, spec.event)
        assert spec.bar == BARS[method]
    assert {m: sorted(s.bar) for m, s in frames.SPECS.items()} == {
        "elm_frames": ["E1", "E2", "E3"],
        "hmode_frames": ["H1", "H2"],
        "ntm_frames": ["N1", "N2"],
        "sawtooth_frames": ["S1", "S2"],
    }
    assert len(set(frames.SPECS.values())) == 4  # hashable, though a bar is a dict
    assert frames.DERIVED == {"lmode_frames": ("hmode_frames", LMODE)}
    assert frames.SEED == 20260923


def test_the_path_helpers(tree):
    root = tree.root
    assert frames.shots_file(tree, "elm_frames") == root / "frames/shots/elm_frames.csv"
    assert (
        frames.shots_meta_file(tree, "elm_frames")
        == root / "frames/shots/elm_frames.json"
    )
    assert (
        frames.features_dir(tree, "ntm_frames") == root / "frames/features/ntm_frames"
    )
    assert frames.model_dir(tree, "ntm_frames") == root / "models/ntm_frames/v1"
    assert frames.model_dir(tree, "ntm_frames", "v2") == root / "models/ntm_frames/v2"
    assert (
        frames.summary_file(tree, "sawtooth_frames")
        == root / "suggestions/sawtooth_frames/v1/summary.csv"
    )
    assert (
        frames.gallery_dir(tree, frames.SPECS["hmode_frames"], "v2")
        == root / "gallery/high_confinement_mode/hmode_frames-v2"
    )
    elm = frames.SPECS["elm_frames"]
    review = frames.store_path(tree, elm, SHOTS[ELM])
    assert review == root / "spectrograms" / ELM / f"{SHOTS[ELM]}.h5"
    assert review.is_file()
    built = root / "frames" / "stores" / ELM
    assert frames.store_path(tree, elm, LEGACY) == built / f"{LEGACY}.h5"
    # Another event's roster shot is not this one's.
    assert frames.store_path(tree, elm, SHOTS[NTM]) == built / f"{SHOTS[NTM]}.h5"


def test_a_shot_joins_the_review_stores_when_the_roster_takes_it(tmp_path):
    p = editor_tree.paths(tmp_path)
    spec = frames.SPECS["ntm_frames"]
    built = p.root / "frames" / "stores" / NTM / "2.h5"
    assert frames.roster_shots(p, NTM) == frozenset()  # no roster yet
    assert frames.store_path(p, spec, 2) == built
    path = rosters.roster_path(NTM, root=p.label_tables)
    rosters.write_roster(_roster([1]), path)
    assert frames.store_path(p, spec, 2) == built
    rosters.write_roster(_roster([1, 2]), path)
    assert frames.roster_shots(p, NTM) == {1, 2}
    assert frames.store_path(p, spec, 2) == p.spectrogram_file(NTM, 2)


def test_legacy_bins_are_the_max_over_rho(tmp_path, tree):
    label = np.zeros((4, 20))
    label[0] = np.nan  # every rho unknown
    label[1, 5] = 1.0  # one rho present
    label[2, :10] = np.nan  # some rho unknown, the rest absent
    path = tmp_path / "grid.npz"
    write_label_grid(path, [0.0, 50.0, 100.0, 150.0], label, categories=CATEGORIES)
    starts, states = targets.legacy_bins(path, 50.0)
    assert starts.tolist() == [0.0, 50.0, 100.0, 150.0]
    assert states.tolist() == [UNKNOWN, PRESENT_T, ABSENT, ABSENT]
    # Two cells a bin: unknown and present is present.
    starts, states = targets.legacy_bins(path, 100.0)
    assert (starts.tolist(), states.tolist()) == ([0.0, 100.0], [PRESENT_T, ABSENT])
    with pytest.raises(ValueError, match="50 ms"):
        targets.legacy_bins(path, 20.0)
    # A cell the grid does not have is unknown.
    gap = tmp_path / "gap.npz"
    write_label_grid(gap, [0.0, 50.0, 150.0], [1.0, 0.0, 0.0], categories=CATEGORIES)
    starts, states = targets.legacy_bins(gap, 50.0)
    assert starts.tolist() == [0.0, 50.0, 100.0, 150.0]
    assert states.tolist() == [PRESENT_T, ABSENT, UNKNOWN, ABSENT]
    # A class-id grid is not a legacy 0/1 grid.
    classes = tmp_path / "classes.npz"
    write_label_grid(classes, [0.0], [2.0], categories={"0": "a", "1": "b", "2": "c"})
    with pytest.raises(ValueError, match="0 or 1"):
        targets.legacy_bins(classes, 50.0)
    # The tree's ELM grid: known over WINDOW, onsets over 600-1400 ms.
    starts, states = targets.legacy_bins(grid_path(tree, ELM, SHOTS[ELM]), 50.0)
    t = np.arange(0.0, 2000.0, 50.0)
    want = np.where((t >= 600) & (t < 1400), PRESENT_T, ABSENT)
    want[(t < WINDOW[0]) | (t >= WINDOW[1])] = UNKNOWN
    assert starts.tolist() == t.tolist()
    assert states.tolist() == want.tolist()
    assert targets.hull(starts, states, 50.0) == WINDOW
    assert targets.hull(starts, np.full(len(starts), UNKNOWN), 50.0) is None


def test_hl_bins_h_only_l_only_and_both(tmp_path, tree):
    shot = SHOTS[HMODE]
    h, l_ = grid_path(tree, HMODE, shot), grid_path(tree, LMODE, shot)
    starts, states = targets.hl_bins(h, l_, 50.0)
    t = np.arange(50.0, 1950.0, 50.0)
    want = np.where((t >= 600) & (t < 1400), PRESENT_T, ABSENT)  # H only; L only
    want[t == 600] = UNCERTAIN_T  # both
    want[(t < WINDOW[0]) | (t >= WINDOW[1])] = UNKNOWN
    assert starts.tolist() == t.tolist()
    assert states.tolist() == want.tolist()
    # Grids on different cells: a cell only one of them has, and H0 L0.
    h, l_ = tmp_path / "h.npz", tmp_path / "l.npz"
    write_label_grid(h, [0.0, 50.0], [1.0, 0.0], categories=CATEGORIES)
    write_label_grid(l_, [50.0, 100.0], [0.0, 1.0], categories=CATEGORIES)
    starts, states = targets.hl_bins(h, l_, 50.0)
    assert _by_start(starts, states) == {0.0: PRESENT_T, 50.0: UNKNOWN, 100.0: ABSENT}


def test_table_bins_mask_the_uncertain_and_the_unobservable(tree):
    label = labels.read_labels(frames_tree.sawtooth_table(tree))[SHOTS[SAWTOOTH]]
    starts, states = targets.table_bins(label, WINDOW, 10.0)
    assert starts.tolist() == np.arange(100.0, 1900.0, 10.0).tolist()
    by = _by_start(starts, states)
    assert by[590.0] == ABSENT and by[600.0] == by[990.0] == PRESENT_T
    assert by[1000.0] == by[1040.0] == UNCERTAIN_T and by[1050.0] == PRESENT_T
    assert by[1390.0] == PRESENT_T and by[1400.0] == by[1840.0] == ABSENT
    assert by[1850.0] == by[1890.0] == UNKNOWN  # not observable
    # 50 ms bins over a wider window: bins off the label's window are unknown.
    by = _by_start(*targets.table_bins(label, (50, 1950), 50.0))
    assert list(by) == np.arange(50.0, 1950.0, 50.0).tolist()
    assert by[50.0] == by[1900.0] == by[1850.0] == UNKNOWN
    assert by[950.0] == by[1050.0] == PRESENT_T and by[1000.0] == UNCERTAIN_T
    # Only whole bins inside the window.
    starts, _ = targets.table_bins(label, (105, 1897), 50.0)
    assert (starts[0], starts[-1]) == (150.0, 1800.0)
    with pytest.raises(ValueError, match="10 ms"):
        targets.table_bins(label, WINDOW, 15.0)


def test_labelled_is_false_only_when_every_bin_is_unknown():
    assert not targets.labelled(np.full(5, UNKNOWN, dtype=np.int8))
    assert not targets.labelled(np.array([], dtype=np.int8))
    assert targets.labelled(np.array([UNKNOWN, ABSENT]))
    assert targets.labelled([UNKNOWN, UNCERTAIN_T])


def test_window_for_takes_the_catalog_then_ip_then_the_labels(tree, monkeypatch):
    tried = editor_tree.no_fetch(monkeypatch)
    # The ELM shot's corpus Ip says 50-1950 ms; its cohort window comes first.
    assert targets.window_for(SHOTS[ELM], tree, (0, 2000)) == (WINDOW, "catalog")
    assert targets.window_for(POPULATION_ONLY, tree, None) == (
        POPULATION_WINDOW,
        "catalog",
    )
    hull = targets.hull(*targets.legacy_bins(grid_path(tree, ELM, IP_SHOT), 50.0), 50.0)
    assert hull == IP_SHOT_HULL
    assert targets.window_for(IP_SHOT, tree, hull) == (IP_WINDOW, "ip")
    hull = targets.hull(*targets.legacy_bins(grid_path(tree, ELM, LEGACY), 50.0), 50.0)
    assert targets.window_for(LEGACY, tree, hull) == (LEGACY_HULL, "labels")
    with pytest.raises(ValueError, match=str(LEGACY)):
        targets.window_for(LEGACY, tree, None)
    assert tried == []


def test_the_tree_stores_hold_each_spec_s_rows(tree):
    matched = {}
    for method, spec in frames.SPECS.items():
        shot = SHOTS[spec.store_event]
        with h5py.File(frames.store_path(tree, spec, shot), "r") as f:
            names = json.loads(f.attrs["rows"])
            metas = [json.loads(f["rows"][name].attrs["meta"]) for name in names]
            levels = {tuple(sorted(f["rows"][name], key=int)) for name in names}
        assert levels == {("1", "8", "64")}
        for role in spec.roles:
            found = [m for m in metas if m["title"].startswith(role.title)]
            if role.name == "sxr":  # the optional role the tree leaves out
                assert found == []
                continue
            assert len(found) == 1, (method, role.title)
            assert role.kind == ("modes" if "modes" in found[0] else found[0]["kind"])
            if role.kind == "trace" and not role.pooled:
                assert found[0]["n_channels"] == role.channels, (method, role.title)
            matched[method, role.title] = found[0]
        saved = labels.read_saved(tree.label_tables / spec.event)
        assert saved[shot].window == WINDOW
    # The ELM role takes the filterscope, not PCPHD03, clipped suffix and all.
    assert matched["elm_frames", "D-alpha FS"]["title"] == (
        "D-alpha FS01, the ELM spans' channel" + CLIPPED
    )
    # The pooled H-mode filterscopes: 7 lit channels, as on 186561.
    fs = matched["hmode_frames", "D-alpha filterscopes"]
    assert (fs["n_channels"], fs["legend"]) == (7, [f"FS{c:02d}" for c in range(2, 9)])
    # The rows a method could read by mistake differ in their shape, not only in
    # offset and scale, which a robust per-row scaling (D64) takes away (M1a).
    _, elm = _fine(tree, "elm_frames")
    pcphd03, fs01 = _titled(elm, "D-alpha PCPHD03"), _titled(elm, "D-alpha FS")
    assert abs(np.corrcoef(pcphd03[0], fs01[0])[0, 1]) < 0.2
    # Each ECE row's channels rise with its own crash phase and no other row's.
    t, ece = _fine(tree, "sawtooth_frames")
    during = (t >= EVENT_MS[0]) & (t < EVENT_MS[1])
    shapes = [
        size * frames_tree.sawtooth_rise(t, phase)[during]
        for size, phase in zip(ECE_RISE_KEV, ECE_PHASE_MS, strict=True)
    ]
    roles = [r for r in frames.SPECS["sawtooth_frames"].roles if r.name == "ece"]
    for i, role in enumerate(roles):
        for channel in _titled(ece, role.title):
            r = [np.corrcoef(channel[during], shape)[0, 1] for shape in shapes]
            assert np.argmax(r) == i and r[i] > 0.9, (role.title, np.round(r, 2))
    # Each H-mode filterscope drops at its own L-H time: robust-scaled, it falls
    # through the middle of its range within a column of its drop.
    assert len(set(HMODE_DROP_MS)) == len(HMODE_FS)
    t, hmode = _fine(tree, "hmode_frames")
    dalpha = _titled(hmode, "D-alpha filterscopes")
    lo, hi = (v[:, None] for v in robust_limits(t, dalpha, None))
    scaled = (dalpha - lo) / (hi - lo)
    for channel, drop in enumerate(HMODE_DROP_MS):
        first = t[np.argmax(scaled[channel] < 0.5)]
        assert abs(first - drop) < DT_MS[HMODE], (HMODE_FS[channel], first)
