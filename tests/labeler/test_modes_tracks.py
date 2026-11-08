"""`labeler.modes.tracks` and `cells`: the owner's rules on synthetic port-v6 tracks.

A harmonic is dropped and an independent 4/2 is kept; n and m map to the event and its
subcategory; the 1 ms grid spans are the frames padded 2 ms and merged. Nothing is read from
the real labels.
"""

from __future__ import annotations

import pytest

from labeler.mhd import rows
from labeler.modes import cells as C
from labeler.modes import suggest
from labeler.modes import tracks as T

from .modes_synth import SHOT, make_shot, write_shot


def test_class_names_map_to_n_events_and_m_subcategories():
    assert T.parse_class("2/1") == (2, 1)
    assert T.parse_class("7/3") == (7, 3)
    assert T.event_of("2/1") == "mhd_n1.m2"
    assert T.event_of("4/2") == "mhd_n2.m4"
    assert T.event_of("4/3") == "mhd_n3.m4"
    assert T.event_of("1/1") == "mhd_n1.m1"
    assert C.parent_of("mhd_n2.m4") == "mhd_n2" == C.parent_of("mhd_n2")
    for bad in ("", "21", "2/", "a/b", "2/1/3"):
        with pytest.raises(ValueError):
            T.parse_class(bad)


def test_children_are_ordered_by_n_then_m_and_deduplicated():
    kids = C.children(["4/3", "2/1", "3/2", "2/1", "1/1", "3/1"])
    assert list(kids) == [
        "mhd_n1.m1",
        "mhd_n1.m2",
        "mhd_n1.m3",
        "mhd_n2.m3",
        "mhd_n3.m4",
    ]
    assert kids["mhd_n2.m3"] == "3/2"


def test_a_harmonic_is_dropped_and_an_independent_4_over_2_is_kept():
    ruled, counts = T.apply_rules(make_shot())
    assert counts == {"harmonics": 1, "out_of_scope": 1, "kept": 7}
    assert 2 not in set(ruled.tracks.family_id)  # the harmonic 4/2 at 2f of the 2/1
    assert 3 in set(ruled.tracks.family_id)  # the independent 4/2
    assert 5 not in set(ruled.tracks.family_id)  # n = 4 is outside the scope
    # the dropped tracks' frames go with them
    assert set(ruled.frames.family_id) == set(ruled.tracks.family_id)
    assert T.classes(ruled) == ["1/1", "2/1", "3/2", "4/2", "4/3"]


def test_the_harmonic_alone_leaves_no_4_over_2():
    only = [row for row in make_shot().tracks.itertuples() if row.family_id in (1, 2)]
    spec = [
        (
            r.family_id,
            r.n,
            (r.t0, r.t1),
            {"status": r.status, "label": r.label, "harm": r.harm},
        )
        for r in only
    ]
    ruled, counts = T.apply_rules(make_shot(spec=spec))
    assert counts["harmonics"] == 1
    assert T.classes(ruled) == ["2/1"]


def test_n_is_taken_as_an_absolute_value():
    spec = [(1, -2, (100, 200), {"status": "m/n", "label": "3/2", "p_res": 0.95})]
    ruled, counts = T.apply_rules(make_shot(spec=spec))
    assert counts["kept"] == 1 and list(ruled.tracks.n) == [2]


def test_only_labelled_modes_are_labelled_and_a_line_never_is():
    ruled, _ = T.apply_rules(make_shot())
    only = T.labelled(ruled)
    assert sorted(only.tracks.family_id) == [
        1,
        3,
        4,
        8,
        9,
    ]  # not the abstention 7, not the line 6
    assert set(only.frames.family_id) == {1, 3, 4, 8, 9}


def test_the_label_view_shows_labelled_modes_only_where_the_suggestion_view_shows_n_tracks():
    ruled, _ = T.apply_rules(make_shot())
    kids = C.children(T.classes(ruled) + list(suggest.V5_CLASSES))
    sug = C.shot_cells(ruled, kids)
    strict = C.shot_cells(ruled, kids, strict=True)
    # the abstention (800-860, n = 1) is an n = 1 mode in the suggestion view only
    assert C.present_runs(sug["mhd_n1"]) == [
        (98, 302, 0.9),
        (798, 862, 0.9),
        (898, 962, 0.9),
    ]
    assert [(a, b) for a, b, _ in C.present_runs(strict["mhd_n1"])] == [
        (98, 302),
        (898, 962),
    ]
    # and it is uncertain for 2/1 and 3/1 there, absent in the label view
    for event in ("mhd_n1.m2", "mhd_n1.m3"):
        assert int((sug[event].state == rows.UNCERTAIN).sum()) == 64
        assert int((strict[event].state == rows.UNCERTAIN).sum()) == 0


def test_spans_on_the_1_ms_grid_are_the_frames_padded_two_ms_and_merged():
    ruled, _ = T.apply_rules(make_shot())
    kids = C.children(T.classes(ruled))
    cells = C.shot_cells(ruled, kids, strict=True)
    # frames 100..300 every 4 ms, padded +-2: cells [98, 302)
    assert C.present_runs(cells["mhd_n1.m2"]) == [(98, 302, 0.97)]
    assert C.present_runs(cells["mhd_n2.m4"]) == [(398, 502, 0.93)]
    assert C.present_runs(cells["mhd_n2.m3"]) == [(598, 702, 0.95)]
    assert C.present_runs(cells["mhd_n3.m4"]) == [(698, 782, 0.91)]
    # the harmonic's 100-300 does not make n = 2 present, the independent 4/2 does
    n2 = cells["mhd_n2"]
    assert n2.state[200 - n2.lo] == rows.ABSENT
    assert n2.state[450 - n2.lo] == rows.PRESENT
    # a child is inside its parent's present time
    parent = cells["mhd_n2"]
    for event in ("mhd_n2.m3", "mhd_n2.m4"):
        child = cells[event]
        for a, b, _ in C.present_runs(child):
            assert (parent.state[a - parent.lo : b - parent.lo] == rows.PRESENT).all()


def test_a_shot_whose_mirnov_cannot_be_read_is_not_observable_for_every_event():
    ruled, _ = T.apply_rules(make_shot(status="no mirnov"))
    kids = C.children(["2/1"])
    for event, cells in C.shot_cells(ruled, kids).items():
        assert (cells.state == rows.UNSEEN).all(), event


def test_read_shot_needs_all_three_files(tmp_path):
    got = make_shot()
    assert T.read_shot(tmp_path, SHOT) is None and not T.has_labels(tmp_path, SHOT)
    write_shot(tmp_path, got)
    assert T.has_labels(tmp_path, SHOT)
    back = T.read_shot(tmp_path, SHOT)
    assert back.info["window"] == [0.0, 1000.0]
    assert len(back.tracks) == len(got.tracks) and len(back.frames) == len(got.frames)
    (tmp_path / f"{SHOT}.frames.parquet").unlink()
    assert T.read_shot(tmp_path, SHOT) is None
