"""The teaser figure drawn from the catalog's labels (`paper.label_figure`),
on small tables and an in-memory shot: no store, corpus or model is read."""

from __future__ import annotations

import numpy as np
import pytest

from labeler.paper import label_figure as lf
from labeler.paper import roster
from labeler.paper.roster import Read, Signal

HEADER = "shot,category,t_start,t_end,confidence,attrs\n"


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_a_table_keeps_absent_rows_and_leaves_unmarked_time_out(tmp_path):
    table = _write(
        tmp_path / "t.csv",
        HEADER
        + '1,0,0,100,,\n1,2,100,300,,"{""iscrowd"": 1}"\n2,1,50,80,,\n3,1,9,9,,\n',
    )
    rows = lf.read_rows(table)
    assert rows[1] == (lf.Row(0, 100, 0), lf.Row(100, 300, 2, 1))
    assert rows[2] == (lf.Row(50, 80, 1),)  # no absent row: the rest is unlabelled
    assert 3 not in rows  # a point has no span to draw
    assert lf.read_rows(tmp_path / "missing.csv") == {}


def test_the_merged_confinement_table_reads_as_the_review_regimes(tmp_path):
    head = "shot,t_start,t_end,regimes,sources,status,label,source_regimes\n"
    table = _write(
        tmp_path / "m.csv",
        head + "7,0,10,H,a,x,1,{}\n7,10,20,L,a,x,0,{}\n7,20,30,QH,a,x,1,{}\n"
        "7,30,40,WP,a,x,1,{}\n7,40,50,H|L,a|b,conflict,,{}\n",
    )
    rows = lf.read_rows(table, regimes=True)
    assert [r.category for r in rows[7]] == [1, 2, 3, 4, 5]
    bad = _write(tmp_path / "bad.csv", head + "7,0,10,X,a,x,1,{}\n")
    with pytest.raises(ValueError, match="unknown regimes"):
        lf.read_rows(bad, regimes=True)


def _specs(tmp_path):
    review = _write(tmp_path / "review.csv", HEADER + "1,0,0,10,,\n1,1,10,20,,\n")
    imported = _write(tmp_path / "format.csv", HEADER + "1,1,0,5,,\n2,1,0,40,,\n")
    sources = (
        lf.Source(lf.SILVER, "review", lambda paths: review),
        lf.Source(lf.LEGACY, "table", lambda paths: imported),
    )
    return (
        lf.TrackSpec("a", "A", lf.BINARY, sources),
        lf.TrackSpec("b", "B", lf.BINARY, sources[1:]),
    )


def test_the_expert_review_wins_and_a_shot_no_source_holds_has_no_label(tmp_path):
    specs = _specs(tmp_path)
    read = lf.read_sources(None, specs)
    one = lf.tracks_of(1, read, specs)
    assert [t.source.tier for t in one] == [lf.SILVER, lf.LEGACY]
    two = lf.tracks_of(2, read, specs)
    assert [t.source.tier for t in two] == [lf.LEGACY, lf.LEGACY]
    assert all(t.source is None for t in lf.tracks_of(3, read, specs))
    assert lf.score(one) == (2, 1, 15)  # absent rows are not labelled time


def test_the_pick_takes_the_most_phenomena_then_expert_then_time(tmp_path):
    specs = _specs(tmp_path)
    read = lf.read_sources(None, specs)
    found = [lf.Candidate(s, 2021, (0, 100)) for s in (3, 2, 1)]
    # 1 and 2 both carry two phenomena; 1 has an expert review.
    assert lf.pick(found, read, specs).shot == 1
    assert lf.pick(found[:2], read, specs).shot == 2
    assert lf.union_ms([(0, 10), (5, 20), (30, 40)]) == 30


def test_a_row_is_found_by_every_part_of_its_role():
    title = "ECE Te, inversion side A, ch 20-23 (0.05 ms median)"
    assert lf.matches(title, "ECE Te, ch 20-23")
    assert not lf.matches(title, "ECE Te, ch 24-27")
    assert lf.matches("D-alpha FS02, the ELM spans' channel", "D-alpha FS")
    assert lf._trace_name(title) == "ch 20-23"


def _shot() -> lf.LabelShot:
    image = Read(
        {"n_y": 4, "y0": 15.0, "dy": 30.0}, np.full((4, 20), 99, np.uint8), 0, 100
    )
    traces = tuple(
        Read(
            {"name": f"p{i}", "title": f"ECE Te, side A, ch {g} (median)"},
            np.stack([np.zeros((1, 20)), np.ones((1, 20))]).astype("<f4"),
            0,
            100,
        )
        for i, g in enumerate(("20-23", "24-27"))
    )
    signals = tuple(
        Signal(p, p.title, traces, None)
        if p.role == "ece"
        else Signal(p, p.title, (), f"no {p.title} data")
        for p in roster.PANELS
    )
    expert = lf.Source(lf.SILVER, "review", lambda paths: None)
    legacy = lf.Source(lf.LEGACY, "table", lambda paths: None, regimes=True)
    by_key = {spec.key: spec for spec in lf.TRACKS}
    tracks = (
        lf.Track(by_key["alfven_eigenmode"]),
        lf.Track(
            by_key["edge_localized_mode"],
            expert,
            None,
            (lf.Row(0, 30, 0), lf.Row(30, 80, 2, 1), lf.Row(80, 100, 0)),
        ),
        lf.Track(
            by_key["confinement"], legacy, None, (lf.Row(20, 60, 1), lf.Row(60, 70, 3))
        ),
    )
    return lf.LabelShot(7, 2021, (0, 100), image, signals, tracks, {})


def test_the_figure_draws_labels_with_their_tiers_and_no_suggestion(tmp_path):
    fig = lf.draw(_shot(), tmp_path / lf.STEM)
    assert (tmp_path / f"{lf.STEM}.pdf").stat().st_size > 0
    (key,) = fig.legends
    names = [t.get_text() for t in key.get_texts()]
    assert names == [
        "absent",
        "uncertain",
        "H-mode",
        "QH-mode",
        lf.CROWD,
        lf.UNLABELLED,
    ]
    texts = {t.get_text() for ax in fig.axes for t in ax.texts}
    tiers = {c.yaxis.label.get_text() for ax in fig.axes for c in ax.child_axes}
    assert tiers == {lf.NO_LABEL, lf.SILVER, lf.LEGACY}
    words = " ".join([*names, *texts, *tiers, fig.axes[0].get_title()])
    assert "suggest" not in words and "model:" not in words
    hatched = [p for ax in fig.axes for p in ax.collections if p.get_hatch()]
    assert len(hatched) == 1  # the one crowd span
