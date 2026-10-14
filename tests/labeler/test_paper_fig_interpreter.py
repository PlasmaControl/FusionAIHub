"""Figure 1's label rows: each event's rows from a table (`--rows`, else the review
page's source table) before the catalog's tiers, adjacent rows of one state drawn
as one run, a table of labels recorded as labels (no detector score or threshold)
while a detector's table keeps its detector's caveats, and no expert review on a
gold evaluation shot."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from labeler.config import Paths
from labeler.events.review import labels
from labeler.paper import figure_sources as fs

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts/labeler/paper/fig_interpreter_tokeye.py"
)
spec = importlib.util.spec_from_file_location("fig_interpreter_tokeye", SCRIPT)
fi = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fi
spec.loader.exec_module(fi)

Row = fi.lf.Row
ELM, NTM, AE = "edge_localized_mode", fi.mode_tags.NTM, fi.mode_tags.AE


def _paths(tmp_path) -> Paths:
    return Paths(root=tmp_path, label_tables=tmp_path / "events")


def _table(path: Path, rows: str, header="shot,category,t_start,t_end,confidence"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n" + rows)
    return path


def test_adjacent_rows_of_one_state_are_drawn_as_one_run():
    rows = [
        Row(30, 40, 1),
        Row(0, 10, 2),
        Row(10, 20, 2),
        Row(25, 30, 2),  # a gap before it: its own run
        Row(40, 50, 1, crowd=1),  # an expert crowd span keeps its own ends
    ]
    runs = fi.display_runs(rows)
    assert [(r.t_start, r.t_end, r.category, r.crowd) for r in runs] == [
        (0, 20, 2, None),
        (25, 30, 2, None),
        (30, 40, 1, None),
        (40, 50, 1, 1),
    ]


def test_a_supplied_table_comes_before_the_review_source_table(tmp_path):
    paths = _paths(tmp_path)
    source = _table(tmp_path / "suggestions/elm_jump/v1/elm.csv", "7,1,100,200,\n")
    labels.write_pointer(
        paths.label_tables / ELM, source, method="elm_jump", version="v1"
    )
    track = fi.table_track(paths, 7, ELM)
    assert track.table == {
        "how": fi.POINTER,
        "path": str(source.resolve()),
        "method": "elm_jump",
        "version": "v1",
        "kind": "labels",
    }
    assert [(r.t_start, r.t_end, r.category) for r in track.rows] == [(100, 200, 1)]
    assert track.source.tier == fi.lf.GENERATED
    assert fi.table_track(paths, 8, ELM) is None, "the table has no row for 8"
    assert fi.table_track(paths, 7, NTM) is None, "no pointer"

    mine = _table(tmp_path / "mine.csv", "7,2,0,50,\n")
    (rows,) = fi.parse_rows([f"{ELM}={mine}"]).items()
    track = fi.table_track(paths, 7, *rows)
    assert (track.table["how"], track.rows[0].category) == (fi.SUPPLIED, 2)
    with pytest.raises(SystemExit, match="no row for shot 8"):
        fi.table_track(paths, 8, ELM, mine)
    for bad in (["sawtooth=x"], [f"{ELM}={mine}", f"{ELM}={mine}"]):
        with pytest.raises(SystemExit):
            fi.parse_rows(bad)

    regimes = _table(
        tmp_path / "merged.csv", "7,0,40,L\n7,40,90,H\n", "shot,t_start,t_end,regimes"
    )
    track = fi.table_track(paths, 7, "confinement", regimes)
    assert [r.category for r in track.rows] == [2, 1], "the regime letters as classes"


def test_a_tables_tm_rows_carry_no_detector_score_or_threshold(tmp_path):
    paths = _paths(tmp_path)
    table = _table(tmp_path / "tm.csv", "7,1,100,200,\n7,0,200,300,\n")
    (tmp_path / "tm.meta.json").write_text(
        json.dumps({"method": "tm_mhd", "version": "v1"})
    )
    record = fi.track_record(fi.table_track(paths, 7, NTM, table), (0, 300))
    assert record["performance"] is None
    assert record["decision_threshold"] is None
    assert (record["model"], record["source_table"]["version"]) == ("tm_mhd", "v1")


def test_a_detectors_table_keeps_its_threshold_score_and_caveats(tmp_path):
    paths = _paths(tmp_path)
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(
        json.dumps(
            {
                "scores": {"ntm_frames": {"f1": 0.457, "shots": 61}},
                "bin_ms": 10,
                "pool": "shots",
                "bar_criteria": {"N1": [["f1", ">=", 0.7]]},
            }
        )
    )
    table = _table(tmp_path / "tm.csv", "7,1,100,200,\n7,0,200,300,\n")
    (tmp_path / "tm.meta.json").write_text(
        json.dumps(
            {
                "method": "ntm_frames",
                "version": "v2",
                "threshold": 0.5,
                "evaluation": str(evaluation),
                "bar": {"N1": False},
            }
        )
    )
    track = fi.table_track(paths, 7, NTM, table)
    assert (track.table["kind"], track.source.tier) == ("detector", fi.lf.GENERATED)
    record = fi.track_record(track, (0, 300))
    assert record["decision_threshold"] == 0.5
    assert record["performance"]["f1"] == 0.457
    assert fs.is_detector(record)
    caption = fs.caption(7, {NTM: record}, {})
    assert (
        "while the TM detector (held-out F1 0.46, below our 0.7 bar, shown at 0.5) "
        "is positive" in caption
    )
    notes = fs.appendix_notes(7, {NTM: record}, {"tokeye_threshold": 0.45})
    assert "Operating probability thresholds: TokEye 0.45; TM 0.5." in notes
    assert "TM: detector (suggestions; held-out F1 0.46 on 61 shots" in notes

    # an event's imported (format) table and a curated source are the legacy tier
    imported = _table(tmp_path / "format/elm.csv", "7,1,100,200,\n")
    assert fi.table_track(paths, 7, ELM, imported).source.tier == fi.lf.LEGACY
    curated = _table(
        tmp_path / "merged.csv", "7,0,40,L\n", "shot,t_start,t_end,regimes"
    )
    regimes = _table(tmp_path / "regimes.csv", "7,2,0,40,\n8,1,0,40,\n")
    (tmp_path / "regimes.meta.json").write_text(
        json.dumps({"sources": {"curated": {"table": str(curated)}}})
    )
    kinds = [
        fi.table_track(paths, shot, "confinement", regimes).table["kind"]
        for shot in (7, 8)
    ]
    assert kinds == ["curated", "labels"], "8 has no curated rows: the rule's"


def test_a_gold_shot_never_draws_an_expert_review(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    ae = next(s for s in fi.lf.TRACKS if s.key == AE)
    review = fi.lf.Track(ae, ae.sources[0], tmp_path / "labels.csv", (Row(0, 9, 1),))
    honour = True  # whether the catalog honours `expert` (else the guard must)

    def catalog(*args, skip=frozenset(), expert=True, **kwargs):
        held = AE not in skip and (expert or not honour)
        return ((review,) if held else ()), None

    monkeypatch.setattr(fi, "catalog_tracks", catalog)
    monkeypatch.setattr(fi, "is_gold", lambda tables, shot: shot == 7)
    tracks, _ = fi.shot_tracks(paths, fi.lf.Candidate(7, 2024, (0, 100)))
    assert tracks == (), "the gold shot's tiers skip the expert review"
    (track,), _ = fi.shot_tracks(paths, fi.lf.Candidate(8, 2024, (0, 100)))
    assert track is review, "a shot outside the gold set may"
    honour = False
    with pytest.raises(SystemExit, match="gold evaluation shot"):
        fi.shot_tracks(paths, fi.lf.Candidate(7, 2024, (0, 100)))
    supplied = {AE: _table(tmp_path / "ae.csv", "7,1,0,9,\n")}
    (track,), _ = fi.shot_tracks(
        paths, fi.lf.Candidate(7, 2024, (0, 100)), rows=supplied
    )
    assert track.table["how"] == fi.SUPPLIED

    # the AE choice itself: the expert review first, unless the shot is gold
    _table(labels.labels_path(paths.label_tables / AE), "7,1,0,9,\n")
    predictions = _table(tmp_path / "pred.csv", "7,0,0,9,\n")
    first = fs.reviewed_or_stored_ae(paths, 7, predictions)
    assert first.source.tier == fi.lf.SILVER
    gold = fs.reviewed_or_stored_ae(paths, 7, predictions, expert=False)
    assert (gold.source.tier, gold.file) == (fi.lf.GENERATED, predictions)


# the transient overlay of (b)


def _z(tra_line=0.3, coh_box=False, zoom_box=False, tra_block=False):
    """A synthetic TokEye record: a wide pass (0.25 ms x 0.5 kHz, 0-160 kHz) with
    a transient line one column wide at 50 ms over 10-150 kHz at probability
    `tra_line`, and optionally a coherent box over it (wide 70-90 kHz) or a zoom
    pass (1 ms x 0.125 kHz, 0-60 kHz) whose coherent mask lights 20-30 kHz there.
    `tra_block`: transient time all round the box (60-100 kHz, 37-62 ms), so the
    red meets the white mask on every side."""
    rng = np.random.default_rng(0)
    t, f = np.arange(400) * 0.25, np.arange(320) * 0.5
    tra = np.zeros((f.size, t.size), np.float32)
    tra[(f >= 10) & (f < 150), 200] = tra_line
    coh = np.zeros_like(tra)
    if coh_box:
        coh[(f >= 70) & (f < 90), 190:211] = 0.9
    if tra_block:
        tra[(f >= 60) & (f < 100), 150:250] = 0.3
        tra[coh > 0] = 0.0
    tz, fz = np.arange(100) * 1.0, np.arange(480) * 0.125
    cz = np.zeros((fz.size, tz.size), np.float32)
    if zoom_box:
        cz[(fz >= 20) & (fz < 30), 45:56] = 0.9
    return {
        "wide_t_ms": t, "wide_f_khz": f, "wide_raw": rng.random(tra.shape),
        "wide_coh": coh, "wide_tra": tra, "wide_row_lit": np.zeros(f.size),
        "zoom_t_ms": tz, "zoom_f_khz": fz, "zoom_raw": rng.random(cz.shape),
        "zoom_coh": cz, "zoom_tra": np.zeros_like(cz), "zoom_row_lit": np.zeros(fz.size),
    }  # fmt: skip


def test_a_one_column_transient_line_survives_print_pooling():
    figure = fi.Figure(figsize=(fi.PAGE_IN, fi.HEIGHT_IN))
    ax = figure.add_axes([0.1, 0.5, 0.85, 0.2])
    ax.set_ylim(0, 150)
    band = SimpleNamespace(f=np.linspace(0, 150, 600))
    mask = np.zeros((600, 12000), bool)
    mask[100:500, 7777] = True
    pooled = fi.print_pool(ax, band, mask)
    assert pooled.shape[1] < 1000, "pooled to the print pixels"
    assert pooled.any(axis=0).sum() == 1, "one print column, not averaged away"


def test_the_transient_display_cut_is_its_own_not_tokeyes(monkeypatch):
    z = _z()
    wide = fi.Band(z, "wide", 0, 151, 0, 100, transient_cut=0.2)
    line = wide.tra[:, 200]
    assert line[(wide.f >= 10) & (wide.f < 150)].all(), "0.3 passes the 0.2 cut"
    assert not fi.Band(z, "wide", 0, 151, 0, 100, transient_cut=0.45).tra.any()
    monkeypatch.setattr(fi, "TOKEYE_THRESHOLD", 0.9)
    again = fi.Band(z, "wide", 0, 151, 0, 100, transient_cut=0.2)
    assert (again.tra == wide.tra).all(), "the coherent threshold does not move it"


def test_the_transient_is_not_drawn_where_a_drawn_mask_is_lit():
    z = _z(coh_box=True, zoom_box=True)
    wide = fi.Band(z, "wide", 0, 151, 0, 100, transient_cut=0.2)
    zoom = fi.Band(z, "zoom", 0, fi.ZOOM_TOP_KHZ, 0, 100)
    assert not (wide.tra & wide.lit).any()
    assert not wide.tra[(wide.f >= 71) & (wide.f < 89), 200].any(), "wide mask"
    red = fi.transient_overlay(wide, zoom)[:, 200]
    assert not red[(wide.f >= 21) & (wide.f < 29)].any(), "zoom mask below 50 kHz"
    assert red[(wide.f >= 11) & (wide.f < 19)].all(), "drawn below 50 kHz"
    assert red[(wide.f >= 91) & (wide.f < 149)].all(), "and above it"


def _panel_rgb(figure, ax):
    rgb, _, _ = fi.panel_raster(figure, ax)
    return rgb


def _frequency_panels(z, **kwargs):
    """`draw_frequency_panels` on a synthetic record: the figure, panel (b) and what
    it returns (`kwargs`: `transient`, left out for the default)."""
    raw = fi.Band(z, "wide", 0, 151, 0, 100, transient_cut=0.2)
    zoom = fi.Band(z, "zoom", 0.0, fi.SPLIT_KHZ, 0, 100)
    top = fi.Band(z, "wide", fi.ZOOM_TOP_KHZ, 151, 0, 100)
    figure = fi.Figure(figsize=(fi.PAGE_IN, fi.HEIGHT_IN))
    ax = {
        "raw": figure.add_axes([0.1, 0.7, 0.2, 0.2]),
        "pr": figure.add_axes([0.1, 0.4, 0.2, 0.2]),
    }
    for a in ax.values():
        a.set_xlim(0, 100)
    return figure, ax["pr"], fi.draw_frequency_panels(ax, raw, zoom, top, **kwargs)


def _red(rgb):
    return (np.abs(rgb - fi.to_rgb(fi.TRANSIENT_COLOUR)) < 0.05).all(axis=-1)


def test_panel_b_draws_no_transient_overlay_by_default():
    """The owner turned the red transient lines off on 2026-10-14 as too
    distracting: the default draws none (no artist, no red pixel, nothing returned)
    though the transient channel has pixels, and `transient=True` brings them back
    at the 0.35 cut."""
    z = _z(coh_box=True, zoom_box=True, tra_block=True)
    assert fi.SHOW_TRANSIENT is False and fi.TRANSIENT_THRESHOLD == 0.35
    assert inspect.signature(fi.draw).parameters["transient"].default is False
    assert fi.Band(z, "wide", 0, 151, 0, 100, transient_cut=0.2).tra.any()
    figure, pr, drawn = _frequency_panels(z)
    assert drawn == {}, "no pass, no columns"
    assert not [a for a in pr.get_children() if a.get_gid() == "panel-b:transient"]
    assert not _red(_panel_rgb(figure, pr)).any(), "no red pixel in panel (b)"
    classes = fi.drawn_classes(figure, pr)
    assert not classes.get("transient", np.zeros(1)).any(), "no chip names a transient"
    # opt in: the code path is still there
    figure, pr, drawn = _frequency_panels(z, transient=True)
    assert drawn[fi.TRANSIENT_PASS] > 0
    assert _red(_panel_rgb(figure, pr)).sum() > 100


def test_no_lit_mask_pixel_turns_red_at_print_resolution():
    """The red is pooled to whole print pixels; a print pixel it shares with a lit
    mask pixel stays white, as the mask is drawn over the red (`MASK_Z`)."""
    z = _z(coh_box=True, zoom_box=True, tra_block=True)
    figure, pr, drawn = _frequency_panels(z, transient=True)
    assert drawn[fi.TRANSIENT_PASS] > 0
    with_red = _panel_rgb(figure, pr)
    plain_figure, plain_pr, plain_drawn = _frequency_panels(z, transient=False)
    assert plain_drawn == {}
    plain = _panel_rgb(plain_figure, plain_pr)
    white = (plain > 0.9).all(axis=-1)
    red = _red(with_red)
    assert white.sum() > 100 and red.sum() > 100, "both drawn"
    assert not (white & red).any(), "a lit mask pixel turned red"
    assert (with_red[white] > 0.9).all(), "the mask reads white with the red drawn"
    # the test bites: with the red over the mask, pooling paints mask pixels red
    for artist in pr.get_children():
        if artist.get_gid() == fi.DRAWN_GID + "transient":
            artist.set_zorder(fi.MASK_Z + 1)
    over = _panel_rgb(figure, pr)
    assert (white & _red(over)).any()


def test_a_tm_outline_over_an_n1_ridge_yields_no_n3_chip():
    """An anti-aliased TM outline (orange) over an n=1 ridge (blue) blends, along
    its edges, to n=3's hue (yellow): a class is named only where its own artists
    draw it (`drawn_classes`), so the blend names no n=3."""
    figure = fi.Figure(figsize=(fi.PAGE_IN, fi.HEIGHT_IN))
    pr = figure.add_axes([0.14, 0.5, 0.84, 0.24])
    pr.set_xlim(0, 1000)
    pr.set_ylim(0, 150)
    pr.set_facecolor(fi.PANEL_FACE)
    t, f = np.linspace(0, 1000, 2000), np.linspace(0, 30, 240)
    T, F = np.meshgrid(t, f)
    centre = 10 + 0.002 * (T - 500)  # a gently sloped ridge, 6 kHz tall
    ridge = (np.abs(F - centre) < 3) & (T > 100) & (T < 900)
    rgba = np.zeros((*ridge.shape, 4))
    rgba[ridge] = (*fi.to_rgb(fi.N_COLOURS[1]), 1.0)
    pr.imshow(
        rgba, origin="lower", aspect="auto", extent=(0, 1000, 0, 30),
        interpolation="nearest", zorder=2, gid=fi.DRAWN_GID + "n",
    )  # fmt: skip
    outline = (np.abs(F - centre) < 1.5) & (T > 150) & (T < 850)
    pr.contour(
        t, f, outline.astype(float), levels=[0.5], colors=fi.NTM_CONTOUR_COLOUR,
        linewidths=fi.NTM_CONTOUR_LW, zorder=5,
    ).set_gid(fi.DRAWN_GID + "TM")  # fmt: skip
    by_hue = fi.event_classes(_panel_rgb(figure, pr))
    found = fi.largest_region(by_hue["n=3"], fi.LABEL_MERGE_DEFAULT_PX)
    assert found and found[1] >= fi.LABEL_MIN_REGION_PX, "hue alone sees an n=3"
    labels = fi.label_events(figure, pr, (0, 1000), transient_drawn=False)
    placed = {c["text"] for c in labels["placed"]}
    assert "n=3" not in placed, "no chip names the blend"
    assert {"n=1", "TM"} <= placed, "the classes drawn are named"
    n3 = labels["regions"]["n=3"]
    assert n3["pixels"] == 0 and n3["hue_pixels"] >= fi.LABEL_MIN_REGION_PX
    for chip in labels["placed"]:
        assert chip["region_px"] <= labels["regions"][chip["text"]]["pixels"]


def test_the_transient_chip_may_name_any_line_of_a_named_size():
    mask = np.zeros((50, 200), bool)
    mask[5:45, 20] = True  # a 40-pixel line
    mask[5:45, 150] = True  # another
    mask[10:15, 100] = True  # a speck
    kept = fi.regions_at_least(mask, fi.LABEL_MERGE_DEFAULT_PX, 30)
    assert kept[:, 20].sum() == kept[:, 150].sum() == 40 and not kept[:, 100].any()


# the label rows


def _row_axes():
    figure = fi.Figure(figsize=(fi.PAGE_IN, 1.0))
    ax = figure.add_axes([0.1, 0.1, 0.85, 0.8])
    ax.set_xlim(0, 1000)
    return ax


def test_a_class_name_is_written_on_a_run_only_where_it_fits():
    ax = _row_axes()
    runs = [Row(0, 500, 1), Row(500, 505, 2), Row(505, 1000, 9)]
    named = fi.name_runs(ax, runs, 0, 1000)
    assert [(n["text"], n["run_ms"]) for n in named] == [
        ("H", [0, 500]),
        ("L", [500, 505]),
    ], "a class with no short name is not written"
    fits, narrow = named
    x0, _, x1, _ = fits["bounds"]
    run = [ax.transData.transform((t, 0))[0] for t in (0, 500)]
    figure_px = ax.figure.bbox.width
    assert run[0] / figure_px < x0 < x1 < run[1] / figure_px, "inside its run"
    assert narrow["bounds"] is None, "5 ms is too narrow for its text"
    assert [t.get_text() for t in ax.texts] == ["H"], "the unfit text is removed"


def test_the_uncertain_hatch_is_lines_clipped_to_the_box_in_one_phase():
    step = 6.0
    for box in [(0, 0, 40, 10), (13, 2, 47, 9), (0, 0, 3, 30)]:
        segments = fi.hatch_segments(box, step)
        assert segments
        for (ax_, ay), (bx, by) in segments:
            for x, y in ((ax_, ay), (bx, by)):
                assert box[0] - 1e-9 <= x <= box[2] + 1e-9
                assert box[1] - 1e-9 <= y <= box[3] + 1e-9
            assert abs((by - ay) - (bx - ax_)) < 1e-9, "a '/' line"
            # anchored to the page: x - y on the step's grid, so the runs on
            # either side of a cut continue the same lines
            assert abs(((ax_ - ay) / step) - round((ax_ - ay) / step)) < 1e-9
    assert fi.hatch_segments((5, 0, 5, 10), step) == []

    ax = _row_axes()
    lines = fi.hatch_lines(ax, 100, 300, 0.12, 0.88, "#D55E00")
    assert lines is not None and lines.get_linewidth()[0] == fi.UNCERTAIN_HATCH_LW
    assert fi.hatch_lines(ax, 2000, 3000, 0.12, 0.88, "#D55E00") is None, "off view"
    shortest = fi.UNCERTAIN_HATCH_MIN_PT * ax.figure.dpi / 72
    assert fi.UNCERTAIN_HATCH_MIN_PT >= 2.0, "a corner never prints as a stub"
    for segment in lines.get_segments():
        a, b = ax.transData.transform(segment)
        assert np.hypot(*(b - a)) >= shortest - 1e-6, "no corner nub"
    # a run too narrow for any whole line still gets one: never blank
    pt = 1000 / (ax.bbox.width * 72 / ax.figure.dpi)  # ms per pt on this axis
    for width_pt in (0.3, 0.7, 1.2):
        (segment,) = fi.hatch_lines(
            ax, 500, 500 + width_pt * pt, 0.12, 0.88, "#D55E00"
        ).get_segments()
        (xa, ya), (xb, yb) = segment
        assert xa == xb == pytest.approx(500 + width_pt * pt / 2), "at its middle"
        assert (ya, yb) == pytest.approx((0.12, 0.88)), "the row's full height"


def test_the_transient_cut_must_lie_in_zero_to_one():
    assert fi.display_cut("0.2") == 0.2 and fi.display_cut("1") == 1.0
    for bad in ("0", "-0.1", "1.5", "nan", "x"):
        with pytest.raises(argparse.ArgumentTypeError):
            fi.display_cut(bad)


# plain names and the q = 1 gate


def test_the_appendix_names_a_table_plainly_never_its_method_id():
    def name(**table):
        return fs.table_name({"source_table": table})

    assert name(method="elm_jump", version="v1") == "D-alpha jump-and-hold rule"
    # a gate is named only where the table's metadata names one (its no-q=1 rows
    # recorded, even none on this shot), so an ungated version never claims one
    assert name(method="saw_physics", version="v1") == "ECE physics rule"
    assert name(method="saw_physics", version="v3", no_q1_rows_ms=[]) == (
        "ECE physics rule with q = 1 gate"
    )
    assert name(method="crash", version="v1") == "ECE crash rule"
    assert name(method="tm_mhd", version="v1") == "MHD resonant-mode rule"
    assert name(method="regimes", kind="curated") == "curated labels"
    assert name(method="new_rule", version=None) == "generated labels"
    assert name(method=None, kind="detector") == "detector output"
    for method in fs.METHOD_NAMES.values():
        assert "_" not in method and "None" not in method
        assert "q = 1" not in method, "the gate comes from the table, not the name"


def test_a_gated_sawtooth_tables_no_q1_time_is_recorded_and_named(tmp_path):
    paths = _paths(tmp_path)
    table = _table(
        tmp_path / "saw.csv",
        "7,3,0,100,1.0\n7,0,100,400,\n7,0,400,500,1.0\n7,1,500,520,1.0\n",
    )
    meta = {
        "method": "saw_physics",
        "version": "v3",
        "rule": {"gate": {"no_q1": "q_min >= 1.2"}},
        "per_shot": {"7": {"physics": True, "no_q1_ms": 300.0}},
    }
    (tmp_path / "saw.meta.json").write_text(json.dumps(meta))
    track = fi.table_track(paths, 7, fi.mode_tags.SAWTOOTH, table)
    assert track.table["no_q1_rows_ms"] == [[100.0, 400.0]], "the blank-confidence row"
    assert track.table["no_q1_ms"] == 300.0
    record = fi.track_record(track, (200, 1000))
    assert record["no_q1_intervals_ms"] == [[200, 400.0]]
    assert fs.sawtooth_caption(record) == (
        "Sawtooth: present 20 ms, absent 300 ms (ECE physics rule with q = 1 gate); "
        "200 ms of the absent time has no q = 1 surface."
    )
    record = fi.track_record(track, (0, 400))
    assert fs.sawtooth_caption(record).endswith(
        "; all of the absent time has no q = 1 surface."
    )

    del meta["rule"]  # an ungated table records no gate
    (tmp_path / "saw.meta.json").write_text(json.dumps(meta))
    track = fi.table_track(paths, 7, fi.mode_tags.SAWTOOTH, table)
    assert "no_q1_rows_ms" not in track.table
    record = fi.track_record(track, (0, 1000))
    assert "no_q1_intervals_ms" not in record
    assert "q = 1" not in fs.sawtooth_caption(record), "no gate claimed"
