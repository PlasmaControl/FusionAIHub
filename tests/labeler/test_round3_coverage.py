"""The paper's coverage of the frame-model phenomena (F2): their original labels
with the owner's reviews over them, counted from the split's meta, the owner's
review apart, the frame models' train/val/test splits and their suggestions,
marked where the model failed its primary bar or is effectively always (F8)."""

from __future__ import annotations

import io
import json
from dataclasses import replace
from itertools import pairwise

import pandas as pd
import pytest

from labeler import frames
from labeler.ae.xpower import train
from labeler.events import suggestions
from labeler.events.review import labels
from labeler.frames import shots as frame_shots
from labeler.paper import AE, COMING, build, coverage, title
from labeler.paper.snapshot import Snapshot

from . import ae_tree
from . import paper_tree as tree

ELM, SAW, NTM, HMODE = (
    "edge_localized_mode",
    "sawtooth_oscillation",
    "neoclassical_tearing_mode",
    "high_confinement_mode",
)
SUMMARY = (
    "shot,year,window_start_ms,window_end_ms,frames,present_frames,"
    "not_observable_frames,present_runs,max_prob\n"
    "501,2024,0,5000,500,0,0,0,0.1\n"
    "502,2024,0,5000,500,40,0,2,0.9\n"
    "503,2025,0,5000,500,3,0,1,0.7\n"
    "504,,0,5000,500,0,0,0,0.2\n"
)  # by year {?: (1, 0), 2024: (2, 1), 2025: (1, 1)}
ELM_LABELS = (
    "shot,category,t_start,t_end,confidence\n"
    "301,0,0,100,\n301,1,100,300,\n301,0,300,2000,\n"
    "302,0,0,2000,\n"
    "309,1,0,1000,\n309,0,1000,2000,\n"
)  # 3 reviewed by the owner; 309 saved after the split
ELM_SPLIT = (
    "shot,split,positive,roster,owner\n"
    "301,train,1,1,1\n302,test,0,1,1\n"
    "303,train,1,0,0\n304,train,0,0,0\n305,train,1,0,0\n306,val,1,0,0\n"
    "307,test,0,0,0\n"
)  # 4 / 1 / 2: the owner's saved shots split with the rest, no owner split
SAW_SPLIT = (
    "shot,split,positive,roster,owner\n401,train,1,1,0\n402,train,1,1,0\n"
    "403,val,1,1,0\n404,test,0,1,0\n"
)
ELM_META = {
    "method": "elm_frames",
    "labelled_shots": 576,
    "positive_shots": 443,
    "legacy_labelled_shots": 575,
    "present_s": 1234.5,
    "owner": {"saved": 3, "overriding": 2, "added": 1},
}  # the merged target's counts: the grids are never re-read by the coverage
ELM_TABLE_META = {
    "bar": {"E1": False, "E2": True, "E3": True, "all": False},
    "effectively_always": False,
}  # E1, the primary bar, failed
AE_COUNTS = coverage.Counts(
    reviewed=4,
    positive=2,
    present_s=0.9,
    split={"train": 1, "val": 1, "test": 1},
    by_year={coverage.UNKNOWN_YEAR: (1, 0), 2024: (2, 1), 2025: (1, 1)},
    unsplit=1,
)


@pytest.fixture(autouse=True)
def paths(tmp_path, monkeypatch):
    return tree.temporary_paths(tmp_path, monkeypatch)


def _elm(tmp_path) -> coverage.Counts:
    event = tmp_path / "elm"
    labels.labels_path(event).parent.mkdir(parents=True)
    labels.labels_path(event).write_text(ELM_LABELS)
    (tmp_path / "split.csv").write_text(ELM_SPLIT)
    (tmp_path / "summary.csv").write_text(SUMMARY)
    split = coverage.frame_split(pd.read_csv(tmp_path / "split.csv"))
    summary = pd.read_csv(tmp_path / "summary.csv")
    return coverage.frame_counts(
        labels.read_saved(event),
        split,
        summary,
        meta=ELM_META,
        table_meta=ELM_TABLE_META,
        primary=coverage.FRAME_SOURCES[ELM].primary,
    )


def _saw(meta: dict | None = None) -> coverage.Counts:
    split = {401: "train", 402: "train", 403: "val", 404: "test"}
    return coverage.frame_counts({}, split, None, meta=meta)


def test_a_frame_models_counts(tmp_path):
    elm = _elm(tmp_path)
    assert elm.reviewed == 3, "the owner's saved shots, 309 among them"
    assert (elm.labelled, elm.positive, elm.present_s) == (576, 443, 1234.5)
    assert elm.labelled_shots == 576, "the meta's, not the owner's saves"
    assert elm.split == {"train": 4, "val": 1, "test": 2}, "no owner split"
    assert elm.unsplit is None, "its unsplit shots are those the split left out"
    assert (elm.frame, elm.cross_validated) == (True, False)
    assert (elm.suggested, elm.suggested_positive) == (4, 2)
    assert (elm.bar_met, elm.always) == (False, False)
    assert elm.marks == [coverage.BAR_NOT_MET]
    saw = _saw()
    assert (saw.reviewed, saw.labelled, saw.positive, saw.present_s) == (
        0,
        None,
        None,
        None,
    ), "no meta: not counted, not a zero"
    assert saw.split == {"train": 2, "val": 1, "test": 1}
    assert saw.by_year is None and saw.suggested is None
    assert (saw.bar_met, saw.always, saw.marks) == (None, None, [])
    bare = coverage.frame_counts({}, None, None)
    assert (bare.split, bare.unsplit, bare.frame) == (None, None, True)
    assert AE_COUNTS.labelled_shots == AE_COUNTS.reviewed == 4, "AE's labels"
    assert (AE_COUNTS.frame, AE_COUNTS.marks) == (False, []), "AE's defaults"


def test_a_frame_model_s_marks_are_its_table_meta_s():
    split = {1: "train"}
    for bar, always, marks in (
        (True, False, []),
        (True, True, [coverage.ALWAYS]),
        (False, True, [coverage.BAR_NOT_MET, coverage.ALWAYS]),
    ):
        table_meta = {"bar": {"S1": bar, "S2": True}, "effectively_always": always}
        counts = coverage.frame_counts(
            {}, split, None, table_meta=table_meta, primary="S1"
        )
        assert counts.marks == marks
    unmarked = replace(AE_COUNTS, bar_met=False, always=True)
    assert unmarked.marks == [], "only a frame model is marked"
    with pytest.raises(KeyError, match="present_s"):
        coverage.frame_counts({}, split, None, meta={"labelled_shots": 1})
    with pytest.raises(KeyError, match="effectively_always"):
        coverage.frame_counts({}, split, None, table_meta={"bar": {}}, primary="S1")


def test_a_split_outside_the_three_is_refused():
    for odd in ("valid", "owner"):  # v1's owner split among them
        table = pd.DataFrame({"shot": [1, 2], "split": ["train", odd]})
        with pytest.raises(ValueError, match=odd):
            coverage.frame_split(table)


def test_the_frame_sources_are_the_four_paper_phenomena():
    assert list(coverage.FRAME_SOURCES) == [NTM, HMODE, ELM, SAW]
    assert "disruption" not in coverage.FRAME_SOURCES
    assert {s.method for s in coverage.FRAME_SOURCES.values()} <= set(frames.SPECS)
    primaries = [s.primary for s in coverage.FRAME_SOURCES.values()]
    assert primaries == ["N1", "H1", "E1", "S1"]
    detectors = [c for c, s in coverage.FRAME_SOURCES.items() if s.detector]
    assert detectors == [SAW], "sawtooth's labels are the detector's (D56)"
    assert coverage.FRAME_SOURCES[SAW].origin == (
        "the ece_sawtooth v3 detector (ECE and SXR crashes; D56 as amended)"
    ), "v3, not v2, which over-called"
    assert coverage.tick(SAW) == "sawtooth (detector)"
    assert [coverage.tick(c) for c in (AE, ELM)] == [title(AE), "ELMing"]
    assert "Jalalvand" not in json.dumps(
        {k: [s.origin, s.also] for k, s in coverage.FRAME_SOURCES.items()}
    )


def _texts(ax) -> list[str]:
    return [t.get_text() for t in ax.texts]


def _ticks(ax) -> list[str]:
    return [t.get_text() for t in ax.get_yticklabels()]


def _widths(ax) -> list[float]:
    return [bar.get_width() for bar in ax.patches]


def _apart(fig, ax) -> bool:
    """No two y tick labels of `ax` overlap."""
    fig.draw_without_rendering()
    boxes = sorted(
        (t.get_window_extent() for t in ax.get_yticklabels()), key=lambda b: b.y0
    )
    return all(b.y0 >= a.y1 for a, b in pairwise(boxes))


def test_the_figure_with_the_frame_phenomena(tmp_path):
    counts = {AE: AE_COUNTS, ELM: _elm(tmp_path), SAW: _saw()}
    fig = coverage.draw_coverage(counts, tmp_path / "fig_coverage")
    assert (tmp_path / "fig_coverage.pdf").is_file()
    assert len(fig.axes) == 7, "AE's row of three, the frame splits' row of four"
    shots, present = fig.axes[:2]
    assert sum(t == COMING for t in _texts(shots)) == 2, "NTM and H-mode"
    assert _texts(shots).count(coverage.NOT_RUN) == 1, "sawtooth: no meta read"
    assert coverage.NONE_REVIEWED not in _texts(shots)
    # On a log axis from 1: labelled AE, ELM; positive AE, ELM, each numbered.
    assert shots.get_xscale() == "log" and present.get_xscale() == "log"
    assert [bar.get_x() for bar in shots.patches] == [1] * 4
    assert _widths(shots) == [4 - 1, 576 - 1, 2 - 1, 443 - 1]
    assert {"4", "576", "2", "443"} <= set(_texts(shots))
    assert shots.get_xlim() == pytest.approx((1, 576**coverage.ROOM))
    assert _ticks(shots)[-2:] == ["ELMing", "sawtooth (detector)"]
    # The present time on a log axis from 1 s too: the meta's 1234.5 s a bar,
    # AE's 0.9 s no bar but its number at the axis's start.
    assert _widths(present) == [1234.5 - 1], "the meta's present_s"
    assert [bar.get_x() for bar in present.patches] == [1]
    [short] = [t for t in present.texts if t.get_text() == "0.9"]
    assert short.xy[0] == 1, "AE's present_s"
    assert "1,234.5" in _texts(present)
    assert present.get_xlim() == pytest.approx((1, 1234.5**coverage.ROOM))
    ntm_split, hmode_split, elm_split, saw_split = fig.axes[3:7]
    for ax in (ntm_split, hmode_split):
        assert _texts(ax) == [COMING] and not ax.patches
    assert _widths(elm_split) == [4, 1, 2]
    assert _texts(elm_split) == ["4", "1", "2"]
    assert _widths(saw_split) == [2, 1, 1]
    for ax in (elm_split, saw_split):
        assert _ticks(ax) == ["train", "val", "test"], "no legacy, no owner bar"
    assert elm_split.get_title() == "ELMing: model split\n(bar not met)", "F8's mark"
    assert saw_split.get_title() == "sawtooth: model split", "no table meta read"
    assert not [ax for ax in fig.axes if "suggest" in ax.get_title()], "no years"
    [legend] = fig.legends
    assert [t.get_text() for t in legend.get_texts()] == [
        "labelled",
        "with a present span",
    ]
    assert tree.small_text(fig) == []
    for ax in (shots, elm_split, saw_split):
        assert _apart(fig, ax), ax.get_title()


BY_YEAR = {
    2021: (481, 130),
    2022: (1567, 420),
    2023: (848, 250),
    2024: (1108, 360),
    2025: (868, 270),
    coverage.UNKNOWN_YEAR: (450, 90),
}  # a population-like application: five campaign years and the roster's "?"


def _tick_labels(ax) -> list:
    """The tick labels `ax` draws: with a text, in its view (its tick labels
    include one beyond each end)."""
    (x0, x1), (y0, y1) = sorted(ax.get_xlim()), sorted(ax.get_ylim())
    xs = [t for t in ax.get_xticklabels() if x0 <= t.get_position()[0] <= x1]
    ys = [t for t in ax.get_yticklabels() if y0 <= t.get_position()[1] <= y1]
    return [t for t in xs + ys if t.get_text()]


def _population(labelled: int = 900) -> coverage.Counts:
    split = {"train": 20400, "val": 4300, "test": 4350}
    return coverage.Counts(
        108, 610, 812.4, split, BY_YEAR, labelled=labelled, frame=True
    )


def test_a_frame_phenomenon_says_none_labelled_only_when_counted(tmp_path):
    empty = {"labelled_shots": 0, "positive_shots": 0, "present_s": 0.0}
    for saw, mark in (
        (_saw(empty), coverage.NONE_LABELLED),
        (_saw(), coverage.NOT_RUN),
    ):
        fig = coverage.draw_coverage({AE: AE_COUNTS, SAW: saw}, tmp_path / "fig")
        shots = fig.axes[0]
        texts = _texts(shots)
        assert texts.count(mark) == 1
        assert coverage.NONE_REVIEWED not in texts, "a frame phenomenon's labels"
        # A count of 0 draws no bar on the log axis, and keeps its number at 1;
        # a count not read (no meta) draws neither.
        zeros = [t for t in shots.texts if t.get_text() == "0"]
        counted = mark == coverage.NONE_LABELLED
        assert [t.xy[0] for t in zeros] == ([1, 1] if counted else [])
        assert len(shots.patches) == 2, "AE's two bars alone"


def test_at_population_sizes_the_keys_cover_no_bar_and_no_tick(tmp_path):
    labelled = (14210, 31876, 28521, 4822)  # the tearing archive's to the detector's
    marks = ((True, False), (False, True), (False, False), (False, True))
    counts = {AE: AE_COUNTS} | {
        category: replace(_population(n), bar_met=bar, always=always)
        for category, n, (bar, always) in zip(
            coverage.FRAME_SOURCES, labelled, marks, strict=True
        )
    }  # both marks on the widest title, sawtooth's
    fig = coverage.draw_coverage(counts, tmp_path / "fig_coverage")
    assert len(fig.axes) == 7
    assert all(ax.get_legend() is None for ax in fig.axes[3:7])
    fig.draw_without_rendering()
    keys = [legend.get_window_extent() for legend in fig.legends]
    assert len(keys) == 1, "one key: labelled / with a present span"
    ticks = [t.get_window_extent() for ax in fig.axes for t in _tick_labels(ax)]
    bars = [bar.get_window_extent() for ax in fig.axes for bar in ax.patches]
    panels = [ax.get_window_extent() for ax in fig.axes]
    assert len(bars) > 25
    for key in keys:
        assert not any(key.overlaps(box) for box in ticks + bars + panels)
    assert _apart(fig, fig.axes[0]), "the phenomena's names"
    assert _texts(fig.axes[5]) == ["20,400", "4,300", "4,350"]
    titles = [ax.get_title() for ax in fig.axes[3:7]]
    assert titles == [
        f"{title(NTM)}: model split",
        f"{title(HMODE)}: model split\n(bar not met, ≈ always)",
        f"{title(ELM)}: model split\n(bar not met)",
        f"{title(SAW)}: model split\n(bar not met, ≈ always)",
    ], "F8's marks on the split headings, both on the widest heading, sawtooth's"
    headings = [ax.title.get_window_extent() for ax in fig.axes[3:7]]
    above = [ax.get_tightbbox() for ax in fig.axes[:3]]
    assert not any(h.overlaps(a) for h in headings for a in above), "below AE's row"
    shown = [
        text
        for ax in fig.axes
        for text in (ax.title, ax.xaxis.label, ax.yaxis.label, *ax.texts)
    ] + [text for legend in fig.legends for text in legend.get_texts()]
    for text in shown + [t for ax in fig.axes for t in _tick_labels(ax)]:
        box = text.get_window_extent()
        inside = fig.bbox.x0 <= box.x0 and box.x1 <= fig.bbox.x1
        assert inside or not text.get_text(), text.get_text()
    assert tree.small_text(fig) == []


def test_no_suggestions_by_year_panel_or_key(tmp_path):
    """The owner, 2026-09-29 23:51: "remove the suggestions by year graphs". No
    panel of suggestions, AE's or a frame model's, and no key of them, alone or
    with the frame rows; the years are still counted, for the table."""
    ae = replace(AE_COUNTS, by_year=BY_YEAR)
    framed = {category: _population() for category in coverage.FRAME_SOURCES}
    for counts, panels in (({AE: ae}, 3), ({AE: ae} | framed, 7)):
        fig = coverage.draw_coverage(counts, tmp_path / "fig_coverage")
        assert len(fig.axes) == panels
        assert not [ax for ax in fig.axes if "suggest" in ax.get_title()]
        [key] = fig.legends
        assert [t.get_text() for t in key.get_texts()] == [
            "labelled",
            "with a present span",
        ]
        years = {t.get_text() for ax in fig.axes for t in _tick_labels(ax)}
        assert not years & {"2021", "2025", "?"}, "no year ticks"
    for gone in ("SUGGESTED", "FRAME_SUGGESTED", "_years_panel", "_years_key"):
        assert not hasattr(coverage, gone), gone
    assert not hasattr(coverage, "_frame_years_panel")
    assert ae.suggested == sum(n for n, _ in BY_YEAR.values()), "for the table"
    row = coverage.table_datasets({AE: ae}).splitlines()[5].split(" & ")
    assert row[-2:] == ["5322", "1520 \\\\"], "the table keeps the suggestions"


def test_ae_alone_keeps_its_one_row(tmp_path):
    fig = coverage.draw_coverage({AE: AE_COUNTS}, tmp_path / "fig_coverage")
    assert len(fig.axes) == 3
    assert tuple(fig.get_size_inches()) == (coverage.PAGE_IN, 2.3)
    assert coverage.NONE_REVIEWED not in _texts(fig.axes[0])


FRAME_CLAUSE = (
    "Labelled counts every shot of the original labels with the owner's reviews "
    "over them (blind and left-out shots too), Train, Val and Test the frame "
    "model's split of those with its inputs on disk, and No split is --: their "
    "other labelled shots are those the split left out (blind, or without the "
    "model's inputs on disk, or with no labelled bin in their window); original "
    "labels: "
)
DAGGER_CLAUSE = (
    "; $^\\dagger$: the frame model failed its primary bar (E1, H1, N1 or S1: bar "
    "not met) or is effectively the always-present baseline (≈ always), so its "
    "suggestions say little: "
)


def test_the_table_with_the_frame_phenomena(tmp_path):
    counts = {AE: AE_COUNTS, ELM: _elm(tmp_path), SAW: _saw()}
    lines = coverage.table_datasets(counts).splitlines()
    assert lines[0] == (
        "% Shots per phenomenon: labelled, reviewed by the owner, with any present "
        "span and their present time, the model's split (Train, Val, Test) and the "
        "extension's suggestions (not labels); -- where that run has not happened; "
        "AE's labels are the owner's reviews, so its Labelled = Reviewed = Train + "
        "Val + Test + No split (the reviewed shots saved after the model was "
        "trained); for ELMing and sawtooth, "
        + FRAME_CLAUSE
        + "ELMing: Hiro's table, sawtooth: the ece_sawtooth v3 detector (ECE and "
        + "SXR crashes; D56 as amended)"
        + DAGGER_CLAUSE
        + "ELMing (bar not met)"
    )
    assert lines[1] == "\\begin{tabular}{lcccccccccc}"
    assert lines[3] == (
        "Phenomenon & Labelled & Reviewed by the owner & Positive & Present (s) "
        "& Train & Val & Test & No split & Suggested & Suggested positive \\\\"
    )
    assert lines[5:10] == [
        "AE & 4 & 4 & 2 & 0.9 & 1 & 1 & 1 & 1 & 4 & 2 \\\\",
        "TM & \\multicolumn{10}{c}{coming} \\\\",
        "H-mode & \\multicolumn{10}{c}{coming} \\\\",
        "ELMing & 576 & 3 & 443 & 1234.5 & 4 & 1 & 2 & -- & 4$^\\dagger$ & 2 \\\\",
        "sawtooth & -- & 0 & -- & -- & 2 & 1 & 1 & -- & -- & -- \\\\",
    ]
    ae = lines[5].removesuffix(" \\\\").split(" & ")
    labelled, reviewed, train_val_test, unsplit = ae[1], ae[2], ae[5:8], ae[8]
    assert (
        int(labelled) == int(reviewed) == sum(map(int, train_val_test)) + int(unsplit)
    ), "the comment's arithmetic holds for AE's row"


def test_h_mode_names_l_mode_with_its_table(tmp_path):
    meta = {"labelled_shots": 4822, "positive_shots": 4800, "present_s": 12.3}
    table_meta = {"bar": {"H1": True, "H2": False}, "effectively_always": True}
    hmode = coverage.frame_counts(
        {},
        {1: "train"},
        pd.read_csv(io.StringIO(SUMMARY)),
        meta=meta,
        table_meta=table_meta,
        primary="H1",
    )
    counts = {AE: AE_COUNTS, HMODE: hmode}
    lines = coverage.table_datasets(counts).splitlines()
    assert lines[0].endswith(
        "; for H-mode, "
        + FRAME_CLAUSE
        + "H-mode (and L-mode): Jalal Butt's table"
        + DAGGER_CLAUSE
        + "H-mode (≈ always)"
    )
    assert lines[7] == (
        "H-mode & 4822 & 0 & 4800 & 12.3 & 1 & 0 & 0 & -- & 4$^\\dagger$ & 2 \\\\"
    )
    fig = coverage.draw_coverage(counts, tmp_path / "fig")
    assert fig.axes[4].get_title() == "H-mode: model split\n(≈ always)"


def test_an_unmarked_model_has_no_dagger(tmp_path):
    elm = replace(_elm(tmp_path), bar_met=True, always=False)
    assert elm.marks == []
    lines = coverage.table_datasets({AE: AE_COUNTS, ELM: elm}).splitlines()
    assert "dagger" not in lines[0]
    assert lines[8] == "ELMing & 576 & 3 & 443 & 1234.5 & 4 & 1 & 2 & -- & 4 & 2 \\\\"


@pytest.fixture
def runs(tmp_path, monkeypatch):
    """test_paper_build's tree, with ELM's four inputs and sawtooth's split."""
    paths = ae_tree.build(tmp_path, {101: "train", 102: "valid", 103: "valid"})
    models = ae_tree.chosen(paths, {101: "train", 102: "test", 103: "test"})
    (models / "evaluation.json").write_text(json.dumps(tree.ae_evaluation()))
    tree.record_labels(paths)
    ae_tree.env(monkeypatch, paths)
    live = labels.labels_path(paths.label_tables / ELM)
    live.parent.mkdir(parents=True)
    live.write_text(ELM_LABELS)
    saw_meta = {"labelled_shots": 4, "positive_shots": 3, "present_s": 2.5}
    for method, text, meta in (
        ("elm_frames", ELM_SPLIT, ELM_META),
        ("sawtooth_frames", SAW_SPLIT, saw_meta),
    ):
        frames.shots_file(paths, method).parent.mkdir(parents=True, exist_ok=True)
        frames.shots_file(paths, method).write_text(text)
        frames.shots_meta_file(paths, method).write_text(json.dumps(meta))
    frames.summary_file(paths, "elm_frames").parent.mkdir(parents=True)
    frames.summary_file(paths, "elm_frames").write_text(SUMMARY)
    table_meta = _table_meta(paths)
    table_meta.parent.mkdir(parents=True, exist_ok=True)
    table_meta.write_text(json.dumps(ELM_TABLE_META))
    return paths


def _table_meta(paths):
    """ELM's v2 suggestion table's meta, which `frames.apply` writes."""
    table = suggestions.table_path(paths, ELM, "elm_frames", frames.VERSION)
    return table.with_suffix(".meta.json")


def _theirs(entries: list[dict]) -> list[dict]:
    return [e for e in entries if "phenomenon" in e]


def test_the_build_counts_the_frame_phenomena(runs, tmp_path):
    manifest = build.build(runs, tmp_path / "paper")
    lines = (tmp_path / "paper" / "table_datasets.tex").read_text().splitlines()
    assert lines[5] == "AE & 3 & 3 & 3 & 1.8 & 1 & 0 & 2 & 0 & -- & -- \\\\"
    assert lines[6:10] == [
        "TM & \\multicolumn{10}{c}{coming} \\\\",
        "H-mode & \\multicolumn{10}{c}{coming} \\\\",
        "ELMing & 576 & 3 & 443 & 1234.5 & 4 & 1 & 2 & -- & 4$^\\dagger$ & 2 \\\\",
        "sawtooth & 4 & 0 & 3 & 2.5 & 2 & 1 & 1 & -- & -- & -- \\\\",
    ]
    assert "ELMing (bar not met)" in lines[0]

    def coming(c, m):
        where = [
            labels.labels_path(runs.label_tables / c),
            frames.shots_file(runs, m),
            frames.summary_file(runs, m),
        ]
        return {
            "reason": build.FRAMES_COMING,
            "phenomenon": c,
            "missing": [str(p) for p in where],
        }

    expected = [
        coming(NTM, "ntm_frames"),
        coming(HMODE, "hmode_frames"),
        {
            "reason": build.NO_FRAMES_SUMMARY,
            "phenomenon": SAW,
            "missing": [str(frames.summary_file(runs, "sawtooth_frames"))],
        },
    ]
    for product in ("fig_coverage", "table_datasets"):
        entries = manifest["partial"][product]
        assert entries[0]["reason"] == "extension not run: A2 failed (D47)"
        assert "phenomenon" not in entries[0]
        assert _theirs(entries) == expected
        assert entries[-len(expected) :] == expected, "after AE's own"
    theirs = {k for k in manifest["inputs"] if k.startswith(("labels_", "frames_"))}
    assert theirs == {
        "labels_edge_localized_mode",
        "frames_split_elm_frames",
        "frames_meta_elm_frames",
        "frames_summary_elm_frames",
        "frames_table_meta_elm_frames",
        "frames_split_sawtooth_frames",
        "frames_meta_sawtooth_frames",
    }
    assert manifest["inputs"]["frames_split_elm_frames"]["path"] == str(
        frames.shots_file(runs, "elm_frames", frames.VERSION)
    ), "v2's split, under frames/shots/v2/"
    assert "/shots/v2/" in manifest["inputs"]["frames_meta_elm_frames"]["path"]
    assert manifest["consistent"] is True


def test_a_split_without_its_meta_leaves_the_labelled_counts_out(runs, tmp_path):
    meta = frames.shots_meta_file(runs, "elm_frames")
    meta.unlink()
    manifest = build.build(runs, tmp_path / "paper")
    lines = (tmp_path / "paper" / "table_datasets.tex").read_text().splitlines()
    assert lines[8] == (
        "ELMing & -- & 3 & -- & -- & 4 & 1 & 2 & -- & 4$^\\dagger$ & 2 \\\\"
    ), "not counted, not a zero: the owner's saves alone are not its labels"
    missing = {
        "reason": build.NO_FRAMES_META,
        "phenomenon": ELM,
        "missing": [str(meta)],
    }
    assert missing in manifest["partial"]["fig_coverage"]
    assert "frames_meta_elm_frames" not in manifest["inputs"]


def test_a_table_without_its_meta_leaves_the_marks_out(runs, tmp_path):
    table_meta = _table_meta(runs)
    table_meta.unlink()
    manifest = build.build(runs, tmp_path / "paper")
    lines = (tmp_path / "paper" / "table_datasets.tex").read_text().splitlines()
    assert lines[8] == "ELMing & 576 & 3 & 443 & 1234.5 & 4 & 1 & 2 & -- & 4 & 2 \\\\"
    assert "dagger" not in lines[0]
    missing = {
        "reason": build.NO_FRAMES_TABLE_META,
        "phenomenon": ELM,
        "missing": [str(table_meta)],
    }
    assert missing in manifest["partial"]["table_datasets"]
    assert "frames_table_meta_elm_frames" not in manifest["inputs"]


def test_the_h_mode_row_reads_the_h_model_s_table_meta_alone(runs, tmp_path):
    # F15: there is no L row; H-mode's, "H-mode (and L-mode)", is marked
    # from hmode_frames' table meta, and lmode_frames' (its own flags) is not
    # read: here only L's says "effectively always".
    method = "hmode_frames"
    frames.shots_file(runs, method).write_text(SAW_SPLIT)
    meta = {"labelled_shots": 4, "positive_shots": 3, "present_s": 2.5}
    frames.shots_meta_file(runs, method).write_text(json.dumps(meta))
    frames.summary_file(runs, method).parent.mkdir(parents=True)
    frames.summary_file(runs, method).write_text(SUMMARY)
    h_meta = {"bar": {"H1": True, "H2": True, "all": True}}
    h_meta["effectively_always"] = False
    l_meta = h_meta | {"effectively_always": True, "effectively_never": False}
    for m, event, found in (
        (method, HMODE, h_meta),
        ("lmode_frames", "low_confinement_mode", l_meta),
    ):
        path = suggestions.table_path(runs, event, m, frames.VERSION)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".meta.json").write_text(json.dumps(found))
    manifest = build.build(runs, tmp_path / "paper")
    lines = (tmp_path / "paper" / "table_datasets.tex").read_text().splitlines()
    assert lines[7] == "H-mode & 4 & 0 & 3 & 2.5 & 2 & 1 & 1 & -- & 4 & 2 \\\\"
    assert "H-mode (≈ always)" not in lines[0]
    assert "frames_table_meta_hmode_frames" in manifest["inputs"]
    assert not [k for k in manifest["inputs"] if "lmode" in k]


def test_labels_alone_count_the_owners_review(runs, tmp_path):
    for method in ("elm_frames", "sawtooth_frames"):
        frames.shots_file(runs, method).unlink()
    frames.summary_file(runs, "elm_frames").unlink()
    manifest = build.build(runs, tmp_path / "paper")
    lines = (tmp_path / "paper" / "table_datasets.tex").read_text().splitlines()
    assert lines[8] == "ELMing & -- & 3 & -- & -- & -- & -- & -- & -- & -- & -- \\\\"
    assert lines[9] == "sawtooth & \\multicolumn{10}{c}{coming} \\\\"
    entries = _theirs(manifest["partial"]["table_datasets"])
    reasons = [(e["phenomenon"], e["reason"]) for e in entries]
    assert reasons == [
        (NTM, build.FRAMES_COMING),
        (HMODE, build.FRAMES_COMING),
        (ELM, build.NO_FRAMES_SPLIT),
        (ELM, build.NO_FRAMES_SUMMARY),
        (SAW, build.FRAMES_COMING),
    ], "no meta entry without a split: there is nothing to count it beside"


def test_no_frame_model_bar_is_read(runs, tmp_path):
    """The coverage's marks are the suggestion table's meta's, never the frame
    model's `evaluation.json`: fig_scores alone reads that (F5), and a broken
    one fails the build."""
    model = frames.model_dir(runs, "elm_frames")
    record = model / "evaluation.json"
    record.parent.mkdir(parents=True)
    record.write_text("not json: the coverage must never read it")
    snap = Snapshot(tmp_path)
    counts, _ = build.frame_coverage(runs, snap)
    assert counts[ELM].bar_met is False, "ELM_TABLE_META's E1"
    assert not [p for p, _ in snap.pinned.values() if p.is_relative_to(model)]
    with pytest.raises(json.JSONDecodeError):
        build.build(runs, tmp_path / "paper")


def test_the_split_file_is_read_as_the_frame_split(runs):
    table = pd.read_csv(frames.shots_file(runs, "elm_frames"))
    assert list(table.columns) == list(frame_shots.COLUMNS)
    assert coverage.frame_split(table) == {
        301: "train",
        302: "test",
        303: "train",
        304: "train",
        305: "train",
        306: "val",
        307: "test",
    }
    with pytest.raises(ValueError):  # five columns: not an AE split.csv
        train.read_split(frames.shots_file(runs, "elm_frames"))
