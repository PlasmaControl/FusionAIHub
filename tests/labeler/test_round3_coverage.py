"""The paper's coverage of the frame-model phenomena: the owner's review, the
legacy tables' labelled shots, the frame models' splits and their suggestions."""

from __future__ import annotations

import json
from itertools import pairwise

import pandas as pd
import pytest

from labeler import frames
from labeler.ae.xpower import train
from labeler.events.review import labels
from labeler.paper import AE, COMING, build, coverage

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
)  # 3 reviewed, 2 with a present span, 1.2 s present; 309 saved after the split
ELM_SPLIT = (
    "shot,split,positive,roster\n"
    "301,owner,1,1\n302,owner,0,1\n"
    "303,train,1,0\n304,train,0,0\n305,train,1,0\n306,val,1,0\n307,test,0,0\n"
)
SAW_SPLIT = (
    "shot,split,positive,roster\n401,train,1,1\n402,train,1,1\n403,val,1,1\n"
    "404,test,0,1\n"
)
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
    return coverage.frame_counts(labels.read_saved(event), split, summary, legacy=576)


def _saw() -> coverage.Counts:
    split = {401: "train", 402: "train", 403: "val", 404: "test"}
    return coverage.frame_counts({}, split, None)


def test_a_frame_models_counts(tmp_path):
    elm = _elm(tmp_path)
    assert (elm.reviewed, elm.positive, elm.present_s) == (3, 2, 1.2)
    assert elm.split == {"train": 3, "val": 1, "test": 1, "owner": 2}, "all its shots"
    assert elm.unsplit == 1, "309 was saved after the split"
    assert (elm.legacy, elm.frame, elm.cross_validated) == (576, True, False)
    assert (elm.suggested, elm.suggested_positive) == (4, 2)
    saw = _saw()
    assert (saw.reviewed, saw.positive, saw.unsplit, saw.legacy) == (0, 0, 0, None)
    assert saw.split == {"train": 2, "val": 1, "test": 1, "owner": 0}
    assert saw.by_year is None and saw.suggested is None
    bare = coverage.frame_counts({}, None, None)
    assert (bare.split, bare.unsplit, bare.frame) == (None, None, True)
    assert AE_COUNTS.legacy is None and AE_COUNTS.frame is False, "AE's defaults"


def test_a_split_outside_the_four_is_refused():
    table = pd.DataFrame({"shot": [1, 2], "split": ["train", "valid"]})
    with pytest.raises(ValueError, match="valid"):
        coverage.frame_split(table)


def test_the_frame_sources_are_the_four_paper_phenomena():
    assert list(coverage.FRAME_SOURCES) == [NTM, HMODE, ELM, SAW]
    assert "disruption" not in coverage.FRAME_SOURCES
    assert {s.method for s in coverage.FRAME_SOURCES.values()} <= set(frames.SPECS)
    assert coverage.FRAME_SOURCES[SAW].legacy is None
    assert "Jalalvand" not in json.dumps(
        {k: [s.legacy, s.also] for k, s in coverage.FRAME_SOURCES.items()}
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
    assert len(fig.axes) == 12
    shots = fig.axes[0]
    assert sum(t == COMING for t in _texts(shots)) == 2, "NTM and H-mode"
    assert _texts(shots).count(coverage.NONE_REVIEWED) == 1, "sawteeth: a true zero"
    assert _widths(shots) == [4, 3, 2, 2], "reviewed AE, ELM; positive AE, ELM"
    ntm_split, hmode_split, elm_split, saw_split = fig.axes[4:8]
    for ax in (ntm_split, hmode_split):
        assert _texts(ax) == [COMING] and not ax.patches
    assert _widths(elm_split) == [576, 3, 1, 1, 2]
    assert _ticks(elm_split) == ["legacy", "train", "val", "test", "owner"]
    assert elm_split.get_title() == "ELMing: model split\nand Hiro's table"
    assert "576" in _texts(elm_split)
    assert _widths(saw_split) == [2, 1, 1, 0]
    assert _ticks(saw_split) == ["train", "val", "test", "owner"]
    assert saw_split.get_title() == "sawteeth: model split"
    ntm_years, _, elm_years, saw_years = fig.axes[8:12]
    assert _texts(ntm_years) == [COMING]
    assert _widths(elm_years) == [2, 1, 1, 1, 1, 0]
    assert _ticks(elm_years) == ["2024", "2025", "?"]
    assert elm_years.get_title() == "ELMing suggestions\nby year"
    assert elm_years.get_legend() is None, "the frame rows share one key"
    assert _texts(saw_years) == [coverage.NOT_RUN] and not saw_years.patches
    legend, key = fig.legends
    assert [t.get_text() for t in legend.get_texts()] == [
        "reviewed",
        "with a present span",
    ]
    assert [t.get_text() for t in key.get_texts()] == list(coverage.FRAME_SUGGESTED)
    assert tree.small_text(fig) == []
    for ax in (elm_split, saw_split, elm_years):
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


def _population(legacy: int | None) -> coverage.Counts:
    split = {"train": 20400, "val": 4300, "test": 4350, "owner": 150}
    return coverage.Counts(
        900, 610, 812.4, split, BY_YEAR, unsplit=12, legacy=legacy, frame=True
    )


def test_at_population_sizes_the_keys_cover_no_bar_and_no_tick(tmp_path):
    legacies = (14210, 31876, 28521, None)  # the tearing archive's to Hiro's
    counts = {AE: AE_COUNTS} | {
        category: _population(legacy)
        for category, legacy in zip(coverage.FRAME_SOURCES, legacies, strict=True)
    }
    fig = coverage.draw_coverage(counts, tmp_path / "fig_coverage")
    assert all(ax.get_legend() is None for ax in fig.axes[4:12])
    fig.draw_without_rendering()
    keys = [legend.get_window_extent() for legend in fig.legends]
    assert len(keys) == 2 and not keys[0].overlaps(keys[1])
    ticks = [t.get_window_extent() for ax in fig.axes for t in _tick_labels(ax)]
    bars = [bar.get_window_extent() for ax in fig.axes for bar in ax.patches]
    panels = [ax.get_window_extent() for ax in fig.axes]
    assert len(bars) > 60
    for key in keys:
        assert not any(key.overlaps(box) for box in ticks + bars + panels)
    for ax in fig.axes[8:12]:
        assert _ticks(ax) == ["2021", "2022", "2023", "2024", "2025", "?"]
        assert _apart(fig, ax), ax.get_title()
    assert "28,521" in _texts(fig.axes[6])
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


def test_ae_alone_keeps_its_one_row(tmp_path):
    fig = coverage.draw_coverage({AE: AE_COUNTS}, tmp_path / "fig_coverage")
    assert len(fig.axes) == 4
    assert tuple(fig.get_size_inches()) == (coverage.PAGE_IN, 2.3)
    assert coverage.NONE_REVIEWED not in _texts(fig.axes[0])


def test_the_table_with_the_frame_phenomena(tmp_path):
    counts = {AE: AE_COUNTS, ELM: _elm(tmp_path), SAW: _saw()}
    lines = coverage.table_datasets(counts).splitlines()
    assert lines[0] == (
        "% Shots per phenomenon: reviewed by a person, with any present span, the "
        "model's split of the reviewed shots and those in no split (reviewed = "
        "train + val + test + no split), the extension's suggestions (not labels), "
        "and the shots a legacy human table labels (Legacy labelled, never counted "
        "as reviewed); -- where that run has not happened; for ELMing and "
        "sawteeth, the split is the frame model's, of its own shots (the owner's "
        "saved shots held out of it as its owner split), not of the reviewed "
        "ones, and No split counts their reviewed shots in none of its splits; "
        "legacy tables: ELMing: Hiro's table, sawteeth: none"
    )
    assert lines[1] == "\\begin{tabular}{lcccccccc}"
    assert lines[3] == (
        "Phenomenon & Reviewed & Positive & Present (s) & Train / val / test "
        "& No split & Suggested & Suggested positive & Legacy labelled \\\\"
    )
    assert lines[5:10] == [
        "AE & 4 & 2 & 0.9 & 1 / 1 / 1 & 1 & 4 & 2 & -- \\\\",
        "NTM & \\multicolumn{8}{c}{coming} \\\\",
        "H-mode & \\multicolumn{8}{c}{coming} \\\\",
        "ELMing & 3 & 2 & 1.2 & 3 / 1 / 1 & 1 & 4 & 2 & 576 \\\\",
        "sawteeth & 0 & 0 & 0.0 & 2 / 1 / 1 & 0 & -- & -- & -- \\\\",
    ]


def test_h_mode_names_l_mode_with_its_table(tmp_path):
    hmode = coverage.frame_counts({}, {1: "train"}, None, legacy=428)
    counts = {AE: AE_COUNTS, HMODE: hmode}
    lines = coverage.table_datasets(counts).splitlines()
    assert lines[0].endswith(
        "; for H-mode, the split is the frame model's, of its own shots (the "
        "owner's saved shots held out of it as its owner split), not of the "
        "reviewed ones, and No split counts their reviewed shots in none of its "
        "splits; legacy tables: H-mode (and L-mode): Jalal Butt's table"
    )
    assert lines[7] == "H-mode & 0 & 0 & 0.0 & 1 / 0 / 0 & 0 & -- & -- & 428 \\\\"
    fig = coverage.draw_coverage(counts, tmp_path / "fig")
    assert fig.axes[5].get_title() == "H-mode: model split\nand Jalal Butt's table"


@pytest.fixture
def runs(tmp_path, monkeypatch):
    """test_paper_build's tree, with ELM's four inputs and sawteeth's split."""
    paths = ae_tree.build(tmp_path, {101: "train", 102: "valid", 103: "valid"})
    models = ae_tree.chosen(paths, {101: "train", 102: "test", 103: "test"})
    (models / "evaluation.json").write_text(json.dumps(tree.ae_evaluation()))
    tree.record_labels(paths)
    ae_tree.env(monkeypatch, paths)
    live = labels.labels_path(paths.label_tables / ELM)
    live.parent.mkdir(parents=True)
    live.write_text(ELM_LABELS)
    for method, text in (("elm_frames", ELM_SPLIT), ("sawtooth_frames", SAW_SPLIT)):
        frames.shots_file(paths, method).parent.mkdir(parents=True, exist_ok=True)
        frames.shots_file(paths, method).write_text(text)
    meta = {"method": "elm_frames", "labelled_shots": 576, "positive_shots": 443}
    frames.shots_meta_file(paths, "elm_frames").write_text(json.dumps(meta))
    frames.summary_file(paths, "elm_frames").parent.mkdir(parents=True)
    frames.summary_file(paths, "elm_frames").write_text(SUMMARY)
    return paths


def _theirs(entries: list[dict]) -> list[dict]:
    return [e for e in entries if "phenomenon" in e]


def test_the_build_counts_the_frame_phenomena(runs, tmp_path):
    manifest = build.build(runs, tmp_path / "paper")
    lines = (tmp_path / "paper" / "table_datasets.tex").read_text().splitlines()
    assert lines[5] == "AE & 3 & 3 & 1.8 & 1 / 0 / 2 & 0 & -- & -- & -- \\\\"
    assert lines[6:10] == [
        "NTM & \\multicolumn{8}{c}{coming} \\\\",
        "H-mode & \\multicolumn{8}{c}{coming} \\\\",
        "ELMing & 3 & 2 & 1.2 & 3 / 1 / 1 & 1 & 4 & 2 & 576 \\\\",
        "sawteeth & 0 & 0 & 0.0 & 2 / 1 / 1 & 0 & -- & -- & -- \\\\",
    ]

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
        "frames_split_sawtooth_frames",
    }
    assert manifest["consistent"] is True


def test_a_split_without_its_meta_leaves_the_legacy_count_out(runs, tmp_path):
    meta = frames.shots_meta_file(runs, "elm_frames")
    meta.unlink()
    manifest = build.build(runs, tmp_path / "paper")
    lines = (tmp_path / "paper" / "table_datasets.tex").read_text().splitlines()
    assert lines[8] == "ELMing & 3 & 2 & 1.2 & 3 / 1 / 1 & 1 & 4 & 2 & -- \\\\"
    missing = {
        "reason": build.NO_FRAMES_META,
        "phenomenon": ELM,
        "missing": [str(meta)],
    }
    assert missing in manifest["partial"]["fig_coverage"]
    assert "frames_meta_elm_frames" not in manifest["inputs"]


def test_labels_alone_count_the_owners_review(runs, tmp_path):
    for method in ("elm_frames", "sawtooth_frames"):
        frames.shots_file(runs, method).unlink()
    frames.summary_file(runs, "elm_frames").unlink()
    manifest = build.build(runs, tmp_path / "paper")
    lines = (tmp_path / "paper" / "table_datasets.tex").read_text().splitlines()
    assert lines[8] == "ELMing & 3 & 2 & 1.2 & -- & -- & -- & -- & -- \\\\"
    assert lines[9] == "sawteeth & \\multicolumn{8}{c}{coming} \\\\"
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
    record = frames.model_dir(runs, "elm_frames") / "evaluation.json"
    record.parent.mkdir(parents=True)
    record.write_text("not json: the coverage must never read it")
    manifest = build.build(runs, tmp_path / "paper")
    assert "fig_coverage" in manifest["products"]
    assert all(
        not v["path"].startswith(str(frames.model_dir(runs, "elm_frames")))
        for v in manifest["inputs"].values()
    )


def test_the_split_file_is_read_as_the_frame_split(runs):
    table = pd.read_csv(frames.shots_file(runs, "elm_frames"))
    assert coverage.frame_split(table) == {
        301: "owner",
        302: "owner",
        303: "train",
        304: "train",
        305: "train",
        306: "val",
        307: "test",
    }
    with pytest.raises(ValueError):  # four columns: not an AE split.csv
        train.read_split(frames.shots_file(runs, "elm_frames"))
