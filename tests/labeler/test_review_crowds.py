"""Annotation resolution survives editing, public tables and saved versions."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from labeler.events.catalog.check import check_table
from labeler.events.databases import DatabaseError
from labeler.events.interval_tables import (
    WITH_ATTRS,
    project_intervals,
    validate_intervals,
)
from labeler.events.review import labels, versions

from .test_review_page import _node, needs_node


@pytest.mark.parametrize("reverse", [False, True])
def test_individual_spans_preserve_the_crowd_envelope_in_either_input_order(reverse):
    spans, flags = [[10, 80, 1], [30, 50, 1]], [1, 0]
    if reverse:
        spans.reverse()
        flags.reverse()
    label = labels.normalise([0, 100], spans, iscrowd=flags)
    assert label.as_json() == {
        "window": [0, 100],
        "intervals": [[10, 80, 1], [30, 50, 1]],
        "iscrowd": [1, 0],
    }


def test_overlapping_rows_mark_only_gaps_outside_both_annotation_lanes():
    label = labels.Label(
        (0, 100), ((10, 80, 1), (30, 50, 1), (60, 70, 1)), (1, 0, 0),
    )
    assert label.rows(11) == [
        [11, 0, 0, 10, "", ""],
        [11, 1, 10, 80, "", '{"iscrowd": 1}'],
        [11, 1, 30, 50, "", '{"iscrowd": 0}'],
        [11, 1, 60, 70, "", '{"iscrowd": 0}'],
        [11, 0, 80, 100, "", ""],
    ]


def test_overlapping_annotations_survive_csv_and_history_round_trip(tmp_path):
    label = labels.Label((0, 100), ((10, 80, 1), (30, 50, 1)), (1, 0))
    labels.save(tmp_path, 11, label, source=None)
    assert labels.read_saved(tmp_path)[11] == label
    assert versions.shot_versions(tmp_path, 11)[0]["intervals"] == [
        [10, 80, 1], [30, 50, 1],
    ]
    assert versions.shot_versions(tmp_path, 11)[0]["iscrowd"] == [1, 0]


def test_an_individual_eraser_leaves_the_overlapping_crowd_intact():
    label = labels.normalise(
        [0, 100], [[10, 80, 1], [30, 50, 1], [35, 40, 0]], iscrowd=[1, 0, 0],
    )
    assert label.intervals == ((10, 80, 1), (30, 35, 1), (40, 50, 1))
    assert label.iscrowd == (1, 0, 0)


def test_erasing_all_crowds_does_not_erase_the_remaining_unspecified_span():
    label = labels.normalise(
        [0, 100], [[0, 100, 1], [20, 30, 1], [20, 30, 0]],
        iscrowd=[None, 1, 1],
    )
    assert label.as_json() == {"window": [0, 100], "intervals": [[0, 100, 1]]}


@needs_node
def test_an_unscoped_eraser_cuts_both_overlapping_annotation_lanes():
    case = {
        "window": [0, 100],
        "intervals": [[10, 80, 1], [30, 50, 1], [35, 40, 0]],
        "iscrowd": [1, 0, None],
    }
    expected = {
        "window": [0, 100],
        "intervals": [[10, 35, 1], [30, 35, 1], [40, 50, 1], [40, 80, 1]],
        "iscrowd": [1, 0, 0, 1],
    }
    assert labels.normalise(
        case["window"], case["intervals"], iscrowd=case["iscrowd"],
    ).as_json() == expected
    assert _node(
        "m.normalise(input.window, input.intervals, [1], input.iscrowd)", case,
    ) == expected


def test_catalog_individual_absence_can_overlap_a_crowd():
    frame = pd.DataFrame([
        [11, 0, 0, 100, None, '{"iscrowd": 0}'],
        [11, 1, 10, 80, None, '{"iscrowd": 1}'],
    ], columns=WITH_ATTRS)
    assert check_table(frame, "edge_localized_mode") == []


def test_catalog_accepts_individuals_inside_a_crowd_with_shared_absent_gaps():
    frame = pd.DataFrame([
        [11, 0, 0, 10, None, ""],
        [11, 1, 10, 80, None, '{"iscrowd": 1}'],
        [11, 1, 30, 50, None, '{"iscrowd": 0}'],
        [11, 0, 80, 100, None, ""],
    ], columns=WITH_ATTRS)
    assert check_table(frame, "edge_localized_mode") == []


@pytest.mark.parametrize("flag", [0, 1])
def test_catalog_still_refuses_overlapping_annotations_in_the_same_lane(flag):
    frame = pd.DataFrame([
        [11, 1, 0, 80, None, json.dumps({"iscrowd": flag})],
        [11, 1, 30, 100, None, json.dumps({"iscrowd": flag})],
    ], columns=WITH_ATTRS)
    assert any(f.check == "tiling" for f in check_table(frame, "edge_localized_mode"))


@needs_node
def test_history_detects_changes_to_a_crowd_underneath_an_individual():
    before = {
        "window": [0, 100], "intervals": [[10, 80, 1], [30, 70, 1]],
        "iscrowd": [1, 0],
    }
    after = {
        "window": [0, 100], "intervals": [[10, 60, 1], [30, 70, 1]],
        "iscrowd": [1, 0],
    }
    assert _node("m.diffRuns(input[0], input[1])", [before, after]) == [[60, 80]]


@needs_node
def test_the_pointer_only_selects_spans_in_its_annotation_lane():
    label = {
        "window": [0, 100], "intervals": [[10, 80, 1], [30, 50, 1]],
        "iscrowd": [1, 0],
    }
    found = _node(
        "input.points.map(([x, lane]) => "
        "m.hitTest(input.label, x, 5, 40, t => t, lane))",
        {"label": label, "points": [[40, 0], [40, 1], [20, 0], [30, 1]]},
    )
    assert found == [
        {"kind": "move", "index": 1},
        {"kind": "move", "index": 0},
        {"kind": "new"},
        {"kind": "move", "index": 0},
    ]


def test_touching_individual_annotations_keep_separate_boundaries():
    label = labels.normalise(
        [0, 100], [[10, 20, 1], [20, 30, 1], [30, 60, 1]],
        iscrowd=[0, 0, 1],
    )
    assert label.as_json() == {
        "window": [0, 100],
        "intervals": [[10, 20, 1], [20, 30, 1], [30, 60, 1]],
        "iscrowd": [0, 0, 1],
    }
    assert label.rows(11) == [
        [11, 0, 0, 10, "", ""],
        [11, 1, 10, 20, "", '{"iscrowd": 0}'],
        [11, 1, 20, 30, "", '{"iscrowd": 0}'],
        [11, 1, 30, 60, "", '{"iscrowd": 1}'],
        [11, 0, 60, 100, "", ""],
    ]


def test_crowd_flags_follow_painting_and_zero_length_removal():
    label = labels.normalise(
        [0, 100], [[10, 80, 1], [30, 50, 2], [60, 70, 0], [90, 90.2, 1]],
        iscrowd=[1, 0, None, 0],
    )
    assert label.intervals == (
        (10, 60, 1), (30, 50, 2), (70, 80, 1),
    )
    assert label.iscrowd == (1, 0, 1)


def test_legacy_resolution_stays_unspecified_in_a_mixed_label():
    label = labels.normalise(
        [0, 100], [[10, 30, 1], [50, 80, 1]], iscrowd=[None, 1],
    )
    assert label.as_json()["iscrowd"] == [None, 1]
    assert label.rows(11)[1][-1] == ""
    legacy = labels.normalise([0, 100], [[10, 20, 1], [20, 30, 1]])
    assert legacy.as_json() == {"window": [0, 100], "intervals": [[10, 30, 1]]}


@pytest.mark.parametrize("flags", [[1, 0], [2], ["1"], [0.5], [{}]])
def test_malformed_resolution_is_rejected(flags):
    with pytest.raises(ValueError, match="iscrowd"):
        labels.normalise([0, 100], [[10, 30, 1]], iscrowd=flags)


def test_saved_crowds_reload_and_appear_in_history_without_reclassifying_legacy(
    tmp_path,
):
    label = labels.normalise(
        [0, 100], [[10, 20, 1], [20, 30, 1], [40, 80, 1]],
        iscrowd=[0, 0, 1],
    )
    labels.save(tmp_path, 11, labels.normalise([0, 100], [[10, 20, 1]]), source=None)
    labels.save(tmp_path, 12, label, source=None)
    assert labels.read_saved(tmp_path)[12] == label
    assert labels.read_saved(tmp_path)[11].iscrowd == ()
    assert versions.shot_versions(tmp_path, 12)[0]["iscrowd"] == [0, 0, 1]
    # Crowd metadata is editable on the selected shot and remains with its spans.
    changed = labels.normalise([0, 100], [[40, 80, 1]], iscrowd=[0])
    labels.save(tmp_path, 12, changed, source=None)
    assert labels.read_saved(tmp_path)[12].iscrowd == (0,)
    assert [v["iscrowd"] for v in versions.shot_versions(tmp_path, 12)] == [
        [0, 0, 1], [0],
    ]


@pytest.mark.parametrize("intervals", [[[10, 30, 1]], []])
def test_a_legacy_client_cannot_silently_erase_stored_resolution(tmp_path, intervals):
    label = labels.normalise([0, 100], [[10, 30, 1]], iscrowd=[1])
    labels.save(tmp_path, 11, label, source=None)
    before = labels.labels_path(tmp_path).read_bytes()
    with pytest.raises(labels.SaveRefused, match="iscrowd"):
        labels.save(tmp_path, 11, labels.normalise([0, 100], intervals),
                    source=None)
    assert labels.labels_path(tmp_path).read_bytes() == before


@pytest.mark.parametrize("flag", [2, "1", 1.5, None])
def test_csv_crowd_metadata_requires_a_binary_flag(flag):
    frame = pd.DataFrame(
        [[11, 1, 10, 30, None, json.dumps({"iscrowd": flag})]],
        columns=WITH_ATTRS,
    )
    with pytest.raises(DatabaseError, match="iscrowd"):
        validate_intervals(frame)


def test_projection_keeps_crowd_resolution_for_model_consumers():
    events = pd.DataFrame({
        "shot": [11, 11], "t0_s": [0.01, 0.05], "t1_s": [0.02, 0.09],
        "confidence": [1.0, 0.8], "attrs": ['{"iscrowd": 0}', '{"iscrowd": 1}'],
    })
    found = project_intervals(events)
    assert found["attrs"].tolist() == ['{"iscrowd": 0}', '{"iscrowd": 1}']
    assert found.t_start.tolist() == [10, 50]


@pytest.mark.parametrize("attrs", ['[]', '{"iscrowd": 0, "iscrowd": 1}'])
def test_projection_refuses_malformed_annotation_attributes(attrs):
    events = pd.DataFrame({
        "shot": [11], "t0_s": [0.01], "t1_s": [0.02],
        "confidence": [1.0], "attrs": [attrs],
    })
    with pytest.raises(DatabaseError, match="attrs"):
        project_intervals(events)


@needs_node
def test_page_and_server_keep_the_same_annotation_boundaries_and_flags():
    cases = [
        {"window": [0, 100], "intervals": [[10, 20, 1], [20, 30, 1], [30, 60, 1]],
         "iscrowd": [0, 0, 1]},
        {"window": [0, 100], "intervals": [[10, 80, 1], [30, 50, 2], [60, 70, 0]],
         "iscrowd": [1, 0, None]},
        {"window": [0, 100], "intervals": [[10, 30, 1], [50, 80, 1]],
         "iscrowd": [None, 1]},
    ]
    expected = [
        {"window": [0, 100], "intervals": [[10, 20, 1], [20, 30, 1], [30, 60, 1]],
         "iscrowd": [0, 0, 1]},
        {"window": [0, 100], "intervals": [[10, 60, 1], [30, 50, 2], [70, 80, 1]],
         "iscrowd": [1, 0, 1]},
        {"window": [0, 100], "intervals": [[10, 30, 1], [50, 80, 1]],
         "iscrowd": [None, 1]},
    ]
    page = _node(
        "input.map(c => m.normalise(c.window, c.intervals, [1, 2], c.iscrowd))", cases,
    )
    server = [labels.normalise(c["window"], c["intervals"], iscrowd=c["iscrowd"])
              .as_json() for c in cases]
    assert page == server == expected


@needs_node
def test_resolution_only_changes_are_visible_in_history():
    source = {"window": [0, 100], "intervals": [[10, 30, 1]], "iscrowd": [1]}
    edited = {"window": [0, 100], "intervals": [[10, 30, 1]], "iscrowd": [0]}
    assert _node("m.diffRuns(input[0], input[1])", [source, edited]) == [[10, 30]]
