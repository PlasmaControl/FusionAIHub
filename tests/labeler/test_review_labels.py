"""One label per shot: normalising, the source table, saving, states, the queue."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pandas as pd
import pytest

from labeler.events.catalog.check import states
from labeler.events.catalog.states import NOT_OBSERVABLE, PHENOMENA, STATE_NAMES
from labeler.events.interval_tables import WITH_ATTRS, attrs_text, validate_intervals
from labeler.events.review import labels
from labeler.events.review.labels import Label, normalise

SOURCE = (
    "shot,category,t_start,t_end,confidence\n"
    "170815,0,0,100,\n"
    "170815,1,100,300,\n"
    "170815,0,300,2000,\n"
    "178642,0,0,2000,\n"
    "178642,1,500,500.4,\n"
)


@pytest.fixture
def event_dir(tmp_path):
    directory = tmp_path / "alfven_eigenmode"
    (directory / "format").mkdir(parents=True)
    (directory / "format" / "alfven_eigenmode_format_2026_v1.csv").write_text(SOURCE)
    return directory


def test_edges_snap_to_whole_milliseconds_half_up():
    assert normalise((0, 100), [(10.4, 20.5, 1)], grid_ms=1).intervals == (
        (10, 21, 1),
    )


def test_overlapping_and_touching_spans_merge():
    label = normalise((0, 100), [(10, 30, 1), (20, 40, 1), (50, 55, 1), (55, 60, 1)])
    assert label.intervals == ((10, 40, 1), (50, 60, 1))


def test_a_later_span_paints_over_an_earlier_one():
    label = normalise((0, 100), [(10, 50, 1), (30, 40, 2)])
    assert label.intervals == ((10, 30, 1), (30, 40, 2), (40, 50, 1))


def test_a_category_zero_span_erases():
    label = normalise((0, 100), [(10, 50, 1), (20, 30, 0)])
    assert label.intervals == ((10, 20, 1), (30, 50, 1))


def test_the_window_grows_to_cover_every_span():
    assert normalise((100, 200), [(50, 120, 1), (190, 260, 1)]).window == (50, 260)


def test_a_zero_length_span_is_dropped():
    assert normalise((0, 100), [(10, 10.2, 1)], grid_ms=1).intervals == ()


def test_the_default_grid_rounds_edges_and_expands_the_window():
    assert normalise((3, 96), [(14.9, 25, 1)]) == Label(
        (0, 100), ((10, 30, 1),),
    )


def test_short_individual_elms_keep_their_midpoint_frame_and_later_spans_win():
    label = normalise(
        (0, 100), [(21, 24, 1), (22, 23, 2), (35, 35, 1)], iscrowd=[0, 0, 0],
    )
    assert label == Label((0, 100), ((20, 30, 2),), (0,))
    assert normalise((0, 100), [(21, 24, 1)], iscrowd=[0]) == Label(
        (0, 100), ((20, 30, 1),), (0,),
    )


def test_short_individuals_and_crowds_keep_separate_lanes_on_the_grid():
    label = normalise(
        (0, 100), [(21, 24, 1), (22, 23, 2)], iscrowd=[1, 0],
    )
    assert label == Label((0, 100), ((20, 30, 2), (20, 30, 1)), (0, 1))


def test_the_window_limit_is_checked_after_snapping():
    assert normalise((1, 20004), []).window == (0, 20000)
    with pytest.raises(ValueError, match="not 1 to 20000 ms"):
        normalise((1, 20006), [])
    assert normalise((1, 20001), [], grid_ms=1).window == (1, 20001)


@pytest.mark.parametrize(
    "window, intervals, known, match",
    [
        ((0, 100), [(30, 20, 1)], None, "runs backwards"),
        ((0, 100), [(10, 20, 7)], {1}, "category 7"),
        ((100, 100), [], None, "not 1 to 20000 ms"),
        ((0, 20006), [], None, "not 1 to 20000 ms"),
    ],
)
def test_normalise_refuses(window, intervals, known, match):
    with pytest.raises(ValueError, match=match):
        normalise(window, intervals, known=known)


def test_rows_tile_the_window_with_category_zero_gaps():
    label = normalise((0, 100), [(10, 20, 1), (50, 60, 1)])
    assert label.rows(170815) == [
        [170815, 0, 0, 10, ""],
        [170815, 1, 10, 20, ""],
        [170815, 0, 20, 50, ""],
        [170815, 1, 50, 60, ""],
        [170815, 0, 60, 100, ""],
    ]
    assert Label((0, 50)).rows(1) == [[1, 0, 0, 50, ""]]


def test_the_source_is_read_per_shot_and_point_events_are_ignored(event_dir):
    assert labels.read_source(event_dir) == {
        170815: Label((0, 2000), ((100, 300, 1),)),
        178642: Label((0, 2000)),
    }


def test_readers_keep_whole_ms_and_the_page_reads_on_the_grid(event_dir):
    """A model's frozen copy (`models/*/review/labels.csv`, 1 ms) reads as trained on."""
    path = labels.labels_path(event_dir)
    path.parent.mkdir()
    path.write_text(
        "shot,category,t_start,t_end,confidence\n"
        "7,0,3,21,\n7,1,21,24,\n7,0,24,96,\n7,1,55,55,\n"
    )
    source = event_dir / "format" / "alfven_eigenmode_format_2027_v1.csv"
    source.write_bytes(path.read_bytes())
    whole = Label((3, 96), ((21, 24, 1),))
    snapped = Label((0, 100), ((20, 30, 1),))
    spans = Label((0, 100), ((20, 30, 1),), (1,))  # AE labels without resolution
    assert labels.read_labels(path)[7] == whole
    assert labels.read_saved(event_dir)[7] == whole
    assert labels.read_source(event_dir)[7] == whole
    assert labels.read_saved(event_dir, grid_ms=10)[7] == snapped
    assert labels.page_label("alfven_eigenmode", whole) == spans
    assert labels.shot_labels(event_dir, 7)["saved"] == spans.as_json()
    assert labels.read_labels(path, grid_ms=10)[7] == snapped
    assert labels.read_labels(path)[7] == whole, "the cache distinguishes grids"


def test_ae_and_confinement_labels_without_resolution_are_spans():
    bare = Label((0, 2000), ((100, 300, 1), (800, 1000, 2)))
    spans = Label((0, 2000), ((100, 300, 1), (800, 1000, 2)), (1, 1))
    for event in ("alfven_eigenmode", "alfven_eigenmode.tae", "confinement"):
        assert labels.page_label(event, bare) == spans
    for event in ("edge_localized_mode", "minimum_safety_factor"):
        assert labels.page_label(event, bare) == bare
    # A table that sets any flag keeps its own; an empty or missing label stays so.
    flagged = Label((0, 2000), ((100, 300, 1), (800, 1000, 2)), (0, None))
    assert labels.page_label("alfven_eigenmode", flagged) == flagged
    assert labels.page_label("alfven_eigenmode", Label((0, 2000))) == Label((0, 2000))
    assert labels.page_label("confinement", None) is None


def test_rwm_export_reader_round_trip_keeps_unassessed_time_and_evidence(tmp_path):
    from labeler.rwm.labels import window_rows

    tiers = {
        0: "assumed_absent", 1: "onset_point_minimal",
        2: "onset_window_uncertain", 4: "unassessed",
    }
    exported = pd.DataFrame(
        [
            [shot, category, a, b, "", attrs_text({
                "evidence_tier": tiers[category], "coverage_verified": False,
            })]
            for shot, category, a, b in window_rows(
                156785, [856.0], (444.0, 4997.0), assumed_absent=True
            )
        ],
        columns=WITH_ATTRS,
    )
    path = tmp_path / "rwm_windows.csv"
    exported.to_csv(path, index=False)
    label = labels.read_labels(path)[156785]
    restored_rows = label.rows(156785)
    assert [row[:4] for row in restored_rows] == [
        [156785, 0, 444, 756], [156785, 4, 756, 836],
        [156785, 2, 836, 856], [156785, 1, 856, 866],
        [156785, 4, 866, 4997],
    ]
    restored = pd.DataFrame(restored_rows, columns=WITH_ATTRS)
    assert restored["attrs"].tolist() == exported["attrs"].tolist()
    # Probe both half-open onset windows and the unassessed time after them.
    for t, expected in [
        (500, 0), (800, 4), (836, 2), (840, 2), (855.9, 2),
        (856, 1), (860, 1), (865.9, 1), (866, 4), (900, 4), (4996, 4),
    ]:
        covering = restored[(restored.t_start <= t) & (t < restored.t_end)]
        assert covering.category.tolist() == [expected]
    assert label.as_json()["attrs"] == [
        {"evidence_tier": "assumed_absent", "coverage_verified": False},
        {"evidence_tier": "unassessed", "coverage_verified": False},
        {"evidence_tier": "onset_window_uncertain", "coverage_verified": False},
        {"evidence_tier": "onset_point_minimal", "coverage_verified": False},
        {"evidence_tier": "unassessed", "coverage_verified": False},
    ]


def test_reader_keeps_distinct_evidence_on_touching_absent_spans(tmp_path):
    path = tmp_path / "evidence.csv"
    attrs = [
        attrs_text({"evidence_tier": tier, "coverage_verified": verified})
        for tier, verified in [("assumed_absent", False), ("reviewed_absent", True)]
    ]
    pd.DataFrame(
        [[7, 0, 0, 10, "", attrs[0]], [7, 0, 10, 20, "", attrs[1]]],
        columns=WITH_ATTRS,
    ).to_csv(path, index=False)
    restored = labels.read_labels(path)[7]
    assert restored.intervals == ((0, 10, 0), (10, 20, 0))
    assert [row[-1] for row in restored.rows(7)] == attrs


def test_a_fractional_unassessed_span_survives_whole_ms_sampling(tmp_path):
    path = tmp_path / "evidence.csv"
    pd.DataFrame(
        [
            [7, category, a, b, "", attrs_text({
                "evidence_tier": tier, "coverage_verified": False,
            })]
            for category, a, b, tier in [
                (0, 0, 10.25, "assumed_absent"),
                (4, 10.25, 10.75, "unassessed"),
                (1, 10.75, 30, "conventional_weak"),
            ]
        ],
        columns=WITH_ATTRS,
    ).to_csv(path, index=False)
    label = labels.read_labels(path)[7]
    assert label.intervals == ((0, 10, 0), (10, 11, 4), (11, 30, 1))


def test_evidence_boundaries_snap_without_midpoint_claims_and_still_tile(tmp_path):
    path = tmp_path / "evidence.csv"
    attrs = attrs_text({"evidence_tier": "reviewed", "coverage_verified": True})
    pd.DataFrame(
        [[7, 0, 3, 21, "", attrs], [7, 4, 21, 24, "", attrs],
         [7, 1, 24, 96, "", attrs]],
        columns=WITH_ATTRS,
    ).to_csv(path, index=False)
    whole = labels.read_labels(path)[7]
    assert whole.window == (3, 96) and len(whole.intervals) == 3
    snapped = labels.read_labels(path, grid_ms=10)[7]
    assert snapped.window == (0, 100)
    assert snapped.intervals == ((0, 20, 0), (20, 100, 1))
    assert snapped.attrs == (attrs, attrs)


def test_reader_refuses_implicit_coverage_in_an_evidence_table(tmp_path):
    path = tmp_path / "evidence.csv"
    attrs = attrs_text({
        "evidence_tier": "conventional_weak", "coverage_verified": False,
    })
    pd.DataFrame(
        [[7, 1, 0, 10, "", attrs], [7, 1, 20, 30, "", attrs]], columns=WITH_ATTRS,
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="explicit.*unassessed"):
        labels.read_labels(path)


def test_tierless_review_save_refuses_an_evidence_source_before_writing(tmp_path):
    event = tmp_path / "resistive_wall_mode"
    (event / "format").mkdir(parents=True)
    path = event / "format" / "rwm_format_2026.csv"
    attrs = attrs_text({
        "evidence_tier": "conventional_weak", "coverage_verified": False,
    })
    pd.DataFrame([[7, 1, 0, 20, "", attrs]], columns=WITH_ATTRS).to_csv(
        path, index=False,
    )
    with pytest.raises(labels.SaveRefused, match="evidence"):
        labels.save(event, 7, normalise([0, 20], [[0, 20, 1]]), source=path.name)
    assert not labels.labels_path(event).exists()
    assert not labels.history_path(event).exists()


def test_the_newest_format_table_by_name_is_the_source(event_dir):
    (event_dir / "format" / "alfven_eigenmode_format_2027_v1.csv").write_text(
        "shot,category,t_start,t_end,confidence\n170815,1,0,50,\n"
    )
    assert labels.source_path(event_dir).name == "alfven_eigenmode_format_2027_v1.csv"
    assert labels.read_source(event_dir) == {170815: Label((0, 50), ((0, 50, 1),))}


def test_an_event_without_a_format_table_has_no_source(tmp_path):
    assert labels.source_path(tmp_path) is None
    assert labels.read_source(tmp_path) == {}


def test_saving_replaces_one_shots_rows_and_appends_the_history(event_dir, monkeypatch):
    monkeypatch.setattr(labels.getpass, "getuser", lambda: "nc1514")
    table = "alfven_eigenmode_format_2026_v1.csv"
    labels.save(event_dir, 170815, normalise((0, 2000), [(100, 300, 1)]), source=table)
    labels.save(event_dir, 178642, normalise((0, 2000), []), source=None)
    second = normalise((0, 2000), [(150, 400, 1)])
    entry = labels.save(event_dir, 170815, second, source=table)

    written = validate_intervals(pd.read_csv(labels.labels_path(event_dir)))
    assert written[["shot", "category", "t_start", "t_end"]].values.tolist() == [
        [170815, 0, 0, 150],
        [170815, 1, 150, 400],
        [170815, 0, 400, 2000],
        [178642, 0, 0, 2000],
    ]
    assert labels.read_saved(event_dir) == {170815: second, 178642: Label((0, 2000))}
    history = labels.read_history(event_dir)
    assert [h["shot"] for h in history] == [170815, 178642, 170815]
    assert history[-1] == entry
    assert entry["reviewer"] == "nc1514"
    assert entry["window"] == [0, 2000] and entry["intervals"] == [[150, 400, 1]]
    assert entry["source"] == table
    assert sorted(p.name for p in (event_dir / "review").iterdir()) == [
        "history.jsonl",
        "labels.csv",
        "saves",
    ]
    [mine] = (event_dir / "review" / "saves").iterdir()
    assert mine.name == f"history.{entry['writer']}.jsonl"


def test_saving_another_shot_refuses_a_ragged_table_without_changes(event_dir):
    review = event_dir / "review"
    review.mkdir()
    (review / "labels.csv").write_text(
        "shot,category,t_start,t_end,confidence\n190001,190002,1,0,100,\n"
    )
    (review / "history.jsonl").write_text('{"shot": 190001}\n')
    before = {name: (review / name).read_bytes() for name in (
        "labels.csv", "history.jsonl"
    )}
    with pytest.raises(labels.SaveRefused, match="row 2: expected 5 fields, got 6"):
        labels.save(event_dir, 190003, Label((0, 100)), source=None)
    assert {name: (review / name).read_bytes() for name in before} == before


def test_reading_a_ragged_saved_table_refuses_index_inference(event_dir):
    path = labels.labels_path(event_dir)
    path.parent.mkdir()
    path.write_text(
        "shot,category,t_start,t_end,confidence\n190001,190002,1,0,100,\n"
    )
    with pytest.raises(pd.errors.ParserError, match="row 2: expected 5 fields, got 6"):
        labels.read_saved(event_dir)


def test_states():
    source = normalise((0, 2000), [(100, 300, 1)])
    assert labels.state(None, source) == "unreviewed"
    assert labels.state(source, source) == "confirmed"
    assert labels.state(normalise((0, 2000), [(100, 310, 1)]), source) == "changed"
    assert labels.state(normalise((0, 2100), [(100, 300, 1)]), source) == "changed"
    assert labels.state(Label((0, 50)), None) == "confirmed"
    assert labels.state(normalise((0, 50), [(0, 10, 1)]), None) == "changed"


def _shots(*states):
    return [{"shot": shot, "state": state} for shot, state in zip([1, 2, 3, 4], states)]


def test_resume_is_the_first_unreviewed_shot_after_the_newest_save():
    shots = _shots("confirmed", "unreviewed", "changed", "unreviewed")
    assert labels.resume(shots, [{"shot": 3}]) == 4
    assert labels.resume(shots, [{"shot": 4}, {"shot": 1}]) == 2
    assert labels.resume(shots, []) == 2
    wraps = _shots("unreviewed", "confirmed", "unreviewed", "confirmed")
    assert labels.resume(wraps, [{"shot": 3}]) == 1
    done = _shots("confirmed", "changed", "confirmed", "changed")
    assert labels.resume(done, [{"shot": 2}]) == 2
    assert labels.resume([], []) is None


def test_the_queue_follows_the_roster_and_resumes(event_dir):
    roster = pd.DataFrame({"shot": [178642, 170815], "tier": ["unverified", "gold"]})
    assert labels.queue(event_dir, roster) == {
        "shots": [
            {"shot": 178642, "tier": "unverified", "state": "unreviewed",
             "saved_at": None},
            {"shot": 170815, "tier": "gold", "state": "unreviewed", "saved_at": None},
        ],
        "resume": 178642,
    }
    entry = labels.save(event_dir, 178642, normalise((0, 2000), []), source=None)
    after = labels.queue(event_dir, roster)
    assert after["shots"][0] == {
        "shot": 178642,
        "tier": "unverified",
        "state": "confirmed",
        "saved_at": entry["saved_at"],
    }
    assert after["resume"] == 170815


def test_shot_labels_carries_source_saved_state_and_last_save(event_dir):
    assert labels.shot_labels(event_dir, 170815) == {
        "source": {"window": [0, 2000], "intervals": [[100, 300, 1]], "iscrowd": [1]},
        "saved": None,
        "state": "unreviewed",
        "last_save": None,
    }
    label = normalise((0, 2000), [(100, 250, 1)])
    entry = labels.save(event_dir, 170815, label, source="x.csv")
    view = labels.shot_labels(event_dir, 170815)
    assert view["saved"] == {
        "window": [0, 2000], "intervals": [[100, 250, 1]], "iscrowd": [1],
    }
    assert view["state"] == "changed"
    assert view["last_save"] == entry


def test_categories_leave_out_absent_and_not_observable():
    assert labels.categories("resistive_wall_mode") == {
        1: "present", 2: "uncertain", 4: "unassessed",
    }
    assert labels.categories("alfven_eigenmode") == {1: "present", 2: "uncertain"}
    assert labels.categories("minimum_safety_factor") == {
        1: "low",
        2: "hybrid",
        3: "elevated",
        4: "high",
        5: "uncertain",
    }
    assert labels.categories("confinement") == {
        1: "high", 2: "low", 3: "qh", 4: "wpqh", 5: "uncertain", 6: "imode",
    }
    for event in ("mhd_n1", "mhd_n3", "mhd_n1.m1", "alfven_eigenmode.rsae"):
        assert labels.categories(event) == {1: "present", 2: "uncertain"}


def test_a_span_in_a_category_the_page_does_not_offer_is_a_gap(event_dir):
    label = normalise(
        (0, 2000), [(100, 250, 3), (400, 500, 1), (600, 700, 2)], iscrowd=[1, 0, 1]
    )
    shown = labels.offered("alfven_eigenmode", label)
    assert shown.window == (0, 2000)
    assert shown.intervals == ((400, 500, 1), (600, 700, 2))
    assert shown.iscrowd == (0, 1)
    assert labels.offered("alfven_eigenmode", shown) is shown
    assert labels.offered("alfven_eigenmode", None) is None
    labels.save(event_dir, 170815, label, source=None)
    saved = labels.shot_labels(event_dir, 170815)["saved"]
    assert saved["intervals"] == [[400, 500, 1], [600, 700, 2]]
    assert saved["iscrowd"] == [0, 1]


def test_a_not_observable_stretch_does_not_make_a_saved_label_differ(event_dir):
    roster = pd.DataFrame({"shot": [170815], "tier": ["gold"]})
    label = normalise((0, 2000), [(100, 300, 1), (1000, 1100, 3)])
    labels.save(event_dir, 170815, label, source=None)
    assert labels.queue(event_dir, roster)["shots"][0]["state"] == "confirmed"
    assert labels.shot_labels(event_dir, 170815)["state"] == "confirmed"


def test_saving_another_shot_preserves_existing_attributes(tmp_path):
    from labeler.events.interval_tables import WITH_ATTRS

    event = tmp_path / "disruption"
    review = event / "review"
    review.mkdir(parents=True)
    attrs = '{"intentional": false, "phase": "flattop"}'
    original = pd.DataFrame(
        [[190001, 1, "0.25", "100.50", "0.750", attrs]], columns=WITH_ATTRS
    )
    original.to_csv(review / "labels.csv", index=False)
    before = pd.read_csv(review / "labels.csv", dtype=str, keep_default_na=False)
    labels.save(event, 190002, normalise([0, 100], [[20, 40, 2]]), source=None)
    after = pd.read_csv(review / "labels.csv", dtype=str, keep_default_na=False)
    assert "attrs" in after.columns
    pd.testing.assert_frame_equal(after[after.shot == "190001"], before)
    assert (after.loc[after.shot == "190002", "attrs"] == "").all()
    before_refusal = {
        name: (review / name).read_bytes() for name in ("labels.csv", "history.jsonl")
    }
    with pytest.raises(ValueError, match="category 3"):
        label = normalise(
            [0, 100], [[30, 50, 3]], known=set(labels.categories(event.name))
        )
        labels.save(event, 190002, label, source=None)
    assert {
        name: (review / name).read_bytes() for name in before_refusal
    } == before_refusal


@pytest.mark.parametrize("event", PHENOMENA)
def test_review_menu_is_the_checkers_span_states_less_not_observable(event):
    accepted = set()
    for state in STATE_NAMES:
        frame = pd.DataFrame({"shot": [190001], "category": [state], "t_start": [0]})
        if not states(frame, category=event):
            accepted.add(state)
    assert set(labels.categories(event)) == accepted - {0, NOT_OBSERVABLE}


@pytest.mark.parametrize("event", [e for e in PHENOMENA if e not in labels.FOLDED])
def test_review_refuses_a_not_observable_span(tmp_path, event):
    from fastapi.testclient import TestClient

    from labeler.config import Paths
    from labeler.events.ui.app import COOKIE, create_app

    tables = tmp_path / "events"
    directory = tables / event
    directory.mkdir(parents=True)
    (directory / "shots.csv").write_text(
        "shot,tier,holdout,reviewers,verified_on,notes\n190001,unverified,false,,,\n"
    )
    paths = Paths(
        root=tmp_path / "root", corpus=tmp_path / "corpus",
        text_root=tmp_path / "text", logs_jsonl=tmp_path / "logs.jsonl",
        label_tables=tables, raw_cache=tmp_path / "raw",
    )
    with TestClient(create_app(paths=paths, token="secret")) as client:
        client.cookies.set(COOKIE, "secret")
        response = client.post("/api/label", json={
            "event": event, "shot": 190001,
            "window": [0, 100], "intervals": [[0, 100, 3]],
        })
    assert response.status_code == 400
    assert "category 3" in response.json()["error"]
    assert not labels.labels_path(directory).exists()
    assert not labels.history_path(directory).exists()


@pytest.mark.parametrize("attrs", ['{"intentional": false}', '{}'])
def test_saving_a_shot_with_attrs_refuses_without_changing_either_file(tmp_path, attrs):
    from labeler.events.interval_tables import WITH_ATTRS

    event = tmp_path / "disruption"
    review = event / "review"
    review.mkdir(parents=True)
    pd.DataFrame(
        [[190001, 1, 0, 100, None, attrs]], columns=WITH_ATTRS
    ).to_csv(review / "labels.csv", index=False)
    (review / "history.jsonl").write_text('{"shot": 190001}\n')
    before = {name: (review / name).read_bytes() for name in [
        "labels.csv", "history.jsonl"
    ]}
    with pytest.raises(ValueError, match="190001.*attrs"):
        labels.save(event, 190001, normalise([0, 100], [[20, 40, 2]]), source=None)
    assert {name: (review / name).read_bytes() for name in before} == before


@pytest.fixture
def regrid():
    script = Path(__file__).resolve().parents[2] / "scripts/labeler/regrid_saved_labels.py"
    return runpy.run_path(str(script))["regrid"]


@pytest.fixture
def regrid_tables(tmp_path):
    root = tmp_path / "tables"
    files = {
        "edge_localized_mode/review/labels.csv": (
            "shot,category,t_start,t_end,confidence,attrs\n"
            "7,0,3,21,,\n"
            '7,1,21,24,,"{""iscrowd"": 0}"\n'
            '7,1,22,23,,"{""iscrowd"": 1}"\n'
            "7,0,24,96,,\n7,1,55,55,,\n"
            '8,1,0,100,,"{""iscrowd"": 1}"\n'
        ),
        "mhd_n1/review/labels.csv": (
            "shot,category,t_start,t_end,confidence\n"
            "7,0,0,14,\n7,1,14,76,\n7,0,76,100,\n"
        ),
        "mhd_n1/subcategories/m1/review/labels.csv": (
            "shot,category,t_start,t_end,confidence\n"
            "7,0,0,13,\n7,1,13,24,\n7,0,24,100,\n"
        ),
        "mhd_n1/subcategories/m2/review/labels.csv": (
            "shot,category,t_start,t_end,confidence\n"
            "7,0,0,5,\n7,1,5,24,\n7,0,24,100,\n"
        ),
        "sawtooth_oscillation/review/labels.csv": (
            "shot,category,t_start,t_end,confidence\n9,1,23,68,\n"
        ),
    }
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        path.with_name("history.jsonl").write_text('{"shot": 7, "reviewer": "a"}\n')
    # This child crosses the snapped parent's boundary without being clipped.
    (root / "mhd_n1/subcategories/m2/review/labels.csv").write_text(
        "shot,category,t_start,t_end,confidence\n"
        "7,1,1,24,\n7,0,24,100,\n"
    )
    return root


def _contents(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


def test_regrid_dry_run_writes_nothing_and_reports_children_and_midpoints(
    regrid, regrid_tables, tmp_path,
):
    before = _contents(tmp_path)
    report = regrid(regrid_tables, backup_root=tmp_path / "backups")
    assert _contents(tmp_path) == before
    files = {row["path"]: row for row in report["files"]}
    elm = files["edge_localized_mode/review/labels.csv"]
    assert elm["rows_before"] == 6 and elm["rows_after"] == 5
    assert elm["shots_changed"] == [7]
    assert elm["midpoint_claims"] == 2 and elm["spans_dropped"] == 1
    assert files["mhd_n1/subcategories/m1/review/labels.csv"]["child_violations"] == []
    violations = files["mhd_n1/subcategories/m2/review/labels.csv"]["child_violations"]
    assert len(violations) == 1 and violations[0]["shot"] == 7
    assert violations[0]["outside"] == [[0, 10]]
    order = [row["path"] for row in report["files"]]
    assert order.index("mhd_n1/review/labels.csv") < order.index(
        "mhd_n1/subcategories/m1/review/labels.csv"
    )
    assert "sawtooth_oscillation/review/labels.csv" in files


def test_regrid_apply_backs_up_preserves_lanes_and_history_and_is_idempotent(
    regrid, regrid_tables, tmp_path,
):
    before = _contents(regrid_tables)
    backups = tmp_path / "backups"
    report = regrid(regrid_tables, apply=True, backup_root=backups)
    assert json.loads((backups / "report.json").read_text()) == report
    for name, content in before.items():
        if name.endswith("labels.csv"):
            assert (backups / name).read_bytes() == content
            frame = pd.read_csv(regrid_tables / name)
            assert (frame.t_start % 10 == 0).all() and (frame.t_end % 10 == 0).all()
        else:
            assert (regrid_tables / name).read_bytes() == content
    elm = labels.read_saved(regrid_tables / "edge_localized_mode", grid_ms=10)
    assert elm[7] == Label((0, 100), ((20, 30, 1), (20, 30, 1)), (0, 1))
    assert elm[8] == Label((0, 100), ((0, 100, 1),), (1,))
    child = labels.read_saved(regrid_tables / "mhd_n1/subcategories/m2", grid_ms=10)[7]
    assert child.intervals == ((0, 20, 1),), "violations are reported, never clipped"
    written = _contents(regrid_tables)
    with pytest.raises(FileExistsError, match="backup"):
        regrid(regrid_tables, apply=True, backup_root=backups)
    assert _contents(regrid_tables) == written
    second = regrid(regrid_tables, apply=True, force=True, backup_root=backups)
    assert all(row["shots_changed"] == [] for row in second["files"])
    assert json.loads((backups / "report.json").read_text()) == report, "first kept"
    [rerun] = backups.glob("report.rerun-*.json")
    assert json.loads(rerun.read_text()) == second
    assert _contents(regrid_tables) == written
    for name, content in before.items():
        if name.endswith("labels.csv"):
            assert (backups / name).read_bytes() == content, "a rerun keeps the original"


def test_regrid_keeps_evidence_attrs_when_tiny_spans_collapse(regrid, tmp_path):
    root = tmp_path / "tables"
    path = root / "resistive_wall_mode/review/labels.csv"
    path.parent.mkdir(parents=True)
    attrs = attrs_text({"evidence_tier": "reviewed", "coverage_verified": True})
    pd.DataFrame(
        [[7, 0, 3, 21, "", attrs], [7, 4, 21, 24, "", attrs],
         [7, 1, 24, 96, "", attrs]], columns=WITH_ATTRS,
    ).to_csv(path, index=False)
    report = regrid(root, apply=True, backup_root=tmp_path / "backups")
    [entry] = report["files"]
    assert entry["midpoint_claims"] == 0 and entry["spans_dropped"] == 1
    label = labels.read_labels(path)[7]
    assert label.intervals == ((0, 20, 0), (20, 100, 1))
    assert label.attrs == (attrs, attrs)


def test_regrid_keeps_extra_attrs_on_adjacent_annotations(regrid, tmp_path):
    root = tmp_path / "tables"
    path = root / "alfven_eigenmode/review/labels.csv"
    path.parent.mkdir(parents=True)
    attrs = [attrs_text({"type": kind}) for kind in ("TAE", "RSAE")]
    pd.DataFrame(
        [[7, 1, 1, 21, "", attrs[0]], [7, 1, 21, 39, "", attrs[1]]],
        columns=WITH_ATTRS,
    ).to_csv(path, index=False)
    regrid(root, apply=True, backup_root=tmp_path / "backups")
    frame = pd.read_csv(path)
    assert frame[["t_start", "t_end"]].values.tolist() == [[0, 20], [20, 40]]
    assert frame["attrs"].tolist() == attrs


def test_regrid_reports_an_individual_overwritten_in_its_midpoint_frame(
    regrid, tmp_path,
):
    root = tmp_path / "tables"
    path = root / "edge_localized_mode/review/labels.csv"
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        [[7, 1, 21, 24, "", '{"iscrowd": 0}'],
         [7, 1, 22, 23, "", '{"iscrowd": 0}']], columns=WITH_ATTRS,
    ).to_csv(path, index=False)
    [entry] = regrid(root)["files"]
    assert entry["midpoint_claims"] == 2 and entry["spans_dropped"] == 1


def test_onset_times_are_the_events_own_marker_sorted():
    markers = [
        {"label": "ELM onset", "t_ms": [30.0, 10.0]},
        {"label": "other", "t_ms": [5.0]},
        {"label": "ELM onset", "t_ms": [20.0]},
    ]
    assert labels.onset_times("edge_localized_mode", markers) == [10.0, 20.0, 30.0]
    assert labels.onset_times("alfven_eigenmode", markers) == []
    assert labels.onset_times("edge_localized_mode", None) == []


def test_with_onsets_drafts_the_source_as_spans_and_each_onset_as_an_individual():
    source = Label((0, 1000), ((100, 300, 1),))
    draft = labels.with_onsets(source, [105.0, 109.9, 250.0, 2000.0], (0, 1000))
    # the unflagged source span is a Span; 105 and 109.9 share a frame; 2000 is outside
    assert draft.window == (0, 1000)
    assert sorted(zip(draft.intervals, draft.iscrowd, strict=True)) == [
        ((100, 110, 1), 0), ((100, 300, 1), 1), ((250, 260, 1), 0),
    ]
    assert source.iscrowd == ()  # the label it was given is not touched


def test_with_onsets_keeps_the_source_flags_that_are_set():
    source = Label((0, 1000), ((100, 300, 1), (400, 500, 2)), (0, None))
    draft = labels.with_onsets(source, [700.0], (0, 1000))
    assert dict(zip(draft.intervals, draft.iscrowd, strict=True)) == {
        (100, 300, 1): 0, (400, 500, 2): 1, (700, 710, 1): 0,
    }


def test_with_onsets_over_no_source_opens_the_shots_frames_with_only_the_onsets():
    draft = labels.with_onsets(None, [105.0, 400.0], (3.0, 1000.0))
    assert draft.window == (0, 1000)
    assert draft.intervals == ((100, 110, 1), (400, 410, 1))
    assert draft.iscrowd == (0, 0)
    long_shot = labels.with_onsets(None, [105.0], (0.0, 30_000.0))
    assert long_shot.window == (0, labels.LONGEST_WINDOW_MS)


def test_with_onsets_offers_no_draft_it_could_not_save_or_edit():
    source = Label((0, 20_000), ((100, 300, 1),))
    assert labels.with_onsets(source, [], (0, 20_000)) is source
    assert labels.with_onsets(None, [], (0, 1000)) is None
    assert labels.with_onsets(None, [5000.0], (0, 1000)) is None  # outside the window
    many = [float(t) for t in range(0, 10 * (labels.MAX_INTERVALS + 1), 10)]
    assert labels.with_onsets(source, many, (0, 20_000)) is source
    assert labels.with_onsets(None, many, (0, 20_000)) is None
    evidence = Label((0, 100), ((0, 100, 1),), (), ('{"evidence_tier": "a"}',))
    assert labels.with_onsets(evidence, [10.0], (0, 100)) is evidence
