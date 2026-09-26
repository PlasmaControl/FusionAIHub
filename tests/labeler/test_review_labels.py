"""One label per shot: normalising, the source table, saving, states, the queue."""

from __future__ import annotations

import pandas as pd
import pytest

from labeler.events.catalog.check import states
from labeler.events.catalog.states import PHENOMENA, STATE_NAMES
from labeler.events.interval_tables import validate_intervals
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
    assert normalise((0, 100), [(10.4, 20.5, 1)]).intervals == ((10, 21, 1),)


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
    assert normalise((0, 100), [(10, 10.2, 1)]).intervals == ()


@pytest.mark.parametrize(
    "window, intervals, known, match",
    [
        ((0, 100), [(30, 20, 1)], None, "runs backwards"),
        ((0, 100), [(10, 20, 7)], {1}, "category 7"),
        ((100, 100), [], None, "not 1 to 20000 ms"),
        ((0, 20001), [], None, "not 1 to 20000 ms"),
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
    ]


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
        "source": {"window": [0, 2000], "intervals": [[100, 300, 1]]},
        "saved": None,
        "state": "unreviewed",
        "last_save": None,
    }
    label = normalise((0, 2000), [(100, 250, 1)])
    entry = labels.save(event_dir, 170815, label, source="x.csv")
    view = labels.shot_labels(event_dir, 170815)
    assert view["saved"] == {"window": [0, 2000], "intervals": [[100, 250, 1]]}
    assert view["state"] == "changed"
    assert view["last_save"] == entry


def test_categories_leave_out_absent():
    assert labels.categories("resistive_wall_mode") == {1: "present"}
    assert labels.categories("alfven_eigenmode") == {
        1: "present",
        2: "uncertain",
        3: "not_observable",
    }
    assert labels.categories("minimum_safety_factor") == {
        1: "low",
        2: "hybrid",
        3: "elevated",
        4: "high",
    }


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
def test_review_menu_and_checker_accept_the_same_span_states(event):
    accepted = set()
    for state in STATE_NAMES:
        frame = pd.DataFrame({"shot": [190001], "category": [state], "t_start": [0]})
        if not states(frame, category=event):
            accepted.add(state)
    assert set(labels.categories(event)) == accepted - {0}


@pytest.mark.parametrize(
    "event", [key for key, spec in PHENOMENA.items() if spec.observable_always]
)
def test_review_refuses_states_that_its_checker_always_rejects(tmp_path, event):
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
    assert "category 3" in response.json()["detail"]
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
