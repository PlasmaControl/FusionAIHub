"""One label per shot: normalising, the source table, saving, states, the queue."""

from __future__ import annotations

import pandas as pd
import pytest

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
    assert labels.categories("alfven_eigenmode") == {1: "present"}
    assert labels.categories("minimum_safety_factor") == {
        1: "low",
        2: "hybrid",
        3: "elevated",
        4: "high",
    }
