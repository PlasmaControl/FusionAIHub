"""Several accounts saving into one folder: per-writer files, the merge, the checks.

Two "writers" are two logins on two nodes (`as_writer`) in one temporary folder; the
folder's own `labeler-sync` commands are run through `sync.main`.
"""

from __future__ import annotations

import errno
import itertools
import json
import os
import stat
from datetime import UTC, datetime, timedelta

import pytest

from labeler.ae.seg import regions
from labeler.events.review import labels, reviewers, saves, sync
from labeler.events.review.labels import normalise

EVENT = "alfven_eigenmode"
WHOLE = (0, 2000)


@pytest.fixture
def events(tmp_path):
    directory = tmp_path / "events"
    (directory / EVENT).mkdir(parents=True)
    return directory


@pytest.fixture
def event_dir(events):
    return events / EVENT


@pytest.fixture
def clock(monkeypatch):
    """Every save is one second after the last, so the order of saves is known."""
    start, count = datetime(2026, 10, 24, 12, 0, 0, tzinfo=UTC), itertools.count()

    def new_save(now=None):
        now = start + timedelta(seconds=next(count))
        return now.isoformat(timespec="seconds"), f"{now:%Y%m%d%H%M%S}000000-aaaaaa"

    monkeypatch.setattr(saves, "new_save", new_save)


def as_writer(monkeypatch, login, host):
    monkeypatch.setattr(labels.getpass, "getuser", lambda: login)
    monkeypatch.setattr(saves.socket, "gethostname", lambda: f"{host}.cluster.edu")


def spans(*cells):
    return normalise(WHOLE, list(cells))


def line(shot, at, who="alice", spans=((100, 300, 1),), **more):
    """A history line as an older server wrote it (no id), saved `at` seconds in."""
    saved_at = (datetime(2026, 10, 24, 12, 0, at, tzinfo=UTC)).isoformat(timespec="seconds")
    return {
        "shot": shot, "reviewer": who, "name": None, "saved_at": saved_at,
        "window": list(WHOLE), "intervals": [list(s) for s in spans],
        "source": None, "suggestions_shown": None, "prefilled": None, **more,
    }


def put(event_dir, writer, entry, kind="history"):
    """Add `entry` to the writer's own file, as a save does, without merging."""
    review = saves.review_dir(event_dir)
    saves.shared_mkdir(review / saves.SAVES, under=review)
    saves.append_text(review / saves.SAVES / f"{kind}.{writer}.jsonl", json.dumps(entry))


def merge(event_dir):
    return saves.merge(
        event_dir, "history", labels.apply_labels(event_dir), labels.stale_labels(event_dir)
    )


def snapshot(folder):
    return {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in sorted(folder.rglob("*")) if p.is_file()
    }


def test_two_writers_save_into_one_folder_and_a_lost_update_is_repaired(
    events, event_dir, monkeypatch, clock
):
    as_writer(monkeypatch, "alice", "n1")
    labels.save(event_dir, 1, spans((100, 300, 1)), source=None)
    labels.save(event_dir, 2, spans((100, 300, 1)), source=None)
    as_writer(monkeypatch, "bob", "n2")
    labels.save(event_dir, 2, spans((500, 700, 1)), source=None)
    review = event_dir / "review"
    assert labels.read_saved(event_dir) == {
        1: spans((100, 300, 1)), 2: spans((500, 700, 1)),
    }
    assert [e["writer"] for e in labels.read_history(event_dir)] == [
        "alice.n1", "alice.n1", "bob.n2",
    ]
    assert sorted(p.name for p in (review / "saves").iterdir()) == [
        "history.alice.n1.jsonl", "history.bob.n2.jsonl",
    ]
    assert not (review / ".lock").exists()

    # The other client read labels.csv before alice's shot 1 landed and renamed its
    # version over hers: the history knows shot 1, the table does not.
    stale = labels.labels_path(event_dir).read_text().split("\n")
    labels.labels_path(event_dir).write_text(
        "\n".join(row for row in stale if not row.startswith("1,")))
    assert 1 not in labels.read_saved(event_dir)
    assert sync.main(["check", "--events", str(events)]) == 1
    labels.save(event_dir, 3, spans((10, 20, 1)), source=None)  # the next merge heals it
    assert labels.read_saved(event_dir)[1] == spans((100, 300, 1))
    assert sync.main(["check", "--events", str(events)]) == 0

    # A label somebody edited on purpose is not a lost update: merges leave it, check
    # reports it, repair puts the last save back.
    table = labels.labels_path(event_dir)
    table.write_text(table.read_text().replace("500,700", "510,700"))
    labels.save(event_dir, 4, spans((10, 20, 1)), source=None)
    assert labels.read_saved(event_dir)[2].intervals == ((510, 700, 1),)
    assert sync.main(["check", "--events", str(events)]) == 1
    assert sync.main(["repair", "--events", str(events)]) == 0
    assert labels.read_saved(event_dir)[2] == spans((500, 700, 1))
    assert sync.main(["check", "--events", str(events)]) == 0


def test_merging_is_idempotent_and_does_not_depend_on_the_batches(tmp_path):
    saved = [
        ("alice.n1", line(1, 1)),
        ("bob.n2", line(1, 2, "bob", ((500, 600, 1),), save_id="20261024120002000000-b1")),
        ("alice.n1", line(2, 3)),
        ("bob.n2", line(2, 4, "bob", ((50, 60, 1),), save_id="20261024120004000000-b2")),
        ("alice.n1", line(1, 5, spans=((900, 950, 1),), save_id="20261024120005000000-a1")),
        ("bob.n2", line(2, 6, "bob", ((70, 80, 1),))),
    ]
    folders = {}
    for name, order, each in (
        ("one by one", saved, True), ("one batch", saved, False),
        ("late arrivals", saved[::-1], True),
    ):
        folder = folders[name] = tmp_path / name
        folder.mkdir()
        for writer, entry in order:
            put(folder, writer, entry)
            if each:
                merge(folder)
        merge(folder)
    reference = next(iter(folders.values()))
    for folder in folders.values():
        for name in ("labels.csv", "history.jsonl"):
            assert (folder / "review" / name).read_bytes() == (
                reference / "review" / name).read_bytes()
    assert labels.read_saved(reference)[1] == spans((900, 950, 1))
    assert labels.read_saved(reference)[2] == spans((70, 80, 1))
    before = snapshot(reference)
    assert not merge(reference).added
    assert snapshot(reference) == before, "a second merge changed a file"


def test_a_legacy_line_keeps_its_id_through_a_rename_and_a_regrid():
    old = line(5, 7)
    assert saves.line_id(old) == saves.line_id(
        {**old, "name": "Jalal Butt", "intervals": [[110, 310, 1]]})
    assert saves.line_id(old) != saves.line_id(line(5, 8))
    assert saves.line_id(old) != saves.line_id(line(6, 7))
    assert saves.line_id({**old, "save_id": "x1"}) == "x1"


def test_a_torn_writer_line_is_skipped_and_the_next_save_ends_it(event_dir):
    put(event_dir, "alice.n1", line(1, 1))
    path = event_dir / "review" / "saves" / "history.alice.n1.jsonl"
    with path.open("a") as stream:
        stream.write('{"shot": 7, "sav')
    bad: list = []
    assert [e["shot"] for e, _ in saves.read_lines(path, bad=bad)] == [1]
    assert bad == [(path, 2)]
    merged = merge(event_dir)
    assert [e["shot"] for e in merged.added] == [1] and len(merged.bad) == 1
    saves.append_text(path, json.dumps(line(2, 2)))
    assert [e["shot"] for e, _ in saves.read_lines(path)] == [1, 2]
    merge(event_dir)
    assert [e["shot"] for e in labels.read_history(event_dir)] == [1, 2]


def test_a_stale_file_handle_is_retried():
    calls = []

    def read():
        calls.append(1)
        if len(calls) == 1:
            raise OSError(errno.ESTALE, "stale")
        return "text"

    assert saves.retrying(read) == "text" and len(calls) == 2


def test_with_automerge_off_a_save_waits_for_labeler_sync_merge(
    events, event_dir, monkeypatch, clock, capsys
):
    monkeypatch.setenv("LABELER_AUTO_MERGE", "0")
    outcome: dict = {}
    entry = labels.save(event_dir, 1, spans((100, 300, 1)), source=None, outcome=outcome)
    assert outcome["pending"] is True
    assert not labels.labels_path(event_dir).exists()
    [(only, _)] = saves.read_lines(next((event_dir / "review" / "saves").iterdir()))
    assert only == entry
    assert sync.main(["status", "--events", str(events)]) == 0
    assert "1 unmerged" in capsys.readouterr().out
    monkeypatch.delenv("LABELER_AUTO_MERGE")
    assert sync.main(["merge", "--events", str(events)]) == 0
    assert labels.read_saved(event_dir) == {1: spans((100, 300, 1))}
    before = snapshot(events)
    assert sync.main(["merge", "--events", str(events)]) == 0
    assert snapshot(events) == before


def test_a_busy_lock_leaves_the_save_durable_and_a_stale_lock_is_broken(
    event_dir, monkeypatch, clock
):
    merge_once = saves.merge
    monkeypatch.setattr(saves, "merge", lambda *a, **k: merge_once(*a, **{**k, "wait": 0.2}))
    review = saves.review_dir(event_dir)
    (review / saves.LOCK).mkdir()
    outcome: dict = {}
    labels.save(event_dir, 1, spans((100, 300, 1)), source=None, outcome=outcome)
    assert outcome["pending"] is True and "held" in outcome["why"]
    assert not labels.labels_path(event_dir).exists()
    assert saves.pending(event_dir, "history") == 1
    old = (datetime.now(UTC) - timedelta(minutes=10)).timestamp()
    os.utime(review / saves.LOCK, (old, old))
    labels.save(event_dir, 2, spans((100, 300, 1)), source=None, outcome=outcome)
    assert outcome["pending"] is False
    assert sorted(labels.read_saved(event_dir)) == [1, 2]
    assert not (review / saves.LOCK).exists()


def test_import_tree_adds_what_is_new_once_and_never_overrides_a_newer_save(
    events, event_dir, tmp_path, capsys
):
    other = tmp_path / "other"
    (other / EVENT / "review").mkdir(parents=True)
    theirs = [line(1, 1), line(2, 2, "carol"), line(3, 3, "carol", ((10, 20, 1),))]
    (other / EVENT / "review" / "history.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in theirs))
    put(event_dir, "alice.n1", line(1, 1))   # known already (same line)
    put(event_dir, "alice.n1", line(2, 9, spans=((500, 700, 1),)))   # newer than carol's
    merge(event_dir)
    run = ["import-tree", str(other), "--events", str(events)]
    before = snapshot(events)
    assert sync.main([*run, "--dry-run"]) == 0
    assert "2 new" in capsys.readouterr().out and snapshot(events) == before
    assert sync.main(run) == 0
    assert sorted(labels.read_saved(event_dir)) == [1, 2, 3]
    assert labels.read_saved(event_dir)[2] == spans((500, 700, 1))
    assert labels.read_saved(event_dir)[3] == spans((10, 20, 1))
    assert len(labels.read_history(event_dir)) == 4
    capsys.readouterr()
    before = snapshot(events)
    assert sync.main(run) == 0
    assert "0 new" in capsys.readouterr().out and snapshot(events) == before


def test_old_lines_with_one_id_are_each_imported_once(events, event_dir, tmp_path):
    """Two saves of one shot by one login in one second (a script saved a label under
    two names) have one computed id; neither is lost, neither is added twice."""
    other = tmp_path / "other"
    (other / EVENT / "review").mkdir(parents=True)
    pair = [line(7, 1, name="Alvin"), line(7, 1, name="Nathaniel")]
    assert len({saves.line_id(e) for e in pair}) == 1
    log = other / EVENT / "review" / "history.jsonl"
    log.write_text("".join(json.dumps(e) + "\n" for e in pair))
    put(event_dir, "alice.n1", pair[0])  # one of the two is here already
    merge(event_dir)
    run = ["import-tree", str(other), "--events", str(events)]
    assert sync.main(run) == 0
    assert [e["name"] for e in labels.read_history(event_dir)] == ["Alvin", "Nathaniel"]
    before = snapshot(events)
    assert sync.main(run) == 0 and snapshot(events) == before
    assert sync.main(["check", "--events", str(events)]) == 0


def test_a_line_the_table_refuses_is_rejected_once_and_does_not_block_the_rest(
    events, event_dir
):
    put(event_dir, "alice.n1", {"shot": 9, "reviewer": "x", "saved_at": "2026-10-24T12:00:09+00:00"})
    put(event_dir, "alice.n1", line(1, 1))
    first = merge(event_dir)
    assert [e["shot"] for e in first.added] == [1] and len(first.skipped) == 1
    assert sorted(labels.read_saved(event_dir)) == [1]
    assert len((event_dir / "review" / "saves" / "rejected.jsonl").read_text().splitlines()) == 1
    again = merge(event_dir)
    assert not again.added and not again.skipped
    assert sync.main(["check", "--events", str(events)]) == 0


def test_files_and_directories_get_their_modes_whatever_the_umask(event_dir, clock):
    old = os.umask(0o077)
    try:
        labels.save(event_dir, 1, spans((100, 300, 1)), source=None)
    finally:
        os.umask(old)
    review = event_dir / "review"
    for path in (review / "labels.csv", review / "history.jsonl",
                 *(review / "saves").glob("*.jsonl")):
        assert stat.S_IMODE(path.stat().st_mode) == 0o644, path
    for path in (review, review / "saves"):
        assert stat.S_IMODE(path.stat().st_mode) & 0o777 == 0o775, path
    assert not [p for p in review.iterdir() if p.name.endswith(".tmp")]


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write anywhere")
def test_a_contributor_s_add_name_leaves_the_owners_list_alone(events, event_dir, clock):
    (events / reviewers.FILE).write_text("Ada\n")
    labels.save(event_dir, 1, spans((100, 300, 1)), source=None, name="Grace")
    events.chmod(0o555)
    (events / reviewers.FILE).chmod(0o444)
    try:
        names, name = reviewers.add(events, " Linus ")
    finally:
        (events / reviewers.FILE).chmod(0o644)
        events.chmod(0o755)
    assert (names, name) == (["Ada", "Grace", "Linus"], "Linus")
    assert (events / reviewers.FILE).read_text() == "Ada\n"
    labels.save(event_dir, 2, spans((100, 300, 1)), source=None, name="Linus")
    assert reviewers.read(events) == ["Ada", "Grace", "Linus"]


def test_two_writers_mask_decisions_merge_in_time_order(
    event_dir, monkeypatch, clock
):
    for login, host, rejected in (("alice", "n1", [1]), ("bob", "n2", [2]), ("alice", "n3", [])):
        as_writer(monkeypatch, login, host)
        regions.save_decision(event_dir, 170815, rejected, pseudo_sha256="0" * 64)
    assert [d["writer"] for d in regions.shot_history(event_dir, 170815)] == [
        "alice.n1", "bob.n2", "alice.n3",
    ]
    assert regions.read_decisions(event_dir)[170815]["rejected"] == []
    assert sorted(p.name for p in (event_dir / "review" / "saves").iterdir()) == [
        "masks.alice.n1.jsonl", "masks.alice.n3.jsonl", "masks.bob.n2.jsonl",
    ]


def test_a_missing_folder_is_never_made_but_a_subcategory_s_own_is(events, clock):
    with pytest.raises(saves.StoreMissing):
        labels.save(events / "no_such_event", 1, spans((100, 300, 1)), source=None)
    assert not (events / "no_such_event").exists()
    child = events / "mhd_n1" / "subcategories" / "m1"
    (events / "mhd_n1").mkdir()
    labels.save(child, 1, spans((100, 300, 1)), source=None)
    assert sorted(labels.read_saved(child)) == [1]
    missing = events.parent / "no_events" / "mhd_n1" / "subcategories" / "m1"
    with pytest.raises(saves.StoreMissing):
        labels.save(missing, 1, spans((100, 300, 1)), source=None)
    assert not (events.parent / "no_events").exists()
