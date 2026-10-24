"""Several accounts saving into one folder: per-writer files, the merge, the checks.

Two "writers" are two logins on two nodes (`as_writer`) in one temporary folder; the
folder's own `labeler-sync` commands are run through `sync.main`.
"""

from __future__ import annotations

import contextlib
import errno
import importlib.util
import itertools
import json
import os
import stat
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from labeler.ae.seg import regions
from labeler.events.review import labels, reviewers, saves, sync
from labeler.events.review.labels import normalise

EVENT = "alfven_eigenmode"
WHOLE = (0, 2000)
JALAL = Path(__file__).resolve().parents[2] / "scripts/labeler/jalal_wpqh_import.py"


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
    saves.append_text(path, json.dumps(line([3], 3)))  # parses, but no merge could file it
    assert [e["shot"] for e, _ in saves.read_lines(path)] == [1, 2]
    merged = merge(event_dir)  # no TypeError, so no later save fails either
    assert [e["shot"] for e in labels.read_history(event_dir)] == [1, 2] and len(merged.bad) == 2


def test_a_stale_file_handle_is_retried_on_the_read_paths(event_dir, monkeypatch, clock):
    labels.save(event_dir, 1, spans((100, 300, 1)), source=None, name="Ada")
    regions.save_decision(event_dir, 170815, [1], pseudo_sha256="0" * 64, name="Ada")
    stale = []

    def once(real, where):
        def read(*args, **kwargs):
            if where not in stale:
                stale.append(where)
                raise OSError(errno.ESTALE, "stale")
            return real(*args, **kwargs)
        return read

    labels._read_table.cache_clear()
    monkeypatch.setattr(labels.pd, "read_csv", once(pd.read_csv, "table"))
    assert labels.read_saved(event_dir) == {1: spans((100, 300, 1))}
    real_text = Path.read_text

    def read_text(self, *args, **kwargs):
        if self.name in ("masks.jsonl", "history.jsonl") and self.name not in stale:
            stale.append(self.name)
            raise OSError(errno.ESTALE, "stale")
        return real_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    assert [d["shot"] for d in regions.shot_history(event_dir)] == [170815]
    assert reviewers.saved_names(event_dir.parent) == ["Ada"]
    assert stale == ["table", "masks.jsonl", "history.jsonl"]


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
    imported = labels.read_history(event_dir)
    assert [e["name"] for e in imported] == ["Alvin", "Nathaniel"]
    assert "legacy_id" in imported[1] and not any("save_id" in e for e in imported)
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


def test_files_and_directories_get_their_modes_whatever_the_umask(events, event_dir, clock):
    """Files are 0644 under any umask. A directory is made group-writable (2775) only
    under a parent that is group-writable already, as `init` leaves a shared folder;
    in a closed folder it is a plain 0755."""
    shared = events / "confinement"
    shared.mkdir()
    shared.chmod(0o2775)
    old = os.umask(0o077)
    try:
        labels.save(event_dir, 1, spans((100, 300, 1)), source=None)
        labels.save(shared, 1, spans((100, 300, 1)), source=None)
    finally:
        os.umask(old)
    for folder, mode in ((event_dir, 0o755), (shared, 0o775)):
        review = folder / "review"
        for path in (review / "labels.csv", review / "history.jsonl",
                     *(review / "saves").glob("*.jsonl")):
            assert stat.S_IMODE(path.stat().st_mode) == 0o644, path
        for path in (review, review / "saves"):
            assert stat.S_IMODE(path.stat().st_mode) & 0o777 == mode, path
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


def test_a_bad_table_stops_the_merge_and_the_pending_saves_wait(events, event_dir, capsys):
    """One invalid row (any shot) is nobody's save to refuse: nothing is rejected, the
    lines stay pending, and the next merge after the table is fixed takes them."""
    put(event_dir, "alice.n1", line(1, 1))
    merge(event_dir)
    table = labels.labels_path(event_dir)
    good = table.read_text()
    table.write_text(good + "1,1,900,800,\n")  # shot 1: the span ends before it starts
    put(event_dir, "bob.n2", line(2, 2))
    with pytest.raises(ValueError):
        merge(event_dir)
    assert not (event_dir / "review" / "saves" / "rejected.jsonl").exists()
    assert saves.pending(event_dir, "history") == 1
    assert sync.main(["merge", "--events", str(events)]) == 2
    assert "not merged" in capsys.readouterr().out
    table.write_text(good)
    assert sync.main(["merge", "--events", str(events)]) == 0
    assert sorted(labels.read_saved(event_dir)) == [1, 2]
    assert [e["shot"] for e in labels.read_history(event_dir)] == [1, 2]


def test_a_refused_line_is_noted_and_retried_when_asked(events, event_dir, capsys):
    """A line the table refuses for its own sake is recorded once, `check` and `status`
    say so, `merge --retry-rejected` takes it after the cause is fixed, and a line an
    owner script took back is never retried."""
    review = saves.review_dir(event_dir)
    table = labels.labels_path(event_dir)
    table.write_text(
        'shot,category,t_start,t_end,confidence,attrs\n9,1,100,200,,"{""note"": 1}"\n')
    put(event_dir, "alice.n1", line(9, 1))      # shot 9 has attrs the page cannot edit
    put(event_dir, "alice.n1", line(1, 2))
    taken_back = line(5, 3)
    put(event_dir, "alice.n1", taken_back)
    saves.reject(review, [(taken_back, "withdrawn")])
    first = merge(event_dir)
    assert [e["shot"] for e in first.added] == [1] and len(first.skipped) == 1
    record = review / "saves" / "rejected.jsonl"
    assert len(record.read_text().splitlines()) == 2
    run = ["--events", str(events)]
    assert sync.main(["check", *run]) == 0
    out = capsys.readouterr().out
    assert "1 saves the label table refused" in out and "shots 9" in out
    assert sync.main(["status", *run]) == 0
    assert "1 saves the table refused" in capsys.readouterr().out

    again = saves.merge(event_dir, "history", labels.apply_labels(event_dir),
                        labels.stale_labels(event_dir), retry_rejected=True)
    assert len(again.skipped) == 1 and not again.released  # still refused, not recorded twice
    assert len(record.read_text().splitlines()) == 2
    table.write_text(table.read_text().replace('"{""note"": 1}"', ""))  # the owner clears it
    assert sync.main(["merge", "--retry-rejected", *run]) == 0
    assert sorted(labels.read_saved(event_dir)) == [1, 9]
    assert [json.loads(t)["shot"] for t in record.read_text().splitlines()] == [5]
    assert sync.main(["merge", "--retry-rejected", *run]) == 0
    assert 5 not in labels.read_saved(event_dir)


def test_a_clock_running_fast_does_not_break_a_live_lock(event_dir, monkeypatch):
    review = saves.review_dir(event_dir)
    (review / saves.LOCK).mkdir()
    real = saves.time.time
    monkeypatch.setattr(saves.time, "time", lambda: real() + 3600)
    assert saves.lock_age(review) > 3000   # this machine's clock says it is an hour old
    assert saves._break_stale(review / saves.LOCK, 120) is False
    old = real() - 600
    os.utime(review / saves.LOCK, (old, old))
    assert saves._break_stale(review / saves.LOCK, 120) is True


def test_leaving_a_lock_never_removes_the_lock_of_the_process_that_broke_it(event_dir):
    """A holder that outlives the stale age (suspended, a hard NFS mount) finds its lock
    broken and taken by B. When A leaves, B's lock stays, so C cannot enter while B is in."""
    review = saves.review_dir(event_dir)
    a, b = contextlib.ExitStack(), contextlib.ExitStack()
    a.enter_context(saves.lock(review))
    old = time.time() - 600
    os.utime(review / saves.LOCK, (old, old))
    b.enter_context(saves.lock(review, wait=1))   # B breaks A's stale lock and takes its own
    a.close()
    assert (review / saves.LOCK).is_dir()
    with pytest.raises(saves.StoreBusy), saves.lock(review, wait=0):  # C is refused
        pass
    b.close()
    assert not (review / saves.LOCK).exists()
    # a lock broken as stale that nobody took since is simply gone when its holder leaves
    c = contextlib.ExitStack()
    c.enter_context(saves.lock(review))
    saves.shutil.rmtree(review / saves.LOCK)
    c.close()
    assert not [p.name for p in review.iterdir() if p.name.startswith(saves.LOCK)]


def test_a_non_utf8_byte_in_a_writer_file_skips_that_line_and_stops_nothing(
    events, event_dir, monkeypatch, clock, capsys
):
    labels.save(event_dir, 1, spans((100, 300, 1)), source=None)
    torn = b'{"shot": 2, "sav\xff\n'                  # cut off, and not UTF-8
    named = json.dumps(line(3, 3, name="caf@")).encode().replace(b"@", b"\xe9")  # JSON, bad byte
    fine = json.dumps(line(4, 4)).encode()
    mallory = event_dir / "review" / saves.SAVES / "history.mallory.n9.jsonl"
    mallory.write_bytes(torn + named + b"\n" + fine + b"\n")
    as_writer(monkeypatch, "bob", "n2")
    outcome: dict = {}
    labels.save(event_dir, 5, spans((100, 300, 1)), source=None, outcome=outcome)
    assert outcome["pending"] is False
    assert sorted(labels.read_saved(event_dir)) == [1, 4, 5]   # 2 and 3 were never filed
    assert saves.pending(event_dir, "history") == 0
    run = ["--events", str(events)]
    capsys.readouterr()
    assert sync.main(["status", *run]) == 0
    assert "2 torn lines" in capsys.readouterr().out
    assert sync.main(["check", *run]) == 0
    assert "2 torn, non-JSON or shotless lines" in capsys.readouterr().out


def test_init_applies_the_permission_table(events, event_dir):
    review = saves.review_dir(event_dir)
    (event_dir / "subcategories").mkdir()
    (event_dir / "raw").mkdir()
    key = event_dir / "raw" / "elm_challenge_key_b71f991.json"
    key.write_text("{}")
    table = review / "labels.csv"
    table.write_text("shot\n")
    table.chmod(0o600)
    key.chmod(0o644)
    assert sync.main(["init", "--events", str(events)]) == 0
    modes = {p: stat.S_IMODE(p.stat().st_mode) for p in (
        review, review / "saves", event_dir / "subcategories", table, key)}
    assert modes == {
        review: 0o2775, review / "saves": 0o2775, event_dir / "subcategories": 0o2775,
        table: 0o644, key: 0o600,
    }


def test_check_notes_a_line_an_older_server_added_after_the_new_ones(events, event_dir, capsys):
    put(event_dir, "alice.n1", line(1, 1, save_id="20261024120001000000-a1"))
    merge(event_dir)
    with (event_dir / "review" / "history.jsonl").open("a") as stream:  # old code appends
        stream.write(json.dumps(line(2, 5)) + "\n")
    assert sync.main(["check", "--events", str(events)]) == 0
    assert "without a save_id come after" in capsys.readouterr().out


def test_a_legacy_line_is_not_held_to_the_span_flags_a_script_added(event_dir, clock):
    """`repair --shot` writes a save back; for an old line, whose history never had the
    iscrowd flags the table has, only a real difference counts."""
    crowd = normalise(WHOLE, [(100, 300, 1), (200, 400, 2)], iscrowd=[1, 1])
    labels.save(event_dir, 1, crowd, source=None, crowd_edit=True)
    path = event_dir / "review" / "history.jsonl"
    saved = json.loads(path.read_text())
    old = {k: v for k, v in saved.items() if k not in ("save_id", "writer", "iscrowd")}
    path.write_text(json.dumps(old) + "\n")
    assert labels.read_saved(event_dir)[1].iscrowd
    assert labels.diverged(event_dir, legacy=True) == []
    old["intervals"] = [[100, 350, 1]]
    path.write_text(json.dumps(old) + "\n")
    assert [e["shot"] for e in labels.diverged(event_dir, legacy=True)] == [1]


def jalal():
    """The owner's import script, which takes saves back (`withdraw`)."""
    spec = importlib.util.spec_from_file_location("jalal_for_sync_tests", JALAL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # the script's dataclasses look their module up
    spec.loader.exec_module(module)
    return module


def test_import_tree_leaves_out_what_the_source_took_back(
    events, event_dir, tmp_path, clock, capsys
):
    """A withdrawn save stays in its writer file in the source and is recorded in the
    source's rejected.jsonl; importing that folder must not bring it back, and must
    record the take-back here."""
    jw = jalal()
    there = tmp_path / "src" / EVENT
    there.mkdir(parents=True)
    labels.save(there, 1, spans((100, 300, 1)), source=None)
    labels.save(there, 2, spans((100, 300, 1)), source=None, name=jw.NEW_NAME,
                extra={"note": jw.SOURCE_NOTE})
    jw.withdraw(there, 2)
    assert sorted(labels.read_saved(there)) == [1]
    run = ["import-tree", str(tmp_path / "src"), "--events", str(events)]
    assert sync.main([*run, "--dry-run"]) == 0   # nothing recorded yet: the filter alone
    assert "1 lines there, 1 new; would add 1" in capsys.readouterr().out
    assert sync.main(run) == 0
    assert sorted(labels.read_saved(event_dir)) == [1]
    assert [e["shot"] for e in labels.read_history(event_dir)] == [1]
    record = event_dir / "review" / "saves" / "rejected.jsonl"
    assert [(r["shot"], r["retry"]) for r in map(json.loads, record.read_text().splitlines())] == [
        (2, False)]
    before = snapshot(events)
    assert sync.main(run) == 0 and snapshot(events) == before
    assert sync.main(["merge", "--retry-rejected", "--events", str(events)]) == 0
    assert sorted(labels.read_saved(event_dir)) == [1]  # a take-back is never retried


def test_import_tree_makes_a_subcategory_and_refuses_what_it_cannot_read_or_place(
    events, event_dir, tmp_path, capsys
):
    other = tmp_path / "other"
    (other / EVENT).mkdir(parents=True)
    (other / "confinement").mkdir()
    put(other / EVENT / "subcategories" / "tae", "carol.n1", line(1, 1))
    put(other / "confinement", "carol.n1", line(2, 2))
    (other / EVENT / "review").mkdir()
    (other / EVENT / "review" / "history.jsonl").write_text(
        '{"shot": 3, "sav\n' + json.dumps(line("x", 4)) + "\n" + json.dumps(line(5, 5)) + "\n")
    run = ["import-tree", str(other), "--events", str(events)]
    child = event_dir / "subcategories" / "tae"
    assert sync.main([*run, "--event", f"{EVENT}/subcategories/tae", "--dry-run"]) == 0
    assert not child.exists()
    assert sync.main([*run, "--event", f"{EVENT}/subcategories/tae"]) == 0
    assert sorted(labels.read_saved(child)) == [1]
    capsys.readouterr()
    assert sync.main([*run, "--event", "confinement"]) == 2
    assert "never made" in capsys.readouterr().out and not (events / "confinement").exists()
    assert sync.main([*run, "--event", EVENT]) == 2   # the good line is taken, the others counted
    assert "2 unreadable lines" in capsys.readouterr().out
    assert [e["shot"] for e in labels.read_history(event_dir)] == [5]


def test_check_status_and_init_leave_a_busy_folder_alone(
    events, event_dir, monkeypatch, clock, capsys
):
    """Another process's probe and temporary files (0600 for a moment) are not problems,
    and a read that is retried on a stale handle makes no probe file in the folder."""
    labels.save(event_dir, 1, spans((100, 300, 1)), source=None)
    review = event_dir / "review"
    for name in (".probe.1234abcd", ".labels.csv.n2.77.abcdef01.tmp"):
        (review / name).write_text("")
        (review / name).chmod(0o600)
    run = ["--events", str(events)]
    assert sync.main(["check", *run]) == 0
    capsys.readouterr()
    assert sync.main(["init", "--dry-run", *run]) == 0
    assert ".probe" not in capsys.readouterr().out
    real = Path.read_text
    for command in ("status", "check"):
        failed = []

        def read_text(self, *args, failed=failed, **kwargs):
            if self.name == "history.jsonl" and not failed:
                failed.append(self)
                raise OSError(errno.ESTALE, "stale")
            return real(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", read_text)
        before = [d.stat().st_mtime_ns for d in (review, review / "saves")]
        assert sync.main([command, *run]) == 0 and failed
        assert [d.stat().st_mtime_ns for d in (review, review / "saves")] == before
        monkeypatch.setattr(Path, "read_text", real)


def test_a_mask_decision_can_skip_the_wait_for_a_held_lock(event_dir):
    review = saves.review_dir(event_dir)
    (review / saves.LOCK).mkdir()
    started = time.monotonic()
    entry = regions.save_decision(event_dir, 170815, [1], pseudo_sha256="0" * 64, wait=0)
    assert time.monotonic() - started < 5
    assert saves.pending(event_dir, "masks") == 1 and entry["shot"] == 170815
