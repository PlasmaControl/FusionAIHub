"""One label folder on /projects for both clusters (docs/labeler/shared_folder.md):
where the label tables resolve, the read-only mirror compute nodes read, pointers
that travel with the folder, and the probe that shows the lock holds across nodes."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from labeler import env
from labeler.config import Paths
from labeler.events import hf_export
from labeler.events.review import labels, saves, shared, sync

NAME = "LABELER_LABEL_TABLES"
EVENT = "edge_localized_mode"
TABLE = "shot,category,t_start,t_end,confidence\n7,0,100,300,\n7,1,300,400,\n"
needs_rsync = pytest.mark.skipif(shutil.which("rsync") is None, reason="needs rsync")


@pytest.fixture
def folders(tmp_path, monkeypatch):
    """(shared folder, this cluster's own folder) on cluster `stellar`, nothing exported."""
    shared_dir, local_dir = tmp_path / "projects" / "events", tmp_path / "scratch" / "events"
    shared_dir.mkdir(parents=True)
    local_dir.mkdir(parents=True)
    monkeypatch.setitem(env.SHARED, NAME, str(shared_dir))
    monkeypatch.setitem(env.CLUSTERS["stellar"]["env"], NAME, str(local_dir))
    monkeypatch.setenv("LABELER_CLUSTER", "stellar")
    monkeypatch.delenv(NAME, raising=False)
    env.switched_on.cache_clear()
    yield shared_dir, local_dir
    env.switched_on.cache_clear()


def suggestion(tmp_path, text=TABLE, *, sidecars=False) -> Path:
    folder = tmp_path / "root" / "suggestions" / "elm_clock" / "v1"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "t.csv").write_text(text)
    if sidecars:
        (folder / "t.meta.json").write_text("{}\n")
        (folder / "t_onsets.csv").write_text("shot\n7\n")
    return folder / "t.csv"


def event_in(events) -> Path:
    directory = Path(events) / EVENT
    (directory / saves.REVIEW).mkdir(parents=True, exist_ok=True)
    return directory


def old_pointer(event_dir, table) -> dict:
    """A pointer as written before copies: the table's absolute path."""
    entry = {"table": str(table), "method": "elm_clock", "version": "v1",
             "set_at": "2026-10-01T00:00:00+00:00", "set_by": "alice"}
    labels.pointer_path(event_dir).write_text(json.dumps(entry, indent=1) + "\n")
    return entry


# ---------------------------------------------------------------- where the tables are


def test_the_cluster_folder_until_the_shared_one_is_switched_on(folders):
    shared_dir, local_dir = folders
    assert env.resolve(NAME) == str(local_dir)
    (shared_dir / env.ACTIVE).write_text("{}\n")
    assert env.resolve(NAME) == str(local_dir)  # asked once per process
    env.switched_on.cache_clear()
    assert env.resolve(NAME) == str(shared_dir)
    assert Paths.from_env().label_tables == shared_dir


def test_a_node_without_the_shared_folder_reads_the_cluster_folder(folders, monkeypatch, tmp_path):
    _, local_dir = folders
    monkeypatch.setitem(env.SHARED, NAME, str(tmp_path / "not-mounted"))
    assert env.resolve(NAME) == str(local_dir)


def test_no_cluster_never_resolves_to_the_shared_folder(folders, monkeypatch):
    shared_dir, _ = folders
    (shared_dir / env.ACTIVE).write_text("{}\n")
    monkeypatch.setenv("LABELER_CLUSTER", "")
    assert env.resolve(NAME, "fallback") == "fallback"


def test_an_exported_shared_folder_a_node_does_not_mount_means_the_mirror(
    folders, monkeypatch, tmp_path, capsys
):
    shared_dir, local_dir = folders
    monkeypatch.setenv(NAME, f"{shared_dir}/")
    assert env.resolve(NAME) == f"{shared_dir}/"  # mounted: as exported
    absent = tmp_path / "not-mounted"
    monkeypatch.setitem(env.SHARED, NAME, str(absent))
    monkeypatch.setenv(NAME, str(absent))
    assert env.resolve(NAME) == env.resolve(NAME) == str(local_dir)
    assert capsys.readouterr().err.count("is not mounted on this node") == 1
    monkeypatch.setenv(NAME, str(tmp_path / "mine"))  # any other export wins
    assert env.resolve(NAME) == str(tmp_path / "mine")


# ---------------------------------------------------------------- the mirror is read-only


def test_nothing_saves_into_a_mirror(tmp_path):
    events = tmp_path / "events"
    event_dir = event_in(events)
    (events / saves.MIRROR).write_text(json.dumps({"source": "/projects/x/events"}))
    deep = event_dir / "subcategories" / "k" / saves.REVIEW / saves.SAVES
    assert saves.mirror_root(deep) == events
    with pytest.raises(saves.MirrorReadOnly, match="read-only mirror of /projects/x/events"):
        saves.append(event_dir, "history", {"shot": 7})
    with pytest.raises(saves.MirrorReadOnly), saves.lock(event_dir / saves.REVIEW):
        pass
    with pytest.raises(saves.MirrorReadOnly):
        labels.write_pointer(event_dir, suggestion(tmp_path), method="elm_clock", version="v1")
    assert issubclass(saves.MirrorReadOnly, PermissionError)
    assert sorted(p.name for p in (event_dir / saves.REVIEW).iterdir()) == []


# ---------------------------------------------------------------- pointers travel with the folder


def test_a_pointer_names_a_copy_that_moves_with_the_folder(tmp_path):
    event_dir = event_in(tmp_path / "events")
    table = suggestion(tmp_path, sidecars=True)
    entry = labels.write_pointer(event_dir, table, method="elm_clock", version="v1")
    copies = event_dir / saves.REVIEW / labels.SOURCES / "elm_clock" / "v1"
    assert sorted(p.name for p in copies.iterdir()) == ["t.csv", "t.meta.json", "t_onsets.csv"]
    assert (entry["table"], entry["copied_from"]) == ("sources/elm_clock/v1/t.csv", str(table))
    shutil.move(tmp_path / "events", tmp_path / "moved")
    moved = tmp_path / "moved" / EVENT
    assert labels.source_path(moved) == moved / "review/sources/elm_clock/v1/t.csv"
    assert labels.read_source(moved)[7].intervals == ((300, 400, 1),)


def test_pointing_again_takes_a_rebuilt_table_and_a_copy_is_not_copied(tmp_path):
    event_dir = event_in(tmp_path / "events")
    table = suggestion(tmp_path)
    labels.write_pointer(event_dir, table, method="elm_clock", version="v1")
    rebuilt = TABLE.replace("300,400", "300,450")
    suggestion(tmp_path, rebuilt)
    entry = labels.write_pointer(event_dir, table, method="elm_clock", version="v1")
    copy = labels.source_path(event_dir)
    assert copy.read_text() == rebuilt
    again = labels.write_pointer(event_dir, copy, method="elm_clock", version="v1")
    assert again["table"] == entry["table"] and "copied_from" not in again
    with pytest.raises(ValueError, match="method or version"):
        labels.write_pointer(event_dir, table, method="../x", version="v1")


def test_localize_copies_an_old_pointer_s_table_in_and_keeps_its_fields(tmp_path, capsys):
    events = tmp_path / "events"
    event_dir = event_in(events)
    table = suggestion(tmp_path, sidecars=True)
    old = old_pointer(event_dir, table)
    pointer = labels.pointer_path(event_dir)
    run = ["--events", str(events)]
    assert sync.main(["check", *run]) == 0
    assert "labeler-sync localize copies it in" in capsys.readouterr().out
    assert sync.main(["localize", *run, "--dry-run"]) == 0
    assert json.loads(pointer.read_text()) == old
    assert sync.main(["localize", *run]) == 0
    entry = json.loads(pointer.read_text())
    assert entry["table"] == "sources/elm_clock/v1/t.csv"
    assert entry["copied_from"] == str(table) and entry["copied_at"]
    assert {k: entry[k] for k in old if k != "table"} == {k: v for k, v in old.items()
                                                          if k != "table"}
    assert (labels.source_path(event_dir).read_bytes(), shared.absolute_pointers([event_dir])) == (
        table.read_bytes(), [])
    before = pointer.read_bytes()
    assert sync.main(["localize", *run]) == 0 and pointer.read_bytes() == before
    capsys.readouterr()
    labels.source_path(event_dir).unlink()
    assert sync.main(["check", *run]) == 1
    assert "review pointer" in capsys.readouterr().out


def test_localize_leaves_a_pointer_whose_table_is_gone(tmp_path, capsys):
    events = tmp_path / "events"
    event_dir = event_in(events)
    old = old_pointer(event_dir, tmp_path / "gone.csv")
    assert sync.main(["localize", "--events", str(events)]) == 2
    assert "does not exist" in capsys.readouterr().out
    assert json.loads(labels.pointer_path(event_dir).read_text()) == old


def test_a_copy_keeps_the_suggestion_s_name_in_a_release(tmp_path):
    paths = Paths(root=tmp_path / "root", label_tables=tmp_path / "events")
    copy = tmp_path / "events" / EVENT / "review" / "sources" / "elm_clock" / "v1" / "t.csv"
    original = tmp_path / "root" / "suggestions" / "elm_clock" / "v1" / "t.csv"
    assert hf_export.dataset_id(paths, copy) == "suggestions/elm_clock/v1/t"
    assert hf_export.dataset_id(paths, original) == "suggestions/elm_clock/v1/t"
    review = tmp_path / "events" / EVENT / "review" / "labels.csv"
    assert hf_export.dataset_id(paths, review) == f"{EVENT}/review/labels"


# ---------------------------------------------------------------- switching on


def test_activate_switches_on_only_a_ready_shared_folder(folders, tmp_path, capsys):
    shared_dir, local_dir = folders
    event_dir = event_in(shared_dir)
    old_pointer(event_dir, suggestion(tmp_path))
    assert sync.main(["activate", "--events", str(local_dir)]) == 2
    assert "only the shared folder" in capsys.readouterr().out
    assert sync.main(["activate"]) == 2  # the default is the shared folder
    assert "localize first" in capsys.readouterr().out
    assert sync.main(["localize", "--events", str(shared_dir)]) == 0
    assert sync.main(["activate", "--dry-run"]) == 0
    assert not (shared_dir / env.ACTIVE).exists()
    assert sync.main(["activate"]) == 0
    switch = json.loads((shared_dir / env.ACTIVE).read_text())
    assert switch["mirrors"]["stellar"] == str(local_dir) and switch["by"]
    assert sync.main(["activate"]) == 0
    assert "switched on already" in capsys.readouterr().out
    assert sync.main(["status", "--events", str(shared_dir)]) == 0
    assert "shared folder, switched on" in capsys.readouterr().out


def test_activate_refuses_a_mirror(folders, capsys):
    shared_dir, _ = folders
    (shared_dir / saves.MIRROR).write_text("{}\n")
    assert sync.main(["activate"]) == 2
    assert "is a mirror" in capsys.readouterr().out
    assert not (shared_dir / env.ACTIVE).exists()


# ---------------------------------------------------------------- refreshing the mirror


@needs_rsync
def test_mirror_refreshes_the_cluster_folder_and_keeps_what_was_written_into_it(
    folders, capsys
):
    shared_dir, local_dir = folders
    review = event_in(shared_dir) / saves.REVIEW
    (review / "labels.csv").write_text("shot\n7\n")
    assert sync.main(["mirror"]) == 2  # not switched on
    assert "not switched on" in capsys.readouterr().out
    (shared_dir / env.ACTIVE).write_text("{}\n")
    mine = event_in(local_dir)
    (mine / saves.REVIEW / "labels.csv").write_text("shot\n")  # the folder before the switch
    (mine / "stray.csv").write_text("x\n")
    assert sync.main(["mirror", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "2 files" in out and "would be kept" in out and "import-tree" in out
    assert not (local_dir / saves.MIRROR).exists() and (mine / "stray.csv").exists()

    assert sync.main(["mirror"]) == 0
    capsys.readouterr()
    assert (mine / saves.REVIEW / "labels.csv").read_text() == "shot\n7\n"
    assert not (mine / "stray.csv").exists() and not (local_dir / env.ACTIVE).exists()
    [kept] = (local_dir.parent / "events.replaced").iterdir()
    assert (kept / EVENT / "stray.csv").read_text() == "x\n"
    assert (kept / EVENT / saves.REVIEW / "labels.csv").read_text() == "shot\n"
    marker = json.loads((local_dir / saves.MIRROR).read_text())
    assert (marker["source"], marker["kept"]) == (str(shared_dir), 2)
    with pytest.raises(saves.MirrorReadOnly):
        saves.append(mine, "history", {"shot": 7})

    (review / "labels.csv").write_text("shot\n7\n8\n")  # a save on /projects
    assert sync.main(["mirror"]) == 0
    assert "changed there" not in capsys.readouterr().out
    assert (mine / saves.REVIEW / "labels.csv").read_text() == "shot\n7\n8\n"

    (mine / "job_output.csv").write_text("y\n")  # a job wrote into the mirror
    later = (local_dir / saves.MIRROR).stat().st_mtime + 5
    os.utime(mine / "job_output.csv", (later, later))
    assert sync.main(["mirror"]) == 0
    out = capsys.readouterr().out
    assert "1 files" in out and "job_output.csv" in out
    assert not (mine / "job_output.csv").exists()
    assert [p.read_text() for p in (local_dir.parent / "events.replaced").rglob(
        "job_output.csv")] == ["y\n"]


@needs_rsync
def test_mirror_never_writes_into_a_shared_folder(folders, capsys):
    shared_dir, local_dir = folders
    (shared_dir / env.ACTIVE).write_text("{}\n")
    assert sync.main(["mirror", "--to", str(shared_dir)]) == 2
    assert "is the shared folder itself" in capsys.readouterr().out
    (local_dir / env.ACTIVE).write_text("{}\n")
    assert sync.main(["mirror"]) == 2
    assert "a shared folder, not a mirror" in capsys.readouterr().out


# ---------------------------------------------------------------- the lock across nodes


def _probe(folder, host, rounds):
    """`labeler-sync probe` in its own process, as if on `host`."""
    code = (
        "import socket, sys\n"
        f"socket.gethostname = lambda: {host!r}\n"
        "from labeler.events.review import sync\n"
        f"sys.exit(sync.main(['probe', {str(folder)!r}, '--rounds', '{rounds}', "
        "'--wait', '30', '--pause', '0.002']))\n"
    )
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    return subprocess.Popen([sys.executable, "-c", code], env=environment,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def test_two_nodes_counting_under_the_lock_lose_nothing(tmp_path, capsys):
    folder = tmp_path / "probe"
    folder.mkdir()
    runs = [_probe(folder, host, 40) for host in ("stellar-vis2.x", "stellarai-vis2.x")]
    for run in runs:
        out, _ = run.communicate(timeout=120)
        assert run.returncode == 0, out
    assert sync.main(["probe", str(folder), "--verify"]) == 0
    out = capsys.readouterr().out
    assert "80 rounds by 2 writers" in out and "ok: no update lost" in out

    log = next((folder / saves.REVIEW / saves.SAVES).glob("probe.*.jsonl"))
    first = log.read_text().splitlines()[0]
    with log.open("a") as stream:  # a round that read a stale counter
        stream.write(first + "\n")
    assert sync.main(["probe", str(folder), "--verify"]) == 1
    assert "an update was lost" in capsys.readouterr().out
