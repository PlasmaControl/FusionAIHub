"""`labeler.modes.roster` and the command-line script: `shots.csv` replaced, counts only.

Everything runs on a temporary tree. The shot numbers are synthetic.
"""

from __future__ import annotations

import importlib
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from labeler.modes import roster

from .modes_synth import make_shot, write_shot

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "labeler"
OLD = pd.DataFrame(
    [
        (1, "gold", "true", "A. Reviewer", "2026-01-01", "kept as it is"),
        (2, "unverified", "false", "", "", ""),
        (3, "silver", "false", "B. Reviewer", "2026-02-02", ""),
        (4, "unverified", "true", "", "", "old note"),
    ],
    columns=roster.COLUMNS,
).astype(str)
AUDITED = {
    3: roster.audited_row(3, "silver", "Nathaniel Chen", "2026-11-07", "audited")
}


def test_rows_of_listed_shots_stay_as_they_are_and_the_rest_go():
    new, counts = roster.replace(OLD, [1, 2, 5, 6])
    assert counts == {
        "old": 4,
        "removed": 2,
        "kept": 2,
        "added": 2,
        "audited": 0,
        "audited_new": 0,
        "noted": 0,
        "new": 4,
    }
    assert list(new.shot) == [
        "1",
        "2",
        "5",
        "6",
    ]  # the old file was sorted: so is the new one
    row = new[new.shot == "1"].iloc[0]
    assert (row.tier, row.holdout, row.reviewers, row.notes) == (
        "gold",
        "true",
        "A. Reviewer",
        "kept as it is",
    )
    added = new[new.shot == "5"].iloc[0]
    assert (
        added.tier,
        added.holdout,
        added.reviewers,
        added.verified_on,
        added.notes,
    ) == ("unverified", "false", "", "", "")


def test_the_audited_shot_takes_its_row_and_keeps_a_holdout_flag():
    new, counts = roster.replace(OLD, [1, 2, 3], AUDITED)
    row = new[new.shot == "3"].iloc[0]
    assert (row.tier, row.reviewers, row.verified_on, row.notes) == (
        "silver",
        "Nathaniel Chen",
        "2026-11-07",
        "audited",
    )
    assert counts["audited"] == 1 and counts["audited_new"] == 0
    held = roster.replace(
        OLD, [1, 4], {4: roster.audited_row(4, "silver", "N", "d", "x")}
    )[0]
    assert held[held.shot == "4"].iloc[0].holdout == "true"  # never rewritten


def test_an_audited_shot_outside_the_old_roster_is_added_even_if_not_listed():
    new, counts = roster.replace(
        OLD, [1], {9: roster.audited_row(9, "silver", "N", "d", "x")}
    )
    assert counts["audited_new"] == 1 and set(new.shot) == {"1", "9"}


def test_a_shot_without_a_label_stays_listed_unverified_with_a_note():
    new, counts = roster.replace(
        OLD, [1, 2, 7], notes={2: "no label (labeller failed)", 7: "no label (x)"}
    )
    assert counts["noted"] == 2
    assert new[new.shot == "2"].iloc[0].notes == "no label (labeller failed)"
    assert new[new.shot == "7"].iloc[0].notes == "no label (x)"
    two = roster.replace(OLD, [4], notes={4: "no label (y)"})[0]
    assert two.iloc[0].notes == "old note; no label (y)"


def test_an_unsorted_roster_keeps_its_order():
    old = OLD.iloc[[3, 0, 1]].reset_index(drop=True)
    new, _ = roster.replace(old, [1, 4, 2])
    assert list(new.shot) == ["4", "1", "2"]


def test_the_roster_round_trips_through_csv_and_a_bad_file_is_refused(tmp_path):
    path = tmp_path / "shots.csv"
    roster.write(path, OLD)
    again = roster.read(path)
    assert again.equals(OLD)
    assert path.read_text().splitlines()[0] == ",".join(roster.COLUMNS)
    (tmp_path / "bad.csv").write_text("shot,tier\n1,gold\n")
    with pytest.raises(ValueError):
        roster.read(tmp_path / "bad.csv")
    (tmp_path / "dup.csv").write_text(",".join(roster.COLUMNS) + "\n1,,,,,\n1,,,,,\n")
    with pytest.raises(ValueError):
        roster.read(tmp_path / "dup.csv")
    with pytest.raises(ValueError):
        roster.audited_row(1, "platinum", "", "", "")


def test_a_backup_mirrors_the_path_keeps_the_first_copy_and_refuses_a_git_tree(
    tmp_path,
):
    base = tmp_path / "tokamaksi"
    shots = base / "diiid" / "events" / "mhd_n1" / "shots.csv"
    shots.parent.mkdir(parents=True)
    shots.write_text("original")
    dest = tmp_path / "backup"
    copy = roster.backup(shots, base, dest)
    assert copy == (dest / "diiid" / "events" / "mhd_n1" / "shots.csv").resolve()
    shots.write_text("changed")
    assert (
        roster.backup(shots, base, dest).read_text() == "original"
    )  # the first copy stays
    tree = tmp_path / "work"
    (tree / ".git").mkdir(parents=True)
    with pytest.raises(ValueError):
        roster.backup(shots, base, tree / "backup")
    assert not (tree / "backup").exists()


def test_a_backup_inside_a_git_tree_is_allowed_only_where_git_ignores_it(tmp_path):
    base = tmp_path / "tokamaksi"
    shots = base / "shots.csv"
    base.mkdir()
    shots.write_text("original")
    tree = tmp_path / "work"
    tree.mkdir()
    subprocess.run(["git", "init", "-q", str(tree)], check=True)
    (tree / ".gitignore").write_text("/data/\n")
    with pytest.raises(ValueError):
        roster.backup(shots, base, tree / "tracked")  # a real tree, not ignored
    assert not (tree / "tracked").exists()
    copy = roster.backup(shots, base, tree / "data" / "backup")
    assert copy.read_text() == "original" and (tree / "data").is_dir()


@pytest.fixture
def script(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return importlib.import_module("mode_fdp_label")


def test_the_shot_list_skips_comments_and_blank_lines_and_refuses_a_repeat(
    script, tmp_path
):
    lst = tmp_path / "shots.txt"
    lst.write_text("# header\n\n300\n100  # trailing\n  200\n")
    assert script.read_shots(lst) == [100, 200, 300]
    lst.write_text("100\n100\n")
    with pytest.raises(SystemExit):
        script.read_shots(lst)


def test_the_roster_command_prints_counts_only_and_changes_nothing_on_a_dry_run(
    script, tmp_path, capsys
):
    events, base, labels = (
        tmp_path / "tokamaksi" / "events",
        tmp_path / "tokamaksi",
        tmp_path / "labels",
    )
    for event in script.EVENTS:
        (events / event).mkdir(parents=True)
        roster.write(events / event / "shots.csv", OLD)
    for shot in (1, 2):
        write_shot(labels, make_shot(shot=shot))
    write_shot(labels, make_shot(shot=5, model_error="boom"))
    shots = tmp_path / "shots.txt"
    shots.write_text("# list\n1\n2\n5\n6\n")
    argv = [
        "--labels",
        str(labels),
        "--shots",
        str(shots),
        "roster",
        "--events",
        str(events),
        "--base",
        str(base),
        "--backup",
        str(tmp_path / "backup"),
        "--audited-shot",
        "2",
        "--reviewer",
        "Nathaniel Chen",
        "--verified-on",
        "2026-11-07",
        "--note",
        "audited",
    ]
    script.main([*argv, "--dry-run"])
    out = capsys.readouterr().out
    assert (
        "removed" in out
        and '"kept": 2' in out
        and "1 no label JSON" in out
        and "1 labeller failed" in out
    )
    assert roster.read(events / "mhd_n1" / "shots.csv").equals(OLD)
    assert not (tmp_path / "backup").exists()
    script.main(argv)
    capsys.readouterr()
    for event in script.EVENTS:
        new = roster.read(events / event / "shots.csv")
        assert list(new.shot) == ["1", "2", "5", "6"]
        assert new[new.shot == "2"].iloc[0].tier == "silver"
        assert new[new.shot == "5"].iloc[0].notes == script.FAILED_NOTE
        assert new[new.shot == "6"].iloc[0].notes == script.NO_DATA_NOTE
        assert (tmp_path / "backup" / "events" / event / "shots.csv").is_file()
        assert roster.read(tmp_path / "backup" / "events" / event / "shots.csv").equals(
            OLD
        )


def test_status_prints_counts_only_and_never_a_shot_number(script, tmp_path, capsys):
    labels = tmp_path / "labels"
    write_shot(labels, make_shot(shot=123456))
    write_shot(labels, make_shot(shot=234567, model_error="boom"))
    shots = tmp_path / "shots.txt"
    shots.write_text("123456\n234567\n345678\n")
    script.main(["--labels", str(labels), "--shots", str(shots), "status"])
    out = capsys.readouterr().out
    assert "3 listed; 1 labeled; 2 without a label" in out
    assert not any(str(shot) in out for shot in (123456, 234567, 345678))


def test_suggest_points_every_event_and_refuses_a_class_the_taxonomy_lacks(
    script, tmp_path, capsys
):
    labels, events, root = (
        tmp_path / "labels",
        tmp_path / "events",
        tmp_path / "root",
    )
    write_shot(labels, make_shot(shot=123456))
    shots = tmp_path / "shots.txt"
    shots.write_text("123456\n234567\n")
    argv = ["--labels", str(labels), "--shots", str(shots), "suggest"]
    short = tmp_path / "short.yaml"
    short.write_text(
        "taxonomy:\n  mhd_n1:\n    subcategories:\n      m1: {m: 1, n: 1}\n"
    )
    with pytest.raises(ValueError, match="not in events.yaml"):
        script.main([*argv, "--root", str(root), "--manifest", str(short)])
    assert not root.exists()  # refused before anything was written
    script.main(
        [*argv, "--root", str(root), "--version", "v7", "--events", str(events)]
    )
    out = capsys.readouterr().out
    assert "1 of 2 listed shots written; 1 skipped" in out
    assert "pointer mhd_n2.m4: mhd_modes v7" in out
    assert not any(str(shot) in out for shot in (123456, 234567))
    for event in (*script.EVENTS, "mhd_n1.m2", "mhd_n3.m4"):
        parent, _, key = event.partition(".")
        base = events / parent / "subcategories" / key if key else events / parent
        assert (base / "review" / "source.json").is_file(), event
