"""Named shot lists: files, the registry, membership, evaluation plans and the CLI."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from labeler.config import Paths, sha256_of
from labeler.events import shotlists
from labeler.events.catalog.cohort import COHORT_COLUMNS
from labeler.events.shotlists import ShotListError

BLIND = 175000
GOLD = ("shot,run_day,year,classes,sources,notes\n"
        "170815,20260101,2026,,,\n171234,20260101,2026,,,\n")
SILVER = ("shot,batch,order,pool,classes,sources,notes\n"
          "170815,1,1,corpus,mhd_n1:present,format,\n170816,2,1,corpus,,format,\n")


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _cohort(paths, shots=(1, 2, 3), blind=(BLIND,)):
    """A catalog cohort whose `blind` shots are the test set lists never hold."""
    rows = [{"shot": shot, "queue_rank": rank, "blind": shot in blind}
            for rank, shot in enumerate([*shots, *blind], 1)]
    blank = dict.fromkeys(COHORT_COLUMNS, "")
    frame = pd.DataFrame([blank | row for row in rows], columns=list(COHORT_COLUMNS))
    path = paths.label_tables / "catalog" / "cohort.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


@pytest.fixture
def paths(tmp_path):
    tables = tmp_path / "events"
    tables.mkdir()
    (tmp_path / "config").mkdir()
    return Paths(
        root=tmp_path / "root", corpus=tmp_path / "corpus",
        text_root=tmp_path / "text", logs_jsonl=tmp_path / "logs.jsonl",
        label_tables=tables, config_dir=tmp_path / "config", raw_cache=tmp_path / "raw",
    )


@pytest.fixture
def cli(paths, monkeypatch):
    monkeypatch.setattr(shotlists.Paths, "from_env", lambda: paths)
    return lambda *argv: shotlists.main(list(argv))


def test_a_list_is_read_strictly_and_never_holds_a_blind_shot(paths):
    _cohort(paths)
    _write(paths.shotlists / "mixed.csv", "shot,batch,note\n3,2,b\n1,1,\n")
    frame = shotlists.read(paths, "mixed")
    assert list(frame.columns) == ["shot", "batch", "note"]
    assert frame.shot.tolist() == [3, 1] and str(frame.shot.dtype) == "int64"
    assert frame.batch.tolist() == [2, 1] and frame.note.tolist() == ["b", ""]
    _write(paths.shotlists / "plain.txt", "# made by hand\n5\n\n6  # a note\n")
    assert shotlists.shots(paths, "plain") == [5, 6]
    for name, text, match in (
        ("twice", "shot\n1\n1\n", "duplicate"),
        ("fraction", "shot\n1.5\n", "nonnegative integer"),
        ("word", "shot\nabc\n", "nonnegative integer"),
        ("noshot", "number\n1\n", "`shot` column"),
        ("blinded", f"shot\n1\n{BLIND}\n", "blind"),
    ):
        _write(paths.shotlists / f"{name}.csv", text)
        with pytest.raises(ShotListError, match=match):
            shotlists.read(paths, name)
    _write(paths.shotlists / "badline.txt", "5\nfive\n")
    with pytest.raises(ShotListError, match=r"badline.txt:2"):
        shotlists.read(paths, "badline")
    with pytest.raises(KeyError):
        shotlists.read(paths, "absent")


def test_gold_and_silver_are_lists_without_a_registry_and_keep_their_files(paths):
    _write(paths.label_tables / "gold/gold_shots.csv", GOLD)
    _write(paths.shotlists / "zeta.txt", "1\n")
    assert shotlists.names(paths) == ["gold", "silver", "zeta"]
    gold = shotlists.entry(paths, "gold")
    assert gold.path == paths.label_tables / "gold/gold_shots.csv"
    assert (gold.title, gold.protect, gold.registered) == ("Gold set", "test", True)
    zeta = shotlists.entry(paths, "zeta")
    assert not zeta.registered and zeta.title == "zeta"
    # An entry that only retitles gold keeps its file and its protection.
    _write(paths.shotlists_manifest,
           "lists:\n  zeta: {title: Z}\n  gold: {title: Gold}\n")
    assert shotlists.names(paths) == ["zeta", "gold", "silver"]
    gold = shotlists.entry(paths, "gold")
    assert (gold.title, gold.protect) == ("Gold", "test")
    assert gold.path == paths.label_tables / "gold/gold_shots.csv"
    assert shotlists.read(paths, "gold").shot.tolist() == [170815, 171234]


@pytest.mark.parametrize(("raw", "match"), [
    ({"lists": {"blind": {}}}, "reserved"),
    ({"lists": {"Bad-Name": {}}}, "plain list name"),
    ({"lists": {"a": {"colour": "red"}}}, "unknown keys"),
    ({"lists": {"a": {"parent": "b"}}}, "parent 'b' is not a list"),
    ({"lists": {"a": {"file": "../outside.csv"}}}, "under the label"),
    ({"lists": {"a": {"protect": "forever"}}}, "protect"),
    ({"events": {"elm": {"default": "nope"}}}, "default"),
    ({"events": {"elm": {"recommended": ["nope"]}}}, "registered lists"),
    ({"events": {"Elm!": {}}}, "not an event id"),
    ({"lists": {"a": {"on_demand": "yes"}}}, "on_demand must be true or false"),
    ({"renamed": ["old"]}, "must be mappings"),
    ({"renamed": {"Bad-Name": "a"}, "lists": {"a": {}}}, "plain list name"),
    ({"renamed": {"old": "nope"}}, "'nope' is not a registered list"),
    ({"renamed": {"old": {"elm": "a"}}, "lists": {"a": {}}}, "needs a `default`"),
    ({"renamed": {"old": {"Elm!": "a", "default": "a"}}, "lists": {"a": {}}},
     "not an event id"),
])
def test_the_registry_names_every_mistake(raw, match):
    with pytest.raises(ShotListError, match=match):
        shotlists.parse(raw)


def test_an_event_takes_its_parents_lists_else_silver_and_gold(paths):
    _write(paths.shotlists_manifest,
           "lists:\n  mine: {}\nevents:\n"
           "  mhd_n1: {recommended: [mine, gold], default: mine, evaluate: [gold]}\n")
    assert shotlists.event_lists(paths, "mhd_n1.m1") == shotlists.EventLists(
        ("mine", "gold"), "mine", ("gold",))
    assert shotlists.event_lists(paths, "detachment") == shotlists.EventLists(
        ("silver", "gold"), shotlists.ROSTER, ())


def test_offered_puts_the_recommended_first_then_each_child_after_its_parent(paths):
    _write(paths.shotlists_manifest,
           "lists:\n  small: {parent: big}\n  big: {}\n  batch: {parent: silver}\n"
           "  other: {}\nevents:\n  elm: {recommended: [other, gold]}\n")
    _write(paths.label_tables / "gold/gold_shots.csv", GOLD)
    _write(paths.label_tables / "silver/silver_shots.csv", SILVER)
    for name in ("small", "big", "batch", "other"):
        _write(paths.shotlists / f"{name}.txt", "170815\n")
    _write(paths.shotlists / "loose.txt", "1\n1\n")  # unreadable: never offered
    found = [(item.name, recommended) for item, recommended
             in shotlists.offered(paths, "elm")]
    assert found == [("other", True), ("gold", True), ("big", False),
                     ("small", False), ("silver", False), ("batch", False)]
    # A child of a recommended list opens the others, as near its parent as it can.
    found = [item.name for item, _ in shotlists.offered(paths, "detachment")]
    assert found == ["silver", "gold", "batch", "big", "small", "other"]


def test_held_out_names_holdout_lists_shots_and_a_long_name_is_no_list(paths):
    _write(paths.shotlists_manifest, "lists:\n  key: {protect: holdout}\n")
    _write(paths.shotlists / "key.txt", "1\n")
    _write(paths.shotlists / "other.txt", "2\n")
    late = _write(paths.shotlists / "late.txt", "3\n")
    _write(shotlists.sidecar_path(late), json.dumps({"protect": "holdout"}))
    assert shotlists.held_out(paths, 1) and shotlists.held_out(paths, 3)
    assert not shotlists.held_out(paths, 2) and not shotlists.held_out(paths, 4)
    _write(paths.shotlists / "key.txt", "1\n1\n")  # unreadable: fails closed
    assert shotlists.held_out(paths, 2) and shotlists.held_out(paths, 4)
    with pytest.raises(KeyError):
        shotlists.entry(paths, "a" * 300)
    with pytest.raises(ShotListError, match="not a plain list name"):
        shotlists.check_name("a" * 300)


def test_membership_follows_every_file_and_holding_names_registered_lists(paths):
    _cohort(paths)
    _write(paths.label_tables / "gold/gold_shots.csv", GOLD)
    _write(paths.shotlists_manifest, "lists:\n  mine: {}\n")
    listed = _write(paths.shotlists / "mine.txt", "1\n170815\n")
    _write(paths.shotlists / "loose.txt", "2\n")
    found = shotlists.members(paths)
    assert found == {"gold": {170815, 171234}, "mine": {1, 170815}, "loose": {2}}
    assert shotlists.members(paths) is found, "cached while nothing changes"
    assert shotlists.in_any(paths, 2) and not shotlists.in_any(paths, 3)
    assert shotlists.holding(paths, 170815) == ["mine", "gold"]
    assert shotlists.holding(paths, 2) == [], "an unregistered list holds nothing"
    listed.write_text("3\n")
    assert shotlists.in_any(paths, 3) and shotlists.holding(paths, 1) == []
    _write(paths.shotlists / "late.txt", "4\n")
    assert shotlists.in_any(paths, 4), "a new file is a list at once"
    _write(paths.shotlists / "loose.txt", f"2\n{BLIND}\n")
    assert not shotlists.in_any(paths, BLIND) and not shotlists.in_any(paths, 2)


def test_plan_scores_a_list_and_names_every_shot_it_does_not_score(paths):
    listed = _write(paths.shotlists / "evalset.txt", "1\n2\n3\n4\n5\n")
    scored, sets = shotlists.plan(paths, ["evalset"], population={1, 2, 3, 4},
                                  trained_on={2}, reference={1, 2, 4})
    assert scored == {"evalset": [1, 4]}
    assert sets["evalset"] == {
        "list": "evalset", "title": "evalset", "list_sha256": sha256_of(listed),
        "n_listed": 5, "n_scored": 2,
        "missing": {"no_input": [5], "not_in_cv": [], "no_reference": [3]},
        "trained_on": [2], "scoring": "heldout", "leaky": False,
    }
    scored, sets = shotlists.plan(paths, ["evalset"], {1, 2, 3, 4}, {2}, {1, 2, 4},
                                  include_trained=True, scoring="oof")
    assert scored == {"evalset": [1, 2, 4]} and sets["evalset"]["leaky"]
    assert sets["evalset"]["missing"]["not_in_cv"] == [5]
    with pytest.raises(ValueError, match="scoring"):
        shotlists.plan(paths, ["evalset"], set(), set(), set(), scoring="all")


def test_make_writes_a_list_and_its_sidecar_and_refuses_what_it_must(
    paths, cli, capsys,
):
    assert cli("make", "mine", "--shots", "1") == 2, "no catalog: fails closed"
    assert "blind test shots are unknown" in capsys.readouterr().err
    _cohort(paths)
    _write(paths.shotlists_manifest,
           "lists:\n  held: {title: Held, protect: holdout}\n")
    assert cli("make", "mine", "--shots", "3", "1", "2", "1", "--title", "Mine") == 0
    path = paths.shotlists / "mine.csv"
    assert path.read_text() == "shot\n3\n1\n2\n"
    meta = json.loads((paths.shotlists / "mine.meta.json").read_text())
    assert (meta["title"], meta["kind"], meta["rule"]) == ("Mine", "derived", "shots")
    assert meta["n_shots"] == 3 and meta["sha256"] == sha256_of(path)
    assert {"made_at", "made_by", "git", "derived_from"} <= set(meta)
    assert sorted(p.name for p in paths.shotlists.iterdir()) == [
        "mine.csv", "mine.meta.json"], "written by rename: no temporary file is left"
    assert cli("make", "risky", "--shots", "5", str(BLIND)) == 2
    assert "blind test shot" in capsys.readouterr().err
    assert cli("make", "risky", "--shots", "5", str(BLIND), "--drop-blind") == 0
    risky = json.loads((paths.shotlists / "risky.meta.json").read_text())
    assert risky["n_shots"] == 1 and risky["blind_dropped"] == 1
    for name in ("blind", "roster", "gold"):
        assert cli("make", name, "--shots", "170900") == 2, name
    assert cli("make", "held", "--shots", "5") == 0
    assert cli("make", "held", "--shots", "6") == 2
    assert "protected (holdout)" in capsys.readouterr().err
    assert (paths.shotlists / "held.csv").read_text() == "shot\n5\n"


def test_derived_lists_keep_columns_record_inputs_and_go_stale(paths, cli, capsys):
    _cohort(paths)
    _write(paths.label_tables / "gold/gold_shots.csv", GOLD)
    _write(paths.label_tables / "silver/silver_shots.csv", SILVER)
    base = _write(paths.shotlists / "base.csv", "shot,batch\n1,1\n2,2\n3,2\n")
    assert cli("make", "base_2", "--from", "base", "--where", "batch==2") == 0
    assert (paths.shotlists / "base_2.csv").read_text() == "shot,batch\n2,2\n3,2\n"
    meta = json.loads((paths.shotlists / "base_2.meta.json").read_text())
    assert meta["rule"] == "where(base, batch==2)"
    assert meta["derived_from"] == {"base": {"sha256": sha256_of(base)}}
    hand = _write(paths.root / "hand.txt", "3\n7\n")
    assert cli("make", "hand", "--from-txt", str(hand)) == 0
    assert cli("make", "both", "--union", "base_2,hand") == 0
    assert shotlists.shots(paths, "both") == [2, 3, 7]
    capsys.readouterr()
    assert cli("check") == 0
    assert capsys.readouterr().out.splitlines()[-1] == "6 lists, 0 problems"
    base.write_text("shot,batch\n1,1\n2,2\n3,2\n4,2\n")
    _write(paths.shotlists / "Bad-Name.txt", "1\n")
    assert cli("check") == 1
    out = capsys.readouterr().out
    assert "PROBLEM: base_2: stale, base changed" in out
    assert "not a usable list name" in out and "base: no sidecar" in out


RENAMES = (
    "lists:\n  new: {title: New}\n  qmin: {}\n  beta: {}\n"
    "renamed:\n  old: new\n  cohort: {minimum_safety_factor: qmin, poloidal_beta: beta,"
    " default: qmin}\n"
)


def test_an_old_name_resolves_to_its_list_and_a_list_still_wins(paths):
    _write(paths.shotlists_manifest, RENAMES)
    for name in ("new", "qmin", "beta"):
        _write(paths.shotlists / f"{name}.txt", "1\n2\n")
    assert shotlists.resolve(paths, "old") == "new"
    assert shotlists.resolve(paths, "new") == "new", "a list resolves to itself"
    assert shotlists.resolve(paths, "nothing") == "nothing"
    assert shotlists.resolve(paths, "cohort") == "qmin", "no event: the default"
    assert shotlists.resolve(paths, "cohort", "poloidal_beta") == "beta"
    assert shotlists.resolve(paths, "cohort", "minimum_safety_factor.q") == "qmin"
    assert shotlists.resolve(paths, "cohort", "locked_mode") == "qmin"
    assert shotlists.resolve(paths, 7) == 7
    found = shotlists.entry(paths, "old")
    assert found.name == "new" and found.title == "New"
    assert shotlists.entry(paths, "cohort", "poloidal_beta").name == "beta"
    assert shotlists.read(paths, "old").shot.tolist() == [1, 2]
    assert shotlists.shots(paths, "cohort") == [1, 2]
    with pytest.raises(KeyError):
        shotlists.entry(paths, "never_was")
    # A leftover file under an old name is no list of its own; the registry's name wins.
    _write(paths.shotlists / "old.txt", "9\n")
    assert "old" not in shotlists.names(paths)
    assert shotlists.shots(paths, "old") == [1, 2]
    _write(paths.shotlists_manifest, RENAMES + "  nolist: new\n")
    assert shotlists.resolve(paths, "nolist") == "new"
    _write(paths.shotlists_manifest,
           RENAMES.replace("lists:\n", "lists:\n  old: {title: Old}\n", 1))
    assert shotlists.resolve(paths, "old") == "old", "still a list: it wins"


def test_the_on_demand_flag_comes_from_the_registry(paths):
    assert shotlists.on_demand(paths) == frozenset()
    _write(paths.shotlists_manifest,
           "lists:\n  far: {on_demand: true}\n  near: {on_demand: false}\n  also: {}\n")
    assert shotlists.on_demand(paths) == frozenset({"far"})
    assert shotlists.entry(paths, "far").on_demand
    assert not shotlists.entry(paths, "near").on_demand
    _write(paths.shotlists_manifest, "lists: [")  # unreadable: nothing is flagged
    assert shotlists.on_demand(paths) == frozenset()


def test_plan_answers_under_the_lists_own_name(paths):
    _write(paths.shotlists_manifest, RENAMES)
    _write(paths.shotlists / "new.txt", "1\n2\n")
    scored, sets = shotlists.plan(paths, ["old"], {1, 2}, set(), {1, 2})
    assert scored == {"new": [1, 2]} and sets["new"]["list"] == "new"


def test_make_refuses_an_old_name_but_reads_it_as_an_input(paths, cli, capsys):
    _cohort(paths)
    _write(paths.shotlists_manifest, RENAMES)
    assert cli("make", "new", "--shots", "1", "2") == 0
    capsys.readouterr()
    assert cli("make", "old", "--shots", "5") == 2
    assert "old name, renamed to new: make that" in capsys.readouterr().err
    assert cli("make", "cohort", "--shots", "5") == 2
    assert "renamed to beta, qmin" in capsys.readouterr().err
    assert cli("make", "both", "--union", "old,qmin") != 0, "qmin is not made yet"
    capsys.readouterr()
    assert cli("make", "qmin", "--shots", "2", "3") == 0
    assert cli("make", "both", "--union", "old,qmin") == 0
    meta = json.loads((paths.shotlists / "both.meta.json").read_text())
    assert meta["rule"] == "union(new, qmin)", "the rule names the lists, not the alias"
    assert set(meta["derived_from"]) == {"new", "qmin"}
    assert shotlists.shots(paths, "both") == [1, 2, 3]
    assert cli("make", "copy", "--from", "old") == 0
    assert json.loads((paths.shotlists / "copy.meta.json").read_text())["rule"] == "new"


def test_a_union_runs_over_tables_and_text_files_too(paths, cli, capsys):
    _cohort(paths)
    _write(paths.label_tables / "gold/gold_shots.csv", GOLD)
    _write(paths.label_tables / "silver/silver_shots.csv", SILVER)
    table = _write(paths.label_tables / "events_a/raw/table.csv",
                   "regime,Shot\nL,5\nH,6\nH,6\n")
    hand = _write(paths.root / "raw_external/hand.txt", "# by hand\n6\n7\n")
    assert cli("make", "mixed", "--union", f"table({table}),txt({hand})",
               "--column", "Shot") == 0
    assert shotlists.shots(paths, "mixed") == [5, 6, 7]
    meta = json.loads((paths.shotlists / "mixed.meta.json").read_text())
    assert meta["rule"] == (
        "union(table(events_a/raw/table.csv), txt($LABELER_ROOT/raw_external/hand.txt))")
    assert set(meta["derived_from"]) == {
        "events_a/raw/table.csv", "$LABELER_ROOT/raw_external/hand.txt"}
    _write(paths.shotlists_manifest,
           f"lists:\n  mixed: {{rule: \"{meta['rule']}\"}}\n")
    capsys.readouterr()
    assert cli("check") == 0
    hand.write_text("6\n8\n")
    assert cli("check") == 1
    assert "mixed: stale, $LABELER_ROOT/raw_external/hand.txt changed" in (
        capsys.readouterr().out)
    assert cli("make", "single", "--union", f"txt({hand})") == 2
    assert "two or more terms" in capsys.readouterr().err


def test_check_reports_a_renamed_name_that_is_also_a_list(paths, cli, capsys):
    _write(paths.label_tables / "gold/gold_shots.csv", GOLD)
    _write(paths.label_tables / "silver/silver_shots.csv", SILVER)
    _write(paths.shotlists_manifest, RENAMES)
    _write(paths.shotlists / "new.txt", "1\n")
    _write(paths.shotlists / "qmin.txt", "1\n")
    _write(paths.shotlists / "beta.txt", "1\n")
    assert cli("check") == 0
    capsys.readouterr()
    _write(paths.shotlists / "cohort.csv", "shot\n1\n")  # a file left behind
    assert cli("check") == 1
    out = capsys.readouterr().out
    assert ("PROBLEM: cohort: renamed to beta, qmin, but it is also a file under "
            "shotlists/") in out
    (paths.shotlists / "cohort.csv").unlink()
    _write(paths.shotlists_manifest,
           RENAMES.replace("lists:\n", "lists:\n  old: {title: Old}\n", 1))
    _write(paths.shotlists / "old.txt", "1\n")
    assert cli("check") == 1
    assert "old: renamed to new, but it is also a list of the registry" in (
        capsys.readouterr().out)
