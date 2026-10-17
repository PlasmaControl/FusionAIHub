"""Jalal Butt's WPQH phases as confinement labels, and his saved name."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from labeler.events.review import labels, reviewers

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/labeler/jalal_wpqh_import.py"
spec = importlib.util.spec_from_file_location("jalal_wpqh_import", SCRIPT)
jw = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = jw  # the script's dataclasses look their module up
spec.loader.exec_module(jw)

CATEGORIES = set(labels.categories("confinement"))
#: Low 0-2000, high 2000-5000, and a WPQH stretch the draft has at 4000-4500.
BASE = {
    "window": [0, 6000],
    "intervals": [[0, 2000, 2], [2000, 4000, 1], [4000, 4500, 4], [4500, 5000, 1]],
    "iscrowd": [1, 1, 1, 1],
}


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


# ---- the archive ----


def test_the_archive_reads_phases_per_shot(tmp_path):
    write(tmp_path / "100-wpqh1.dat", "ON OFF\n2000.000 2152.000\n2236.000 2329.000\n")
    write(tmp_path / "101-wpqh0.dat", "ON OFF\n")
    write(tmp_path / "102-wpqh0.dat", "ON OFF\n2000.000 2246.000\n")
    write(tmp_path / "103-wpqh0.dat", "ONs OFFs\n0.0 0.0\n")
    write(tmp_path / "100_wpqh_may262026.pkl", "not read")
    write(tmp_path / "postprocessed_2" / "104-wpqh1-pp.dat", "ON OFF\n1 2\n")
    found = jw.read_archive(tmp_path, final=False)
    assert sorted(found) == [100, 101, 102, 103]
    assert found[100].flag == 1
    assert found[100].wpqh == ((2000.0, 2152.0), (2236.0, 2329.0))
    assert found[101].flag == 0 and found[101].spans == ()
    # A wpqh0 file's blocks are too short to count: they are not WPQH.
    assert found[102].spans == ((2000.0, 2246.0),) and found[102].wpqh == ()
    assert found[103].spans == ()


def test_wpqh1_wins_when_a_shot_has_both_files(tmp_path):
    write(tmp_path / "105-wpqh0.dat", "ON OFF\n2505.000 2857.000\n")
    write(tmp_path / "105-wpqh1.dat", "ON OFF\n2505.000 3438.000\n")
    phases = jw.read_archive(tmp_path, final=False)[105]
    assert phases.both and phases.flag == 1 and phases.wpqh == ((2505.0, 3438.0),)


def test_a_phase_that_runs_backwards_is_refused(tmp_path):
    write(tmp_path / "106-wpqh1.dat", "ON OFF\n3000.0 2000.0\n")
    with pytest.raises(ValueError, match="runs backwards"):
        jw.read_archive(tmp_path, final=False)


def test_the_final_version_is_read_and_the_detector_fills_what_it_lacks(tmp_path):
    write(tmp_path / "100-wpqh1.dat", "ON OFF\n2000.0 2152.0\n2236.0 2329.0\n")
    write(tmp_path / "100_wpqh_may262026.pkl", "not read")
    write(tmp_path / "postprocessed_2" / "100-wpqh1-pp.dat", "ON OFF\n2000.0 2329.0\n")
    write(tmp_path / "101-wpqh0.dat", "ON OFF\n")
    write(tmp_path / "postprocessed_2" / "101-wpqh1-pp.dat", "ON OFF\n2000.0 2300.0\n")
    write(tmp_path / "102-wpqh1.dat", "ON OFF\n1530.0 3831.0\n")  # not post-processed
    write(tmp_path / "postprocessed_2" / "postprocess_summary.dat", "# summary\n")
    final = jw.read_archive(tmp_path)
    assert sorted(final) == [100, 101, 102]
    assert final[100].wpqh == ((2000.0, 2329.0),)
    assert final[100].file == "postprocessed_2/100-wpqh1-pp.dat"
    assert final[101].flag == 1  # the later version flips a no-WPQH shot
    assert final[102].wpqh == ((1530.0, 3831.0),) and final[102].file == "102-wpqh1.dat"
    raw = jw.read_archive(tmp_path, final=False)
    assert raw[100].wpqh == ((2000.0, 2152.0), (2236.0, 2329.0))
    assert raw[101].flag == 0


# ---- the label ----


def test_class_4_is_replaced_and_every_other_class_kept():
    label = jw.with_wpqh(jw.label_of(BASE), [(2100.4, 3000.6)], known=CATEGORIES)
    assert label.window == (0, 6000)
    assert label.intervals == (
        (0, 2000, 2),
        (2000, 2100, 1),
        (2100, 3000, 4),
        (3000, 4000, 1),
        (4500, 5000, 1),
    )
    # The draft's own 4000-4500 WPQH is not Jalal's: that time is left unlabelled.
    assert label.iscrowd == (1,) * len(label.intervals)


def test_a_wpqh0_shot_loses_its_class_4():
    label = jw.with_wpqh(jw.label_of(BASE), (), known=CATEGORIES)
    assert [c for _, _, c in label.intervals] == [2, 1, 1]
    assert jw.spans_ms(label) == 0


def test_no_label_to_start_from():
    label = jw.with_wpqh(None, [(2000.0, 2152.0)], known=CATEGORIES)
    assert label.window == (1000, 5000)
    assert label.intervals == ((2000, 2150, 4),)
    assert label.iscrowd == (1,)
    assert jw.with_wpqh(None, (), known=CATEGORIES) is None


def test_the_window_grows_over_a_phase_past_it():
    label = jw.with_wpqh(
        jw.label_of({"window": [0, 3000], "intervals": []}), [(2900.0, 3500.0)]
    )
    assert label.window == (0, 3500)


# ---- the saved name ----


def line(shot, name, **more):
    return json.dumps({"shot": shot, "reviewer": "nc1514", "name": name, **more}) + "\n"


def test_the_saved_name_is_renamed_exactly(tmp_path):
    tables = tmp_path
    confinement = write(
        tables / "confinement" / "review" / "history.jsonl",
        line(1, "Jalal")
        + line(2, "Nathaniel Chen")
        + line(3, None)
        + line(4, "Azarakhsh Jalalvand")
        + line(5, "Jalal Butt")
        + "not json\n",
    )
    masks = write(
        tables / "alfven_eigenmode" / "review" / "masks.jsonl", line(7, "Jalal")
    )
    child = write(
        tables / "mhd_n1" / "subcategories" / "m1" / "review" / "history.jsonl",
        line(8, "Jalal") + line(9, "Jalal"),
    )
    other = write(
        tables / "edge_localized_mode" / "review" / "history.jsonl",
        line(10, "Alvin Garcia"),
    )
    names = write(tables / "reviewers.txt", "Alvin Garcia\nJalal\nJalalvand\n")
    before = {p: p.read_text() for p in (confinement, masks, child, other, names)}

    counted = jw.rename_saved_names(tables, apply=False)
    assert counted == {
        "alfven_eigenmode/review/masks.jsonl": 1,
        "confinement/review/history.jsonl": 1,
        "mhd_n1/subcategories/m1/review/history.jsonl": 2,
        "reviewers.txt": 1,
    }
    assert {p: p.read_text() for p in before} == before  # a dry run writes nothing

    assert jw.rename_saved_names(tables, apply=True) == counted
    got = confinement.read_text().splitlines(keepends=True)
    assert json.loads(got[0]) == {"shot": 1, "reviewer": "nc1514", "name": "Jalal Butt"}
    assert got[1:] == before[confinement].splitlines(keepends=True)[1:]
    assert other.read_text() == before[other]
    assert names.read_text() == "Alvin Garcia\nJalal Butt\nJalalvand\n"
    assert reviewers.read(tables) == ["Alvin Garcia", "Jalal Butt", "Jalalvand"]
    assert jw.rename_saved_names(tables, apply=False) == {}  # nothing left to rename


def test_the_names_list_keeps_one_jalal_butt(tmp_path):
    write(tmp_path / "reviewers.txt", "Jalal Butt\nJalal\n")
    jw.rename_saved_names(tmp_path, apply=True)
    assert (tmp_path / "reviewers.txt").read_text() == "Jalal Butt\n"


# ---- the save ----


def phases(flag=1, spans=((2000.0, 2500.0),), shot=200):
    return jw.Phases(shot, flag, tuple(spans), file=f"{shot}-wpqh{flag}.dat")


def run(tmp_path, phases, reviewed=None, names=(), **options):
    options.setdefault("apply", True)
    return jw.import_shot(
        phases, reviewed or {}, list(names), tmp_path, known=CATEGORIES, **options
    )


def test_a_phase_is_saved_as_the_page_saves_it_under_jalal_butt(tmp_path):
    table = write(tmp_path / "format" / "draft.csv", "shot,category,t_start,t_end\n")
    row = run(tmp_path, phases(), {"source": BASE}, table=table, origin="suggest:v1")
    assert row.action == "save" and row.base == "draft:suggest:v1"
    assert row.before_ms == 500 and row.after_ms == 500
    entry = labels.read_history(tmp_path)[-1]
    assert entry["name"] == "Jalal Butt" and entry["verified"] is True
    assert entry["note"] == "WPQHphasesMay262026"
    assert entry["source"] == "suggest:v1" and entry["source_sha256"]
    assert entry["suggestions_shown"] is None and entry["prefilled"] is None
    assert entry["wpqh_flag"] == 1 and entry["iscrowd"] == [1] * len(entry["intervals"])
    saved = labels.read_saved(tmp_path, grid_ms=10)[200]
    assert jw.class4(saved) == ((2000, 2500),)
    assert reviewers.shot_reviewers(tmp_path, 200)[0]["name"] == "Jalal Butt"


def test_a_dry_run_writes_nothing(tmp_path):
    row = run(tmp_path, phases(), {"source": BASE}, apply=False)
    assert row.action == "save"
    assert not labels.labels_path(tmp_path).exists()
    assert labels.read_history(tmp_path) == []


def test_a_label_saved_by_someone_else_is_a_conflict(tmp_path):
    saved = {"window": [0, 6000], "intervals": [[3000, 3500, 4]], "iscrowd": [1]}
    row = run(tmp_path, phases(), {"saved": saved}, names=["Nathaniel Chen"])
    assert row.action == "conflict" and "Nathaniel Chen" in row.reason
    assert row.existing == "3000-3500" and row.jalal == "2000-2500"
    assert labels.read_history(tmp_path) == []
    # An unnamed save is no one's either.
    assert run(tmp_path, phases(), {"saved": saved}, names=[None]).action == "conflict"


def test_a_label_saved_by_jalal_is_his_to_replace(tmp_path):
    saved = {"window": [0, 6000], "intervals": [[3000, 3500, 4]], "iscrowd": [1]}
    for name in ("Jalal", "Jalal Butt"):
        row = run(tmp_path, phases(), {"saved": saved}, names=[name], apply=False)
        assert row.action == "save" and row.base == "saved"
    assert row.before_ms == 500


def test_a_wpqh0_shot_with_no_label_is_not_saved_unless_asked(tmp_path):
    none = phases(0, ())
    row = run(tmp_path, none)
    assert row.action == "skip_no_label" and not labels.labels_path(tmp_path).exists()
    row = run(tmp_path, none, save_empty=True)
    assert row.action == "save"
    assert labels.read_saved(tmp_path, grid_ms=10)[200].intervals == ()
    assert labels.read_history(tmp_path)[-1]["window"] == [1000, 5000]


def test_a_wpqh0_shot_with_a_draft_keeps_the_draft_less_its_class_4(tmp_path):
    row = run(tmp_path, phases(0, ((2000.0, 2246.0),)), {"source": BASE})
    assert row.action == "save" and row.before_ms == 500 and row.after_ms == 0
    assert [
        c for _, _, c in labels.read_saved(tmp_path, grid_ms=10)[200].intervals
    ] == [2, 1, 1]


def test_a_shot_off_every_list_is_not_reviewable(tmp_path):
    row = run(tmp_path, phases(), reviewable=False)
    assert row.action == "not_reviewable" and labels.read_history(tmp_path) == []


def test_the_same_import_twice_saves_once(tmp_path):
    reviewed = {"source": BASE}
    first = run(tmp_path, phases(), reviewed)
    again = run(
        tmp_path,
        phases(),
        {
            "source": BASE,
            "saved": labels.page_label(
                "confinement", labels.read_saved(tmp_path, grid_ms=10)[200]
            ).as_json(),
        },
        names=["Jalal Butt"],
        last_save=labels.read_history(tmp_path)[-1],
    )
    assert first.action == "save" and again.action == "already"
    assert len(labels.read_history(tmp_path)) == 1


def test_a_refused_save_is_reported_not_raised(tmp_path):
    write(
        labels.labels_path(tmp_path),
        "shot,category,t_start,t_end,confidence,attrs\n"
        '200,1,2000,2500,,"{""evidence_tier"": ""x""}"\n',
    )
    row = run(tmp_path, phases(), {"source": BASE})
    assert row.action == "refused" and "cannot edit" in row.reason
    assert labels.read_history(tmp_path) == []


# ---- human starts only ----


def test_a_model_draft_is_not_a_start(tmp_path):
    row = run(tmp_path, phases(), {"source": BASE}, human=False, origin="suggest:x/v2")
    assert row.action == "save" and row.base == "none" and row.dropped == "suggest:x/v2"
    saved = labels.read_saved(tmp_path, grid_ms=10)[200]
    assert saved.window == (1000, 5000) and saved.intervals == ((2000, 2500, 4),)
    entry = labels.read_history(tmp_path)[-1]
    assert entry["base"] == "none" and entry["dropped_draft"] == "suggest:x/v2"
    # wpqh0 on a model draft has nothing of his to say: no save.
    shut = run(tmp_path, phases(0, (), shot=201), {"source": BASE}, human=False)
    assert shut.action == "skip_no_label" and shut.before_ms == 500


def test_a_human_draft_is_a_start(tmp_path):
    row = run(tmp_path, phases(), {"source": BASE}, human=True, origin="format:x/y")
    assert row.base == "draft:format:x/y" and row.dropped == ""
    classes = [c for _, _, c in labels.read_saved(tmp_path, grid_ms=10)[200].intervals]
    assert classes == [2, 4, 1, 1]


def test_a_save_of_this_import_is_not_a_start(tmp_path):
    """Its other classes came from a draft: the shot starts again from its draft."""
    run(tmp_path, phases(), {"source": BASE})
    last = labels.read_history(tmp_path)[-1]
    derived = labels.page_label(
        "confinement", labels.read_saved(tmp_path, grid_ms=10)[200]
    ).as_json()
    both = {"source": BASE, "saved": derived}
    again = run(tmp_path, phases(), both, names=["Jalal Butt"], last_save=last)
    assert again.action == "already" and again.base == "draft:source"
    # The draft is now a model's: the derived label is replaced by his spans alone.
    redone = run(
        tmp_path, phases(), both, names=["Jalal Butt"], last_save=last, human=False
    )
    assert redone.action == "save" and redone.base == "none"
    assert labels.read_saved(tmp_path, grid_ms=10)[200].intervals == ((2000, 2500, 4),)


def test_a_label_he_saved_himself_is_a_start(tmp_path):
    saved = {"window": [0, 6000], "intervals": [[3000, 3500, 2]], "iscrowd": [1]}
    last = {"shot": 200, "name": "Jalal Butt", "note": "edited on the page"}
    row = run(
        tmp_path, phases(), {"saved": saved}, names=["Jalal Butt"], last_save=last,
        human=False, apply=False,
    )
    assert row.action == "save" and row.base == "saved"
    assert row.dropped == ""


def test_a_wpqh0_shot_with_only_a_model_start_loses_this_imports_save(tmp_path):
    other = phases(shot=300)
    run(tmp_path, other, {"source": BASE})
    run(tmp_path, phases(shot=200), {"source": BASE})
    last = labels.read_history(tmp_path)[-1]
    derived = labels.page_label(
        "confinement", labels.read_saved(tmp_path, grid_ms=10)[200]
    ).as_json()
    kept = (tmp_path / "review" / "history.jsonl").read_text().splitlines()[0]
    args = ({"source": BASE, "saved": derived},)
    kw = {"names": ["Jalal Butt"], "last_save": last, "human": False}
    dry = run(tmp_path, phases(0, (), shot=200), *args, apply=False, **kw)
    assert dry.action == "withdraw" and 200 in labels.read_saved(tmp_path)
    done = run(tmp_path, phases(0, (), shot=200), *args, **kw)
    assert done.action == "withdraw"
    assert sorted(labels.read_saved(tmp_path)) == [300]
    assert [e["shot"] for e in labels.read_history(tmp_path)] == [300]
    assert (tmp_path / "review" / "history.jsonl").read_text().splitlines() == [kept]
    assert reviewers.shot_reviewers(tmp_path, 200) == []


def test_a_withdrawal_refuses_a_shot_someone_else_saved(tmp_path):
    run(tmp_path, phases(), {"source": BASE})
    labels.save(
        tmp_path,
        200,
        jw.label_of(BASE),
        source=None,
        name="Nathaniel Chen",
        crowd_edit=True,
    )
    with pytest.raises(ValueError, match="not this import's"):
        jw.withdraw(tmp_path, 200)
    assert 200 in labels.read_saved(tmp_path)


def test_which_drafts_are_a_persons(tmp_path):
    curated = {10, 11}
    assert jw.human_draft(None, 10, curated) and not jw.human_draft(None, 12, curated)
    assert jw.human_draft(None, 12, None)  # no pointer: a format table
    assert jw.human_draft("format:high_confinement_mode/x", 12, curated)
    assert not jw.human_draft("suggest:hmode_frames/v2/x", 10, curated)
    assert not jw.human_draft("extend:confinement/extend_x/roster", 10, None)


def test_curated_shots_come_from_the_source_tables_sidecar(tmp_path):
    review = tmp_path / "review"
    assert jw.curated_shots(tmp_path) is None  # no pointer
    table = write(tmp_path / "suggest.csv", "shot,category,t_start,t_end\n")
    pointer = write(review / "source.json", json.dumps({"table": str(table)}))
    assert jw.curated_shots(tmp_path) == set()  # no sidecar: nothing is curated
    merged = write(tmp_path / "merged.csv", "shot,t_start\n10,0\n11,0\n10,5\n")
    sidecar = tmp_path / "suggest.meta.json"
    meta = {"sources": {"curated": {"table": str(merged), "sha256": jw.sha256_of(merged)}}}
    sidecar.write_text(json.dumps(meta))
    assert jw.curated_shots(tmp_path) == {10, 11}
    merged.write_text("shot,t_start\n12,0\n")
    with pytest.raises(ValueError, match="not the table"):
        jw.curated_shots(tmp_path)
    assert pointer.is_file()
