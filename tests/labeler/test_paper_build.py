"""The paper's build: what the inputs allow, its manifest, the manuscript's copy."""

from __future__ import annotations

import builtins
import errno
import hashlib
import io
import json
import os
from collections import Counter
from pathlib import Path

import h5py
import pytest

from labeler.ae import xpower
from labeler.paper import (
    AE,
    ORDER,
    build,
    coverage,
    frame_examples,
    roc,
    roster,
    scores,
    shots,
)

from . import ae_tree
from . import paper_tree as tree

EXAMPLES = tuple(
    frame_examples.figure_name(m) for m in roc.SELECTED.values() if m != roc.AE_METHOD
)


@pytest.fixture(autouse=True)
def paths(tmp_path, monkeypatch):
    """A temporary Paths, set before every call; `runs` sets its tree's, laid
    out the same way, over it."""
    return tree.temporary_paths(tmp_path, monkeypatch)


@pytest.fixture
def runs(tmp_path, monkeypatch):
    """The AE runs' records, and no roster candidate, so `fig_interpreter` is
    skipped (`NO_ROSTER`) unless a test gives it one (`_roster`)."""
    paths = ae_tree.build(tmp_path, {101: "train", 102: "valid", 103: "valid"})
    models = ae_tree.chosen(paths, {101: "train", 102: "test", 103: "test"})
    (models / "evaluation.json").write_text(json.dumps(tree.ae_evaluation()))
    tree.record_labels(paths)
    ae_tree.env(monkeypatch, paths)
    monkeypatch.setattr(roster, "candidates", lambda paths, snap=None: [])
    return paths


NO_ROSTER = {"reason": build.NO_CANDIDATE, "missing": []}


def _roster(paths, monkeypatch, shots=(102,), corpus=(102,)) -> list:
    """`shots` as the roster's candidates, over 0-1000 ms, and `corpus` with
    1.2 s of corpus CO2."""
    found = [roster.Candidate(s, 2024, (0, 1000)) for s in shots]
    monkeypatch.setattr(roster, "candidates", lambda paths, snap=None: found)
    ae_tree.corpus(paths.corpus.parent, corpus, seconds=(-0.1, 1.1))
    return found


def _no_old_keys(manifest: dict) -> None:
    """The old AE-shot interpreter's keys are gone: `interpreter` alone."""
    assert "interpreter" in manifest
    assert not [k for k in manifest if k.startswith("interpreter_")], manifest


def test_the_build_draws_what_its_inputs_allow(runs, tmp_path, capsys):
    out, dest = tmp_path / "paper", tmp_path / "manuscript" / "figures"
    assert build.main(["--out", str(out), "--copy-to", str(dest)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["products"] == [
        "fig_coverage",
        "fig_examples_ae",
        "fig_scores",
        "table_ae_scores",
        "table_datasets",
        "table_differences",
    ]
    missing = _missing(build.inputs(runs)["seg_evaluation"])
    assert _skipped(printed) == {
        "table_seg_scores": missing,
        "fig_interpreter": NO_ROSTER,
    }
    assert printed["copied"] == sorted(p.name for p in dest.iterdir())
    assert printed["old_output_kept"] is None, "the old output was deleted whole"
    assert len(printed["copied"]) == 6
    assert all(name.endswith((".pdf", ".tex")) for name in printed["copied"])
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["interpreter"] is None
    _no_old_keys(manifest)
    assert manifest["example_shots"] == [102, 103]
    assert manifest["example_rule"] == shots.EXAMPLES_RULE
    assert sorted(manifest["shot_f1"]) == ["102", "103"], "the shots drawn"
    assert set(manifest["shot_f1"]["102"]) == {"f1_0_2s", "f1_window"}
    assert sorted(manifest["inputs"]) == [
        "ae_chosen",
        "ae_evaluation",
        "ae_labels",
        "ae_model",
        "ae_scored_labels",
        "ae_split",
        "store_102",
        "store_103",
    ]
    for name, files in manifest["products"].items():
        assert all((out / f).is_file() for f in files), name


def _skipped(record: dict) -> dict:
    """`skipped` without the frame models' example figures, which these tests
    leave to `test_paper_frame_examples`."""
    return {k: v for k, v in record["skipped"].items() if k not in EXAMPLES}


def _missing(*paths) -> dict:
    return {"reason": build.MISSING, "missing": [str(p) for p in paths]}


def _ae(entries: list[dict]) -> list[dict]:
    """AE's own `partial` entries: the frame-model phenomena's name theirs
    (`phenomenon`) and come after AE's (D62)."""
    return [e for e in entries if "phenomenon" not in e]


def test_without_the_chosen_model_the_shot_figures_wait(runs, tmp_path):
    (xpower.model_dir(runs) / "chosen.json").unlink()
    manifest = build.build(runs, tmp_path / "paper")
    missing = _missing(build.inputs(runs)["ae_chosen"])
    assert manifest["skipped"]["fig_interpreter"] == missing
    assert manifest["skipped"]["fig_examples_ae"] == missing
    assert "fig_coverage" in manifest["products"], "the split is left empty"
    assert manifest["interpreter"] is None


def test_the_interpreter_is_the_roster_figure(runs, tmp_path, monkeypatch):
    """The build draws `roster.draw`'s figure as `fig_interpreter`, from the
    roster's pick, with the build's AE model (read once, as the examples read
    it) and SegNet `--interpreter-seg-version` (v3, none here: no mask)."""
    _roster(runs, monkeypatch)
    drawn = []
    draw = roster.draw

    def spy(s, stem):
        drawn.append((s, stem.name))
        return draw(s, stem)

    monkeypatch.setattr(roster, "draw", spy)
    manifest = build.build(runs, tmp_path / "paper")
    assert manifest["products"]["fig_interpreter"] == build.files_of("fig_interpreter")
    [(s, stem)] = drawn
    assert (s.shot, stem, list(s.tracks)) == (102, "fig_interpreter", list(ORDER))
    _no_old_keys(manifest)
    found, got = build.inputs(runs), manifest["interpreter"]
    assert (got["shot"], got["pick_rule"], got["candidates"]) == (
        102,
        roster.PICK_RULE,
        1,
    )
    assert (got["ae_version"], got["seg_version"]) == ("v1", "v3")
    assert got["ae_model"] == manifest["inputs"]["ae_model"]
    assert got["seg_model"] is None and got["seg_band_khz"] is None
    assert got["tables"] == dict.fromkeys(roster.TABLES.values())
    assert got["stores"] == dict.fromkeys(roster.TABLES), "no store of 102's"
    assert got["n_gate"] is None, "no n map to gate"
    assert got["tier"] == roster.TIER == "suggestions"
    no_mask = {
        "reason": build.NO_INTERPRETER_MASK,
        "missing": [str(found["interpreter_seg_model"])],
    }
    tables = [
        {
            "reason": build.NO_TRACK_TABLE,
            "phenomenon": category,
            "missing": [str(roster.table_file(runs, category))],
        }
        for category in roster.TABLES
    ]
    panels = [
        {
            "reason": build.NO_PANEL_STORE.format(f"no {p.title} data"),
            "panel": p.title,
            "missing": [str(runs.spectrogram_file(p.event, 102))],
        }
        for p in roster.PANELS
    ]
    assert manifest["partial"]["fig_interpreter"] == [no_mask, *tables, *panels]


def test_the_interpreter_shot_must_be_a_roster_candidate(
    runs, tmp_path, monkeypatch, capsys
):
    _roster(runs, monkeypatch, shots=(102, 103))
    manifest = build.build(runs, tmp_path / "paper", shot=101, examples=1)
    assert manifest["skipped"]["fig_interpreter"] == {
        "reason": "101: not a non-blind roster shot with corpus CO2",
        "missing": [],
    }
    assert manifest["interpreter"] is None
    assert manifest["example_shots"] == [102], "the examples keep their own pick"
    manifest = build.build(runs, tmp_path / "paper", shot=102)
    assert manifest["interpreter"]["pick_rule"] == roster.NAMED == "named by --shot"
    assert "fig_interpreter" in manifest["products"]
    corpus = runs.corpus_file(103)
    manifest = build.build(runs, tmp_path / "paper", shot=103)
    entry = manifest["skipped"]["fig_interpreter"]
    assert entry["reason"].startswith(f"{build.NO_CO2}: shot 103 has no corpus file")
    assert entry["missing"] == [str(corpus)]
    with pytest.raises(SystemExit):
        build.main(["--help"])
    said = " ".join(capsys.readouterr().out.split())
    assert (
        "--shot SHOT the interpreter's shot, which must be a roster candidate" in said
    )
    assert "--interpreter-seg-version" in said


def test_the_interpreter_s_segnet_has_its_own_flag(runs, tmp_path, monkeypatch):
    """`--seg-version` is the AE figures' SegNet, `--interpreter-seg-version`
    the interpreter's; one version is one SegNet, read once."""
    _roster(runs, monkeypatch)
    tree.seg_record(runs)
    v3 = tree.seg_model(runs, "v3")
    out = tmp_path / "paper"
    assert build.main(["--out", str(out), "--seg-version", "v1"]) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    pinned, got = manifest["inputs"], manifest["interpreter"]
    assert (manifest["seg_version"], got["seg_version"]) == ("v1", "v3")
    assert pinned["seg_model"]["path"].endswith("/models/ae_seg/v1/model.pt")
    assert got["seg_model"] == pinned["interpreter_seg_model"]
    assert got["seg_model"]["path"] == str(v3)
    assert got["seg_band_khz"] is not None
    argv = ["--out", str(out), "--interpreter-seg-version", "v1"]
    assert build.main(argv) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["interpreter"]["seg_version"] == manifest["seg_version"] == "v1"
    assert manifest["interpreter"]["seg_model"] == manifest["inputs"]["seg_model"]
    assert "interpreter_seg_model" not in manifest["inputs"]


def _skipped_shot_figures(manifest: dict, reason: str) -> None:
    assert manifest["skipped"]["fig_examples_ae"] == {"reason": reason, "missing": []}
    assert "fig_examples_ae" not in manifest["products"]
    assert "fig_scores" in manifest["products"], "the rest is drawn"


def test_no_test_shot_with_an_f1_skips_the_shot_products(runs, tmp_path):
    copy = tree.scored_labels(runs)
    rows = [r for r in copy.read_text().splitlines() if r[:4] not in ("102,", "103,")]
    rows += ["102,2,0,2000,", "103,2,0,2000,"]  # uncertain: no frame is scored
    copy.write_text("\n".join(rows) + "\n")
    tree.record_labels(runs)
    manifest = build.build(runs, tmp_path / "paper")
    _skipped_shot_figures(manifest, build.NO_F1)


def test_products_drawn_without_an_input_are_recorded_as_partial(runs, tmp_path):
    found = build.inputs(runs)
    manifest = build.build(runs, tmp_path / "paper")
    no_summary = {
        "reason": "extension not run: A2 failed (D47)",
        "missing": [str(found["summary"])],
    }
    no_mask = {"reason": build.NO_MASK, "missing": [str(found["seg_model"])]}
    ae_own = {k: _ae(v) for k, v in manifest["partial"].items() if k != "fig_scores"}
    assert ae_own == {
        "fig_coverage": [no_summary],
        "table_datasets": [no_summary],
        "fig_examples_ae": [no_mask],
        "table_differences": [_missing(found["seg_evaluation"])],
    }
    assert manifest["partial"]["fig_scores"] == _unscored(runs, scored=(AE,))
    assert set(manifest["partial"]) <= set(manifest["products"])
    (xpower.model_dir(runs) / "chosen.json").unlink()
    manifest = build.build(runs, tmp_path / "paper")
    no_split = {"reason": build.NO_CHOSEN, "missing": [str(found["ae_chosen"])]}
    assert _ae(manifest["partial"]["fig_coverage"]) == [no_split, no_summary]
    assert _ae(manifest["partial"]["table_datasets"]) == [no_split, no_summary]


def test_the_extension_reason_comes_from_the_bar(runs, tmp_path):
    evaluation = xpower.model_dir(runs) / "evaluation.json"
    record = json.loads(evaluation.read_text())
    record["bar"].update(A1=False, A2=False)
    evaluation.write_text(json.dumps(record))
    [entry] = _ae(build.build(runs, tmp_path / "paper")["partial"]["fig_coverage"])
    assert entry["reason"] == "extension not run: A1 and A2 failed (D47)"
    record["bar"].update(A1=True, A2=True)
    evaluation.write_text(json.dumps(record))
    [entry] = _ae(build.build(runs, tmp_path / "paper")["partial"]["fig_coverage"])
    assert entry["reason"] == build.NO_SUMMARY, "A1 and A2 pass: not written yet"
    evaluation.unlink()
    [entry] = _ae(build.build(runs, tmp_path / "paper")["partial"]["fig_coverage"])
    assert entry["reason"] == build.NO_EVALUATION


NO_BAR_KEY = "the record has no bar"
UNDECIDED = "extension not run: the AE evaluation's bar leaves {} undecided (D47)"


@pytest.mark.parametrize(
    ("bar", "reason"),
    [
        (
            {"A1": True, "A2": True, "all": False},
            "the extension passed its gate (D47) but has not written summary.csv",
        ),
        ({"A1": False, "A2": True}, "extension not run: A1 failed (D47)"),
        ({"A1": True, "A2": False}, "extension not run: A2 failed (D47)"),
        ({"A1": False, "A2": False}, "extension not run: A1 and A2 failed (D47)"),
        ({"A1": False}, "extension not run: A1 failed (D47)"),
        ({"A1": True}, UNDECIDED.format("A2")),
        ({"A1": None, "A2": True}, UNDECIDED.format("A1")),
        ({"A1": "true", "A2": 1}, UNDECIDED.format("A1 and A2")),
        ({}, UNDECIDED.format("A1 and A2")),
        (None, "extension not run: the AE evaluation records no bar (D47)"),
        (NO_BAR_KEY, "extension not run: the AE evaluation records no bar (D47)"),
    ],
    ids=[
        "both-pass",
        "A1-fails",
        "A2-fails",
        "both-fail",
        "A1-fails-A2-unrecorded",
        "A2-unrecorded",
        "A1-null",
        "not-booleans",
        "empty-bar",
        "bar-null",
        "no-bar",
    ],
)
def test_the_extension_reason_says_what_the_bar_records(bar, reason):
    record = tree.ae_evaluation()
    del record["bar"]
    if bar != NO_BAR_KEY:
        record["bar"] = bar
    said = build.extension_reason(record)
    assert said == reason
    passed = isinstance(bar, dict) and bar.get("A1") is True and bar.get("A2") is True
    assert ("passed its gate" in said) == passed, "only when both are recorded True"


def _files(out: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(out.iterdir())}


def test_labels_missing_entirely_skip_and_never_crash(runs, tmp_path, monkeypatch):
    live = build.inputs(runs)["ae_labels"]
    live.unlink()
    out = tmp_path / "paper"
    manifest = build.build(runs, out)
    for product in ("fig_coverage", "table_datasets"):
        assert manifest["skipped"][product] == _missing(live)
    assert "fig_examples_ae" in manifest["products"], "from the model's own copy"
    copy = tree.scored_labels(runs)
    copy.unlink()
    _roster(runs, monkeypatch)
    manifest = build.build(runs, out)
    assert manifest["skipped"]["fig_examples_ae"] == _missing(copy)
    assert "fig_interpreter" in manifest["products"], "it reads no labels"
    assert "fig_scores" in manifest["products"], "the rest is drawn"
    assert set(_files(out)) == {
        f for files in manifest["products"].values() for f in files
    } | {"manifest.json"}


def test_a_test_shot_without_a_label_makes_the_shot_products_partial(runs, tmp_path):
    copy = tree.scored_labels(runs)
    rows = copy.read_text().splitlines()
    copy.write_text("\n".join(r for r in rows if not r.startswith("103,")) + "\n")
    tree.record_labels(runs)
    manifest = build.build(runs, tmp_path / "paper")
    assert "fig_examples_ae" in manifest["products"]
    no_label = {"reason": build.NO_LABEL, "shots": [103]}
    assert no_label in manifest["partial"]["fig_examples_ae"]
    assert manifest["example_shots"] == [102]


def test_a_failure_mid_build_leaves_out_as_it_was(runs, tmp_path, monkeypatch):
    out = tmp_path / "paper"
    build.build(runs, out, examples=1)
    (out / "notes.txt").write_text("the owner's, not a product")
    before = _files(out)

    def fails(*args):
        raise RuntimeError("drawing failed")

    evaluation = xpower.model_dir(runs) / "evaluation.json"
    record = json.loads(evaluation.read_text())
    record["methods"]["ae_xpower"]["f1"] = tree.est(0.5, 0.4, 0.6)
    evaluation.write_text(json.dumps(record))  # new scores, drawn before the failure
    with monkeypatch.context() as failing:
        failing.setattr(shots, "draw_examples", fails)
        with pytest.raises(RuntimeError, match="drawing failed"):
            build.build(runs, out)
    assert _files(out) == before
    assert sorted(p.name for p in tmp_path.iterdir() if "paper" in p.name) == ["paper"]
    assert os.environ["LABELER_ROOT"] == str(runs.root), "the fixture's Paths stay"
    (xpower.model_dir(runs) / "chosen.json").unlink()
    manifest = build.build(runs, out)
    after = _files(out)
    assert "fig_examples_ae.pdf" not in after, "a skipped product leaves with its files"
    assert after["notes.txt"] == before["notes.txt"], "a file not ours is kept"
    assert json.loads(after["manifest.json"]) == manifest


def _tree(root: Path) -> dict[str, bytes | None]:
    """Every entry under `root` by its relative path: a file's bytes, or None
    for a directory."""
    return {
        p.relative_to(root).as_posix(): None if p.is_dir() else p.read_bytes()
        for p in sorted(root.rglob("*"))
    }


def _beside(tmp_path: Path) -> list[str]:
    """What sits beside `out` (`tmp_path / "paper"`): it, and any staging or
    holding directory the build left."""
    return sorted(p.name for p in tmp_path.iterdir() if "paper" in p.name)


def test_a_failed_swap_puts_the_old_output_back(runs, tmp_path, monkeypatch):
    """The new output's rename fails and the old one is renamed back: `out` is
    as it was, file for file, and nothing is left beside it."""
    out = tmp_path / "paper"
    build.build(runs, out, examples=1)
    (out / "notes.txt").write_text("the owner's, not a product")
    before = _tree(out)
    rename = Path.rename

    def refuses_the_new_output(self, target):
        if Path(self).name.startswith(f".{out.name}.staging-"):
            raise OSError(errno.EIO, "the swap's rename failed")
        return rename(self, target)

    with monkeypatch.context() as patched:
        patched.setattr(Path, "rename", refuses_the_new_output)
        with pytest.raises(OSError, match="the swap's rename failed"):
            build.build(runs, out)
    assert _tree(out) == before
    assert _beside(tmp_path) == ["paper"], "no staging or holding directory left"


def test_a_second_build_between_the_renames_deletes_nothing(
    runs, tmp_path, monkeypatch
):
    """Another build puts its output at `out` while this one has the old output
    moved aside, so neither can be put there. Nothing is deleted: the old output
    and the new one are kept, and the error says where."""
    out = tmp_path / "paper"
    build.build(runs, out, examples=1)
    (out / "notes.txt").write_text("the owner's, not a product")
    before = _tree(out)
    rename = Path.rename

    def the_other_build_lands(self, target):
        moved = rename(self, target)
        if Path(self) == out:  # the old output is aside: the other build's lands
            out.mkdir()
            (out / "fig_scores.pdf").write_text("the other build's")
        return moved

    with monkeypatch.context() as patched:
        patched.setattr(Path, "rename", the_other_build_lands)
        with pytest.raises(OSError) as caught:
            build.build(runs, out)
    near = [*tmp_path.glob("*paper*"), *tmp_path.glob("*paper*/*")]
    homes = [d for d in near if d.is_dir() and _tree(d) == before]
    assert homes, "every file of the old output still exists, together"
    error = caught.value
    assert isinstance(error, build.Stranded), repr(error)
    assert error.errno in (errno.ENOTEMPTY, errno.EEXIST), "the undo's rename"
    assert error.strerror == os.strerror(error.errno)
    assert homes == [error.old] and str(error.old) in str(error), "named"
    assert str(error.new) in str(error), "the new output is named too"
    assert error.entries == {"notes.txt": error.old / "notes.txt"}, "moved back"
    new = _tree(error.new)
    assert "manifest.json" in new and "notes.txt" not in new
    assert _tree(out) == {"fig_scores.pdf": b"the other build's"}, "not touched"


def test_only_the_builds_own_files_leave_out(runs, tmp_path):
    """The build claims the manifest and its products' own files, by exact
    name: an owner's `fig_scores.svg`, a `fig_scores/` directory, or a link to
    a directory elsewhere, stays as it is."""
    out = tmp_path / "paper"
    build.build(runs, out, examples=1)
    (out / "fig_scores.svg").write_text("the owner's own drawing")
    (out / "fig_scores").mkdir()
    (out / "fig_scores" / "notes.txt").write_text("the owner's notes")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "big.dat").write_text("not to be copied")
    (out / "linked").symlink_to(elsewhere, target_is_directory=True)
    mine = ("fig_scores.svg", "fig_scores", "fig_scores/notes.txt", "linked")
    theirs = {k: v for k, v in _tree(out).items() if k in mine}
    manifest = build.build(runs, out)
    after = _tree(out)
    assert {k: after.get(k, "gone") for k in theirs} == theirs
    drawn = {f for files in manifest["products"].values() for f in files}
    assert set(after) == drawn | {"manifest.json"} | set(theirs)
    assert os.readlink(out / "linked") == str(elsewhere), "still a link"


def test_a_symlinked_out_keeps_its_link(runs, tmp_path):
    """The reviewer's probe A through the build: `--out` is a link to the
    owner's real directory. The build draws beside the link's target and swaps
    the new output in there: the link still points at it, the owner's files
    stay in it, and no stale product is left anywhere."""
    target = tmp_path / "real" / "paper"
    build.build(runs, target, examples=1)  # an old output, with the shot figures
    (target / "notes.txt").write_text("the owner's")
    out = tmp_path / "paper"
    out.symlink_to(target, target_is_directory=True)
    (xpower.model_dir(runs) / "chosen.json").unlink()  # now the shot figures skip
    manifest = build.build(runs, out)
    assert out.is_symlink() and os.readlink(out) == str(target), "still that link"
    drawn = {f for files in manifest["products"].values() for f in files}
    assert "fig_examples_ae.pdf" not in drawn
    files = _files(target)
    assert set(files) == drawn | {"manifest.json", "notes.txt"}
    assert files["notes.txt"] == b"the owner's"
    assert json.loads(files["manifest.json"]) == manifest, "the new output"
    assert list(tmp_path.rglob("fig_examples_ae*")) == [], "no stale product anywhere"
    assert sorted(p.name for p in target.parent.iterdir()) == ["paper"]
    assert _beside(tmp_path) == ["paper"], "nothing left beside the link"


def test_a_broken_link_as_out_is_built_into_its_target(runs, tmp_path):
    """A link whose target is not there yet is built into its target, as a
    new `--out` is: the build makes it (its parents too), the link then
    points at the new output, and nothing is left beside either."""
    target = tmp_path / "later" / "paper"
    out = tmp_path / "paper"
    out.symlink_to(target, target_is_directory=True)
    manifest = build.build(runs, out)
    assert out.is_symlink() and os.readlink(out) == str(target)
    drawn = {f for files in manifest["products"].values() for f in files}
    assert set(_files(target)) == drawn | {"manifest.json"}
    assert sorted(p.name for p in target.parent.iterdir()) == ["paper"]
    assert _beside(tmp_path) == ["paper"]


def test_a_kept_old_output_is_named_and_the_build_succeeds(
    runs, tmp_path, monkeypatch, capsys
):
    """The owner saves `notes.txt` again through a handle held on the old
    output, after the swap has moved the first one into the new output: the
    late one cannot be moved over it, so the old output's directory is kept.
    The build still succeeds, its output in place, and names the directory on
    stderr and in its JSON line."""
    out = tmp_path / "paper"
    build.build(runs, out, examples=1)
    (out / "notes.txt").write_text("the owner's")
    held = os.open(out, os.O_RDONLY | os.O_DIRECTORY)
    rename = Path.rename

    def saves_late(self, target):
        if Path(self).name.startswith(f".{out.name}.staging-"):
            fd = os.open("notes.txt", os.O_WRONLY | os.O_CREAT, 0o644, dir_fd=held)
            with os.fdopen(fd, "w") as file:
                file.write("saved again, late")
        return rename(self, target)

    capsys.readouterr()
    try:
        with monkeypatch.context() as patched:
            patched.setattr(Path, "rename", saves_late)
            assert build.main(["--out", str(out)]) == 0
    finally:
        os.close(held)
    said = capsys.readouterr()
    kept = Path(json.loads(said.out)["old_output_kept"])
    assert kept.parent == tmp_path and kept.name.startswith(".paper.old-")
    assert _tree(kept) == {"paper": None, "paper/notes.txt": b"saved again, late"}
    assert str(kept) in said.err
    assert (out / "notes.txt").read_text() == "the owner's", "not moved over"
    manifest = json.loads((out / "manifest.json").read_text())
    assert "fig_examples_ae" in manifest["products"], "the new output is in place"
    assert _beside(tmp_path) == sorted(["paper", kept.name])


def _spy_reads(monkeypatch, opened: Counter) -> None:
    """Count every file opened for reading through Python, pathlib, pandas,
    torch and h5py."""

    def wrap(real):
        def spy(file, mode="r", *args, **kwargs):
            if isinstance(file, str | os.PathLike) and not set(mode) & set("wax+"):
                opened[str(Path(file).resolve())] += 1
            return real(file, mode, *args, **kwargs)

        return spy

    monkeypatch.setattr(builtins, "open", wrap(builtins.open))
    monkeypatch.setattr(io, "open", wrap(io.open))
    monkeypatch.setattr(h5py, "File", wrap(h5py.File))


def test_the_manifest_pins_every_file_the_build_reads(runs, tmp_path, monkeypatch):
    tree.seg_model(runs)
    opened: Counter = Counter()
    _spy_reads(monkeypatch, opened)
    out = tmp_path / "paper"
    manifest = build.build(runs, out)
    tree_root = str(tmp_path.resolve())
    read = {p for p in opened if p.startswith(tree_root) and not p.startswith(str(out))}
    pinned = {str(Path(v["path"]).resolve()) for v in manifest["inputs"].values()}
    assert read and read <= pinned, sorted(read - pinned)
    twice = {p: n for p, n in opened.items() if p in pinned and n != 2}
    assert not twice, f"each input read once to draw, once to check: {twice}"
    assert manifest["consistent"] is True and manifest["changed_during_build"] == {}
    assert {"ae_model", "ae_split", "store_102", "store_103", "seg_model"} <= set(
        manifest["inputs"]
    )
    for entry in manifest["inputs"].values():
        data = Path(entry["path"]).read_bytes()
        assert entry["sha256"] == hashlib.sha256(data).hexdigest()


def test_the_snapshot_refuses_a_second_read(runs, tmp_path):
    snap = build.Snapshot(tmp_path / "scratch")
    evaluation = xpower.model_dir(runs) / "evaluation.json"
    model = xpower.model_dir(runs) / "band80-mhd3" / "model.pt"
    record = snap.json("ae_evaluation", evaluation)
    snap.model("ae_model", model, {102: "test"})
    drawn = {key: snap.sha(key) for key in ("ae_evaluation", "ae_model")}
    evaluation.write_text(json.dumps({**record, "rerun": True}))  # a run rewrites it
    again = {
        "ae_evaluation": lambda: snap.json("ae_evaluation", evaluation),
        "ae_model": lambda: snap.model("ae_model", model, {102: "test"}),
    }
    for key, read in again.items():
        with pytest.raises(ValueError, match=f"{key}: already read"):
            read()
        assert snap.sha(key) == drawn[key], "the pin stays the bytes drawn"
    now = hashlib.sha256(evaluation.read_bytes()).hexdigest()
    assert snap.changed() == {
        "ae_evaluation": {
            "path": str(evaluation),
            "drawn": drawn["ae_evaluation"],
            "now": now,
        }
    }


def test_the_snapshot_pins_bytes_its_caller_read(tmp_path):
    """`pin` pins bytes read elsewhere as `read` pins its own, under the same
    second-read guard, and `changed` hashes their file again."""
    snap = build.Snapshot(tmp_path / "scratch")
    path = tmp_path / "101.npz"
    path.write_bytes(b"drawn")
    data = path.read_bytes()
    assert snap.pin("features", path, data) is data
    assert snap.pinned["features"] == (path, hashlib.sha256(b"drawn").hexdigest())
    for again in (
        lambda: snap.pin("features", path, b"other"),
        lambda: snap.read("features", path),
    ):
        with pytest.raises(ValueError, match="features: already read"):
            again()
    assert snap.sha("features") == hashlib.sha256(b"drawn").hexdigest()
    snap.read("read", path)
    with pytest.raises(ValueError, match="read: already read"):
        snap.pin("read", path, data)
    assert snap.changed() == {}
    path.write_bytes(b"rewritten")
    assert set(snap.changed()) == {"features", "read"}


def test_an_input_changed_mid_build_is_recorded(runs, tmp_path, monkeypatch):
    live = build.inputs(runs)["ae_labels"]
    drawn = hashlib.sha256(live.read_bytes()).hexdigest()
    draw = coverage.draw_coverage

    def then_the_owner_saves(*args):
        figure = draw(*args)
        live.write_text(live.read_text() + "104,0,0,2000,\n")
        return figure

    monkeypatch.setattr(coverage, "draw_coverage", then_the_owner_saves)
    manifest = build.build(runs, tmp_path / "paper")
    now = hashlib.sha256(live.read_bytes()).hexdigest()
    assert manifest["inputs"]["ae_labels"]["sha256"] == drawn, "the bytes drawn"
    assert manifest["labels_sha256"]["live"] == drawn
    assert manifest["consistent"] is False
    assert manifest["changed_during_build"] == {
        "ae_labels": {"path": str(live), "drawn": drawn, "now": now}
    }
    table = (tmp_path / "paper" / "table_datasets.tex").read_text()
    assert "AE & 3 & " in table, "drawn from the bytes read, before the save"


def test_the_manifest_records_the_commit_and_the_labels_check(runs, tmp_path):
    manifest = build.build(runs, tmp_path / "paper")
    assert manifest["git_sha"] == "unknown" or len(manifest["git_sha"]) == 40
    assert manifest["git_dirty"] in (True, False, None)
    assert manifest["version"] == "v1"
    scored = hashlib.sha256(tree.scored_labels(runs).read_bytes()).hexdigest()
    assert manifest["labels_match"] is True
    assert manifest["labels_sha256"] == {
        "scored": scored,
        "live": scored,
        "ae_evaluation": scored,
    }


def _name_labels(paths, sha: str | None) -> None:
    evaluation = xpower.model_dir(paths) / "evaluation.json"
    record = json.loads(evaluation.read_text())
    record["meta"].pop("labels_sha256", None)
    if sha is not None:
        record["meta"]["labels_sha256"] = sha
    evaluation.write_text(json.dumps(record))


def test_labels_unlike_the_evaluations_refuse_the_shot_products(runs, tmp_path):
    _name_labels(runs, "0" * 64)
    manifest = build.build(runs, tmp_path / "paper")
    assert manifest["labels_match"] is False
    assert "fig_scores" in manifest["products"], "the scores are the record's own"
    entry = manifest["skipped"]["fig_examples_ae"]
    assert entry["reason"].startswith(build.LABELS_DIFFER), entry
    _name_labels(runs, None)
    manifest = build.build(runs, tmp_path / "paper")
    assert manifest["labels_match"] is None
    assert manifest["skipped"]["fig_examples_ae"]["reason"] == build.LABELS_UNNAMED


def _name_copy(paths, sha: str) -> None:
    evaluation = xpower.model_dir(paths) / "evaluation.json"
    record = json.loads(evaluation.read_text())
    record["meta"]["labels_copy_sha256"] = sha
    evaluation.write_text(json.dumps(record))


def test_a_copy_unlike_labels_copy_sha256_refuses_the_shot_products(runs, tmp_path):
    scored = hashlib.sha256(tree.scored_labels(runs).read_bytes()).hexdigest()
    other = "f" * 64
    why = (
        f"{build.LABELS_DIFFER}: the copy is {scored[:12]}, "
        f"the record names labels_copy_sha256 {other[:12]}"
    )
    for named in (scored, None):  # labels_sha256 matches, or is absent
        _name_labels(runs, named)
        _name_copy(runs, other)
        manifest = build.build(runs, tmp_path / "paper")
        assert manifest["labels_match"] is False, named
        assert manifest["labels_sha256"]["ae_evaluation_copy"] == other
        _skipped_shot_figures(manifest, why)
    _name_copy(runs, scored)
    manifest = build.build(runs, tmp_path / "paper")
    assert manifest["labels_match"] is True, "the copy's own sha256 is enough"
    assert "fig_examples_ae" in manifest["products"]


def test_the_scored_products_use_the_labels_the_model_was_scored_on(runs, tmp_path):
    before = build.build(runs, tmp_path / "before")
    live = build.inputs(runs)["ae_labels"]
    rows = [r for r in live.read_text().splitlines() if not r.startswith("102,")]
    rows += ["102,0,0,300,", "102,1,300,1500,", "102,0,1500,2000,"]  # changed
    rows += ["104,0,0,500,", "104,1,500,700,", "104,0,700,2000,"]  # in no split
    live.write_text("\n".join(rows) + "\n")
    after = build.build(runs, tmp_path / "after")
    for key in ("shot_f1", "example_shots"):
        assert after[key] == before[key], key
    assert after["labels_match"] is True, "the copy is the evaluation's"
    shas = after["labels_sha256"]
    assert shas["live"] != shas["scored"] == shas["ae_evaluation"]
    table = (tmp_path / "after" / "table_datasets.tex").read_text().splitlines()
    assert table[5].startswith("AE & 4 & 4 & 4 & 2.6 & 1 & 0 & 2 & 1 & "), table[5]


AE_PRODUCTS = (
    "fig_scores",
    "table_ae_scores",
    "table_seg_scores",
    "table_differences",
    "fig_examples_ae",
)


FRAME_PRODUCTS = (
    *EXAMPLES,
    "fig_scores",
    "table_ae_scores",
    "fig_interpreter",
    "fig_examples_ae",
)


def test_version_v2_skips_until_its_records_exist(runs, tmp_path, capsys):
    """v2 as it stands before its models are chosen: a copy of the labels alone.
    SegNet v1 is drawn beside it, as it stands."""
    tree.seg_record(runs)
    models = xpower.model_dir(runs, "v2")
    (models / "review").mkdir(parents=True)
    (models / "review" / "labels.csv").write_bytes(
        tree.scored_labels(runs).read_bytes()
    )
    out = tmp_path / "paper"
    assert build.main(["--out", str(out), "--version", "v2"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["products"] == [
        "fig_coverage",
        "table_datasets",
        "table_differences",
        "table_seg_scores",
    ]
    manifest = json.loads((out / "manifest.json").read_text())
    assert (manifest["version"], manifest["seg_version"]) == ("v2", "v1")
    assert sorted(manifest["skipped"]) == sorted(FRAME_PRODUCTS)
    for product, entry in manifest["skipped"].items():
        assert entry["reason"] == build.MISSING, product
        assert entry["missing"], product
        assert all("/v2/" in m for m in entry["missing"]), (product, entry)
    evaluation, chosen = models / "evaluation.json", models / "chosen.json"
    assert manifest["skipped"]["fig_examples_ae"]["missing"] == [
        str(evaluation),
        str(chosen),
    ]
    assert manifest["skipped"]["fig_interpreter"]["missing"] == [str(chosen)]
    found = build.inputs(runs, "v2")
    assert manifest["skipped"]["fig_scores"]["missing"] == [
        str(found[build.evaluation_key(m)]) for m in roc.SELECTED.values()
    ], "no selected model's evaluation"
    assert manifest["skipped"]["fig_scores"]["missing"][0] == str(evaluation)
    assert manifest["partial"]["table_differences"] == [_missing(evaluation)]
    assert {"reason": build.NO_CHOSEN, "missing": [str(chosen)]} in manifest["partial"][
        "fig_coverage"
    ], "the owner's labels have no version"
    frame = [k for k in manifest["inputs"] if not k.startswith("seg_")]
    assert all("/v1/" not in manifest["inputs"][k]["path"] for k in frame)
    assert manifest["seg_labels_match"] is True


def test_version_v2_draws_from_v2_records(runs, tmp_path):
    tree.seg_record(runs)
    models = tree.as_version(runs, "v2")
    assert not xpower.model_dir(runs).exists(), "nothing left in v1 to read"
    out = tmp_path / "paper"
    assert build.main(["--out", str(out), "--version", "v2"]) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert (manifest["version"], manifest["seg_version"]) == ("v2", "v1")
    assert _skipped(manifest) == {"fig_interpreter": NO_ROSTER}
    assert set(AE_PRODUCTS) <= set(manifest["products"])
    assert manifest["labels_match"] is True
    pinned = manifest["inputs"]
    for key in ("ae_evaluation", "ae_chosen", "ae_model", "ae_split"):
        assert pinned[key]["path"].startswith(str(models)), key
    assert pinned["ae_scored_labels"]["path"] == str(
        models / "band80-mhd3" / "review" / "labels.csv"
    )


V1_DATASETS = (
    "% Shots per phenomenon: labelled, reviewed by the owner, with any present "
    "span and their present time, the model's split (Train, Val, Test) and the "
    "extension's suggestions (not labels); -- where that run has not happened; "
    "AE's labels are the owner's reviews, so its Labelled = Reviewed = Train + "
    "Val + Test + No split (the reviewed shots saved after the model was "
    "trained)\n"
    "\\begin{tabular}{lcccccccccc}\n"
    "\\toprule\n"
    "Phenomenon & Labelled & Reviewed by the owner & Positive & Present (s) "
    "& Train & Val & Test & No split & Suggested & Suggested positive \\\\\n"
    "\\midrule\n"
    "AE & 3 & 3 & 3 & 1.8 & 1 & 0 & 2 & 0 & -- & -- \\\\\n"
    + "".join(
        f"{name} & \\multicolumn{{10}}{{c}}{{coming}} \\\\\n"
        for name in ("NTM", "H-mode", "ELMing", "sawteeth")
    )
    + "\\bottomrule\n\\end{tabular}\n"
)  # the `runs` tree's table_datasets.tex at 2258daf, before any folds were drawn,
#   less the disruption row the paper has left out since 2026-09-28, in the
#   columns of frames v2 (F8): Labelled and Reviewed by the owner apart (AE's
#   are equal), and Train, Val and Test a column each


def _split_panels(monkeypatch) -> list:
    """The coverage figures the build draws, each as its split panel."""
    panels = []
    draw = coverage.draw_coverage

    def spy(counts, stem):
        fig = draw(counts, stem)
        panels.append(fig.axes[2])
        return fig

    monkeypatch.setattr(coverage, "draw_coverage", spy)
    return panels


def _ticks(ax) -> list[str]:
    return [t.get_text() for t in ax.get_xticklabels()]


def test_v1_is_not_cross_validated(runs, tmp_path, monkeypatch):
    """v1, with no folds, draws its three-way split as before, its table byte
    for byte (in frames v2's columns, F8), and records no folds check."""
    panels = _split_panels(monkeypatch)
    v1 = build.build(runs, tmp_path / "v1")
    assert (tmp_path / "v1" / "table_datasets.tex").read_text() == V1_DATASETS
    assert _ticks(panels[0]) == ["train", "val", "test", "no\nsplit"]
    assert "ae_folds" not in v1["inputs"]
    assert v1["folds_match"] is None


FOLDS = "shot,split,fold\n101,train,0\n102,val,1\n103,test,\n"
NO_FOLD_COLUMN = "shot,split\n101,train\n102,val\n103,test\n"
FINAL_SPLIT = "shot,split\n101,train\n102,train\n103,test\n"  # train_from_cv's


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _cross_validated(paths, folds: str | None, named: str | None) -> Path:
    """v2 as `train_from_cv` leaves it: the final model's split (the folds'
    shots are its train shots), `cv/folds.csv` holding `folds` (None: no file)
    and `chosen.json` naming `named` as its `folds_sha256` (None: no key). The
    folds file's path."""
    models = tree.as_version(paths, "v2")
    (models / "band80-mhd3" / "split.csv").write_text(FINAL_SPLIT)
    path = models / "cv" / "folds.csv"
    path.parent.mkdir()
    if folds is not None:
        path.write_text(folds)
    if named is not None:
        chosen = models / "chosen.json"
        record = json.loads(chosen.read_text())
        chosen.write_text(json.dumps(record | {"folds_sha256": named}))
    return path


@pytest.mark.parametrize(
    ("folds", "named", "match", "why", "tick"),
    [
        (FOLDS, _sha(FOLDS), True, None, "train\n(2 folds)"),
        (FOLDS, "0" * 64, False, "FOLDS_DIFFER", "train"),
        (None, _sha(FOLDS), None, "NO_FOLDS", "train"),
        (NO_FOLD_COLUMN, _sha(NO_FOLD_COLUMN), True, "FOLDS_NO_COLUMN", "train"),
        ("", _sha(""), True, "FOLDS_NO_COLUMN", "train"),
        (FOLDS, None, None, "FOLDS_UNNAMED", "train"),
    ],
    ids=["matching", "unlike", "missing", "no-fold-column", "empty", "unnamed"],
)
def test_a_cross_validated_version_is_checked_against_its_record(
    runs, tmp_path, monkeypatch, folds, named, match, why, tick
):
    """v2 is cross-validated: its train and test shots are drawn, with no
    validation bar or cell, whatever its folds. The train shots carry their
    fold count only when `cv/folds.csv` is there, is the one `chosen.json`
    names and has a `fold` column; otherwise the coverage is partial, saying
    why, and the build never crashes."""
    panels = _split_panels(monkeypatch)
    path = _cross_validated(runs, folds, named)
    manifest = build.build(runs, tmp_path / "v2", version="v2")
    [split] = panels
    assert _ticks(split) == [tick, "test", "no\nsplit"], "no validation bar"
    assert [bar.get_height() for bar in split.patches] == [2, 1, 0]
    lines = (tmp_path / "v2" / "table_datasets.tex").read_text().splitlines()
    assert lines[5] == "AE & 3 & 3 & 3 & 1.8 & 2 & -- & 1 & 0 & -- & -- \\\\"
    over = " over 2 folds" if why is None else ""
    assert (
        f"AE's train shots are cross-validated{over}, so no shot is held out for "
        "validation (--)"
    ) in lines[0]
    assert manifest["folds_match"] is match
    pinned = {"path": str(path), "sha256": _sha(folds)} if folds is not None else None
    assert manifest["inputs"].get("ae_folds") == pinned
    extension = {
        "reason": "extension not run: A2 failed (D47)",
        "missing": [str(build.inputs(runs, "v2")["summary"])],
    }
    entries = _ae(manifest["partial"]["fig_coverage"])
    assert _ae(manifest["partial"]["table_datasets"]) == entries
    assert entries[-1] == extension
    if why is None:
        assert entries == [extension]
        return
    [entry] = entries[:-1]
    assert entry["reason"].startswith(getattr(build, why)), entry
    if why == "FOLDS_DIFFER":
        assert entry["reason"].endswith(
            f": the file is {_sha(FOLDS)[:12]}, chosen.json names {'0' * 12}"
        )
    assert entry.get("missing") == ([str(path)] if folds is None else None)


def test_folds_without_a_chosen_model_are_not_read(runs, tmp_path):
    path = _cross_validated(runs, FOLDS, _sha(FOLDS))
    (path.parent.parent / "chosen.json").unlink()  # the folds made, no model yet
    manifest = build.build(runs, tmp_path / "v2", version="v2")
    assert "ae_folds" not in manifest["inputs"], "no split to show the folds in"
    assert manifest["folds_match"] is None, "no record to check them against"
    assert manifest["partial"]["fig_coverage"][0]["reason"] == build.NO_CHOSEN


def test_validation_shots_in_a_cross_validated_split_are_named(
    runs, tmp_path, monkeypatch
):
    """A cross-validated version holds no shot out for validation; one its
    split calls `val` anyway is in no bar or cell, and named as such."""
    panels = _split_panels(monkeypatch)
    path = _cross_validated(runs, FOLDS, _sha(FOLDS))
    (path.parent.parent / "band80-mhd3" / "split.csv").write_text(
        "shot,split\n101,train\n102,val\n103,test\n"
    )
    manifest = build.build(runs, tmp_path / "v2", version="v2")
    [split] = panels
    assert _ticks(split) == ["train\n(2 folds)", "test", "no\nsplit"]
    assert [bar.get_height() for bar in split.patches] == [1, 1, 0]
    lines = (tmp_path / "v2" / "table_datasets.tex").read_text().splitlines()
    assert lines[5] == "AE & 3 & 3 & 3 & 1.8 & 1 & -- & 1 & 0 & -- & -- \\\\"
    assert manifest["folds_match"] is True
    held = {"reason": build.CV_VAL, "shots": [102]}
    for product in ("fig_coverage", "table_datasets"):
        assert held in manifest["partial"][product]


def _seg_products_of(manifest: dict) -> dict:
    return {
        k: manifest["partial"].get(k) for k in ("table_seg_scores", "table_differences")
    }


def test_the_segmentation_keeps_its_own_version(runs, tmp_path):
    """v2 changes the frame model only: SegNet v1 is drawn beside it, pinned and
    checked against its own record's labels, not the frame model's."""
    sha = tree.seg_record(runs, "v1")
    models = tree.as_version(runs, "v2", keep=True)
    earlier = xpower.model_dir(runs) / "band80-mhd3" / "split.csv"
    earlier.write_text("shot,split\n101,train\n102,test\n103,train\n")
    live = tree.scored_labels(runs, "v2")
    live.write_text(live.read_text() + "104,0,0,2000,\n")  # v2 has more labels
    frame = tree.record_labels(runs, "v2")
    assert frame != sha
    out = tmp_path / "paper"
    argv = ["--out", str(out), "--version", "v2", "--seg-version", "v1"]
    assert build.main(argv) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert (manifest["version"], manifest["seg_version"]) == ("v2", "v1")
    assert _skipped(manifest) == {"fig_interpreter": NO_ROSTER}
    assert {"table_seg_scores", "table_differences"} <= set(manifest["products"])
    assert _seg_products_of(manifest) == dict.fromkeys(_seg_products_of(manifest))
    seg = runs.root / "models" / "ae_seg" / "v1"
    pinned = manifest["inputs"]
    assert pinned["seg_evaluation"]["path"] == str(seg / "evaluation.json")
    assert pinned["seg_labels"]["path"] == str(seg / "review" / "labels.csv")
    assert not [k for k in pinned if "poi" in k], "no points of interest are read"
    assert pinned["seg_model"]["path"] == str(seg / "model.pt"), "the mask's SegNet"
    assert pinned["ae_model"]["path"].startswith(str(models))
    assert manifest["labels_match"] is True, "the frame model on its own copy"
    assert manifest["seg_labels_match"] is True, "SegNet on its own copy"
    shas = manifest["labels_sha256"]
    assert shas["scored"] == shas["ae_evaluation"] == frame
    assert shas["seg_scored"] == shas["seg_evaluation"] == sha
    beside = str(xpower.model_dir(runs) / "band80-mhd3" / "model.pt")
    assert manifest["seg_ae_model"] == beside
    comment = (out / "table_seg_scores.tex").read_text().splitlines()[0]
    assert comment.startswith("%") and "ae_xpower/v1/band80-mhd3" in comment
    look = manifest["second_look"]
    assert (look["shots"], look["of"], look["candidate"]) == (1, 2, "band80-mhd3")
    assert pinned["ae_earlier_split"]["sha256"] == _sha(earlier.read_text())
    said = "1 of the 2 test shots were v1's test shots, scored with v1's model"
    comment = (out / "table_ae_scores.tex").read_text().splitlines()[0]
    assert f"; {said} band80-mhd3" in comment, "counted from v1's records"
    assert "fig_examples_ae" not in manifest["partial"], manifest["partial"]


def test_the_segmentation_is_checked_against_its_own_copy(runs, tmp_path):
    tree.seg_record(runs)
    seg = runs.root / "models" / "ae_seg" / "v1"
    copy = seg / "review" / "labels.csv"
    copy.write_text(copy.read_text() + "104,0,0,2000,\n")
    manifest = build.build(runs, tmp_path / "paper")
    assert manifest["seg_version"] == "v1"
    assert manifest["labels_match"] is True
    assert manifest["seg_labels_match"] is False
    assert "table_seg_scores" in manifest["products"], "the scores are the record's"
    (entry,) = manifest["partial"]["table_seg_scores"]
    assert entry["reason"].startswith(build.SEG_LABELS_DIFFER), entry
    copy.unlink()
    manifest = build.build(runs, tmp_path / "paper")
    assert manifest["seg_labels_match"] is None
    no_copy = {"reason": build.NO_SEG_COPY, "missing": [str(copy)]}
    assert manifest["partial"]["table_seg_scores"] == [no_copy]


def test_a_seg_version_not_run_skips_only_the_segmentation(runs, tmp_path):
    tree.seg_record(runs)
    tree.as_version(runs, "v2")
    out = tmp_path / "paper"
    argv = ["--out", str(out), "--version", "v2", "--seg-version", "v9"]
    assert build.main(argv) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    seg = str(build.inputs(runs, "v2", "v9")["seg_evaluation"])
    assert "/ae_seg/v9/" in seg
    assert _skipped(manifest) == {
        "table_seg_scores": _missing(seg),
        "fig_interpreter": NO_ROSTER,
    }
    unlooked = {
        "reason": build.NO_EARLIER,
        "missing": [str(xpower.model_dir(runs) / "evaluation.json")],
    }
    assert manifest["partial"]["table_differences"] == [_missing(seg), unlooked]
    assert {"fig_scores", "fig_examples_ae"} <= set(manifest["products"])


def _unscored(paths, scored=(), rocs=(), version=build.VERSION) -> list[dict]:
    """fig_scores' `partial` entries: each phenomenon's without its evaluation
    (not in `scored`), then without its ROC (not in `rocs`)."""
    found, entries = build.inputs(paths, version), []
    for category, method in roc.SELECTED.items():
        if category not in scored:
            where = [str(found[build.evaluation_key(method)])]
            reason = build.NOT_SCORED_WHY
            entries.append({"reason": reason, "phenomenon": category, "missing": where})
        if category not in rocs:
            where = [str(found[build.roc_key(method)])]
            reason = build.NO_ROC_WHY
            entries.append({"reason": reason, "phenomenon": category, "missing": where})
    return entries


def _selected_records(paths) -> dict[str, dict]:
    """A frame evaluation for each frame model (F1 0.8 - 0.1 i) and a
    `roc.json` for every selected model, each naming its evaluation's sha256."""
    records = {}
    for i, method in enumerate(roc.SELECTED.values()):
        evaluation = roc.evaluation_file(paths, method, build.VERSION)
        if method != roc.AE_METHOD:
            v = round(0.8 - 0.1 * i, 2)
            key = roc.f1_key(method)
            f1 = {"intervals": {method: {key: tree.est(v, v - 0.1, v + 0.05)}}}
            evaluation.parent.mkdir(parents=True, exist_ok=True)
            evaluation.write_text(json.dumps(f1))
        record = {
            "auroc": 0.9,
            "auprc": 0.8,
            "positive_share": 0.3,
            "curve": {"fpr": [0.0, 0.2, 1.0], "tpr": [0.0, 0.8, 1.0]},
            "pr": {"recall": [0.0, 0.8, 1.0], "precision": [1.0, 0.7, 0.3]},
            "threshold": {
                "value": 0.5,
                "fpr": 0.2,
                "tpr": 0.8,
                "precision": 0.7,
                "recall": 0.8,
            },
            "evaluation": {"sha256": _sha(evaluation.read_text())},
        }
        roc.roc_file(paths, method, build.VERSION).write_text(json.dumps(record))
        records[method] = record
    return records


def _spy_scores(monkeypatch) -> list:
    drawn = []
    draw = scores.draw_scores

    def spy(selected, stem):
        drawn.append(selected)
        return draw(selected, stem)

    monkeypatch.setattr(scores, "draw_scores", spy)
    return drawn


def test_fig_scores_draws_each_selected_models_f1_and_roc(runs, tmp_path, monkeypatch):
    records = _selected_records(runs)
    drawn = _spy_scores(monkeypatch)
    manifest = build.build(runs, tmp_path / "paper")
    assert manifest["products"]["fig_scores"] == build.files_of("fig_scores")
    assert "fig_scores" not in manifest["partial"]
    [selected] = drawn
    assert [s.category for s in selected] == list(ORDER)
    assert [s.method for s in selected] == list(roc.SELECTED.values())
    assert [s.roc for s in selected] == list(records.values())
    assert selected[0].f1 == tree.ae_evaluation()["methods"]["ae_xpower"]["f1"]
    assert [s.f1["value"] for s in selected[1:]] == [0.7, 0.6, 0.5, 0.4]
    pinned, found = manifest["inputs"], build.inputs(runs)
    for method in roc.SELECTED.values():
        for key in (build.evaluation_key(method), build.roc_key(method)):
            assert pinned[key]["path"] == str(found[key]), key
            assert pinned[key]["sha256"] == _sha(found[key].read_text()), key
    assert "frames_evaluation_hmode_frames" in pinned and "roc_ae_xpower" in pinned


def test_a_selected_model_without_its_roc_or_evaluation_is_partial(
    runs, tmp_path, monkeypatch
):
    _selected_records(runs)
    found = build.inputs(runs)
    found[build.roc_key("ntm_frames")].unlink()
    moved = found[build.evaluation_key("hmode_frames")]
    moved.write_text(moved.read_text() + "\n")  # not the evaluation its ROC names
    found[build.evaluation_key("elm_frames")].unlink()
    found[build.roc_key("elm_frames")].unlink()
    drawn = _spy_scores(monkeypatch)
    manifest = build.build(runs, tmp_path / "paper")
    assert "fig_scores" in manifest["products"]
    hmode = "high_confinement_mode"
    other = {
        "reason": build.ROC_OTHER,
        "phenomenon": hmode,
        "roc": str(found[build.roc_key("hmode_frames")]),
    }
    missing = _unscored(
        runs,
        scored=set(ORDER) - {"edge_localized_mode"},
        rocs=set(ORDER) - {"neoclassical_tearing_mode", "edge_localized_mode"},
    )
    assert manifest["partial"]["fig_scores"] == missing[:1] + [other] + missing[1:]
    [selected] = drawn
    assert [s.roc is None for s in selected] == [False, True, True, True, False]
    assert [s.f1 is None for s in selected] == [False, False, False, True, False]


def test_a_roc_without_a_pr_curve_is_partial(runs, tmp_path, monkeypatch):
    """An old roc.json (no auprc) draws its ROC, and is listed as partial."""
    _selected_records(runs)
    found = build.inputs(runs)
    old = found[build.roc_key("sawtooth_frames")]
    record = json.loads(old.read_text())
    for key in ("auprc", "pr", "positive_share"):
        del record[key]
    old.write_text(json.dumps(record))
    drawn = _spy_scores(monkeypatch)
    manifest = build.build(runs, tmp_path / "paper")
    assert "fig_scores" in manifest["products"]
    assert manifest["partial"]["fig_scores"] == [
        {
            "reason": build.NO_PR_WHY,
            "phenomenon": "sawtooth_oscillation",
            "roc": str(old),
        }
    ]
    [selected] = drawn
    assert all(s.roc is not None for s in selected)
    assert "auprc" not in selected[-1].roc


def test_the_retired_products_leave_an_older_output(runs, tmp_path):
    """fig_mhd, fig_segmentation, fig_roster_interpreter and fig_examples are
    no longer drawn, and a rebuild drops them from an older output; the
    tables and fig_examples_ae are written."""
    tree.seg_record(runs)
    out = tmp_path / "paper"
    out.mkdir()
    retired = sorted(build.RETIRED)
    for name in [*retired, "notes.txt"]:
        (out / name).write_text("an older build's")
    assert len(retired) == 8 and not build.RETIRED & build.OWNED
    assert not set(build.RETIRED_PRODUCTS) & set(build.PRODUCTS)
    manifest = build.build(runs, out)
    files = set(_files(out))
    assert not files & build.RETIRED, "dropped"
    assert (out / "notes.txt").read_text() == "an older build's", "not the build's"
    assert {"fig_examples_ae.pdf", "fig_examples_ae.png"} <= files
    for table in ("table_ae_scores", "table_seg_scores", "table_differences"):
        assert manifest["products"][table] == [f"{table}.tex"], table
        assert (out / f"{table}.tex").stat().st_size > 0, table
    assert set(manifest["products"]) <= set(build.PRODUCTS)
    assert build.SEG_PRODUCTS == ("table_seg_scores",)
