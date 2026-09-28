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
from labeler.paper import build, coverage, shots

from . import ae_tree
from . import paper_tree as tree


@pytest.fixture
def runs(tmp_path, monkeypatch):
    paths = ae_tree.build(tmp_path, {101: "train", 102: "valid", 103: "valid"})
    models = ae_tree.chosen(paths, {101: "train", 102: "test", 103: "test"})
    (models / "evaluation.json").write_text(json.dumps(tree.ae_evaluation()))
    tree.record_labels(paths)
    ae_tree.env(monkeypatch, paths)
    return paths


def test_the_build_draws_what_its_inputs_allow(runs, tmp_path, capsys):
    out, dest = tmp_path / "paper", tmp_path / "manuscript" / "figures"
    assert build.main(["--out", str(out), "--copy-to", str(dest)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["products"] == [
        "fig_coverage",
        "fig_examples",
        "fig_interpreter",
        "fig_mhd",
        "fig_scores",
        "table_ae_scores",
        "table_datasets",
        "table_differences",
    ]
    missing = _missing(build.inputs(runs)["seg_evaluation"])
    assert printed["skipped"] == {
        "fig_segmentation": missing,
        "table_seg_scores": missing,
    }
    assert printed["copied"] == sorted(p.name for p in dest.iterdir())
    assert len(printed["copied"]) == 8
    assert all(name.endswith((".pdf", ".tex")) for name in printed["copied"])
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["interpreter_shot"] == 102, "the best test shot; no POI yet"
    assert manifest["example_shots"] == [102, 103]
    assert manifest["interpreter_rule"] == shots.INTERPRETER_RULE
    assert manifest["interpreter_branch"] == shots.POOL_GAP, "AE ends at 900 ms"
    pool = manifest["interpreter_pool"]
    assert sorted(pool) == ["102", "103"]
    assert set(pool["102"]) == {"f1", "poi", "absent_after_onset"}
    assert pool["102"]["absent_after_onset"] == 110, "900 ms to 2 s"
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


def _missing(*paths) -> dict:
    return {"reason": build.MISSING, "missing": [str(p) for p in paths]}


def test_without_the_chosen_model_the_shot_figures_wait(runs, tmp_path):
    (xpower.model_dir(runs) / "chosen.json").unlink()
    manifest = build.build(runs, tmp_path / "paper")
    missing = _missing(build.inputs(runs)["ae_chosen"])
    assert manifest["skipped"]["fig_interpreter"] == missing
    assert manifest["skipped"]["fig_examples"] == missing
    assert "fig_coverage" in manifest["products"], "the split is left empty"
    assert "interpreter_shot" not in manifest


def test_the_interpreter_shot_can_be_named(runs, tmp_path):
    manifest = build.build(runs, tmp_path / "paper", shot=103, examples=1)
    assert (manifest["interpreter_shot"], manifest["example_shots"]) == (103, [102])
    assert manifest["interpreter_rule"] == "named by --shot"


def test_products_drawn_without_an_input_are_recorded_as_partial(runs, tmp_path):
    found = build.inputs(runs)
    manifest = build.build(runs, tmp_path / "paper")
    no_summary = {
        "reason": "extension not run: A2 failed (D47)",
        "missing": [str(found["summary"])],
    }
    no_poi = {"reason": build.NO_POI, "missing": [str(found["poi"])]}
    assert manifest["partial"] == {
        "fig_coverage": [no_summary],
        "table_datasets": [no_summary],
        "fig_interpreter": [no_poi],
        "fig_examples": [no_poi],
        "table_differences": [_missing(found["seg_evaluation"])],
    }
    assert set(manifest["partial"]) <= set(manifest["products"])
    (xpower.model_dir(runs) / "chosen.json").unlink()
    manifest = build.build(runs, tmp_path / "paper")
    no_split = {"reason": build.NO_CHOSEN, "missing": [str(found["ae_chosen"])]}
    assert manifest["partial"]["fig_coverage"] == [no_split, no_summary]
    assert manifest["partial"]["table_datasets"] == [no_split, no_summary]


def test_the_extension_reason_comes_from_the_bar(runs, tmp_path):
    evaluation = xpower.model_dir(runs) / "evaluation.json"
    record = json.loads(evaluation.read_text())
    record["bar"].update(A1=False, A2=False)
    evaluation.write_text(json.dumps(record))
    [entry] = build.build(runs, tmp_path / "paper")["partial"]["fig_coverage"]
    assert entry["reason"] == "extension not run: A1 and A2 failed (D47)"
    record["bar"].update(A1=True, A2=True)
    evaluation.write_text(json.dumps(record))
    [entry] = build.build(runs, tmp_path / "paper")["partial"]["fig_coverage"]
    assert entry["reason"] == build.NO_SUMMARY, "A1 and A2 pass: not written yet"
    evaluation.unlink()
    [entry] = build.build(runs, tmp_path / "paper")["partial"]["fig_coverage"]
    assert entry["reason"] == build.NO_EVALUATION


def _files(out: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(out.iterdir())}


def test_labels_missing_entirely_skip_and_never_crash(runs, tmp_path):
    live = build.inputs(runs)["ae_labels"]
    live.unlink()
    out = tmp_path / "paper"
    manifest = build.build(runs, out)
    for product in ("fig_coverage", "table_datasets"):
        assert manifest["skipped"][product] == _missing(live)
    assert "fig_examples" in manifest["products"], "from the model's own copy"
    copy = tree.scored_labels(runs)
    copy.unlink()
    manifest = build.build(runs, out)
    for product in ("fig_interpreter", "fig_examples"):
        assert manifest["skipped"][product] == _missing(copy)
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
    for product in ("fig_interpreter", "fig_examples"):
        assert product in manifest["products"]
        assert {"reason": build.NO_LABEL, "shots": [103]} in manifest["partial"][
            product
        ]
    assert manifest["example_shots"] == [102]
    assert sorted(manifest["interpreter_pool"]) == ["102"]


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
    monkeypatch.setattr(shots, "draw_examples", fails)
    with pytest.raises(RuntimeError, match="drawing failed"):
        build.build(runs, out)
    assert _files(out) == before
    assert sorted(p.name for p in tmp_path.iterdir() if "paper" in p.name) == ["paper"]
    monkeypatch.undo()
    (xpower.model_dir(runs) / "chosen.json").unlink()
    manifest = build.build(runs, out)
    after = _files(out)
    assert "fig_examples.pdf" not in after, "a skipped product leaves with its files"
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
    assert homes == [error.old] and str(error.old) in str(error), "named"
    assert str(error.new) in str(error), "the new output is named too"
    new = _tree(error.new)
    assert "manifest.json" in new and new["notes.txt"] == before["notes.txt"]
    assert _tree(out) == {"fig_scores.pdf": b"the other build's"}, "not touched"


def test_only_the_builds_own_files_leave_out(runs, tmp_path):
    """The build claims the manifest and its products' own files, by exact
    name: an owner's `fig_scores.svg`, or a `fig_scores/` directory, stays."""
    out = tmp_path / "paper"
    build.build(runs, out, examples=1)
    (out / "fig_scores.svg").write_text("the owner's own drawing")
    (out / "fig_scores").mkdir()
    (out / "fig_scores" / "notes.txt").write_text("the owner's notes")
    theirs = {
        k: v
        for k, v in _tree(out).items()
        if k in ("fig_scores.svg", "fig_scores", "fig_scores/notes.txt")
    }
    manifest = build.build(runs, out)
    after = _tree(out)
    assert {k: after.get(k, "gone") for k in theirs} == theirs
    drawn = {f for files in manifest["products"].values() for f in files}
    assert set(after) == drawn | {"manifest.json"} | set(theirs)


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
    poi = build.inputs(runs)["poi"]
    poi.parent.mkdir(parents=True)
    poi.write_text(
        "shot,region,t_start_ms,t_end_ms,f_lo_khz,f_hi_khz,pixels,in_scored_window\n"
        "102,1,300,900,140,152,40,True\n"
    )
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
    assert {"ae_model", "ae_split", "store_102", "store_103"} <= set(manifest["inputs"])
    for entry in manifest["inputs"].values():
        data = Path(entry["path"]).read_bytes()
        assert entry["sha256"] == hashlib.sha256(data).hexdigest()


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
    for product in ("fig_interpreter", "fig_examples"):
        entry = manifest["skipped"][product]
        assert entry["reason"].startswith(build.LABELS_DIFFER), entry
    assert "interpreter_shot" not in manifest
    _name_labels(runs, None)
    manifest = build.build(runs, tmp_path / "paper")
    assert manifest["labels_match"] is None
    assert manifest["skipped"]["fig_examples"]["reason"] == build.LABELS_UNNAMED


def test_the_scored_products_use_the_labels_the_model_was_scored_on(runs, tmp_path):
    before = build.build(runs, tmp_path / "before")
    live = build.inputs(runs)["ae_labels"]
    rows = [r for r in live.read_text().splitlines() if not r.startswith("102,")]
    rows += ["102,0,0,300,", "102,1,300,1500,", "102,0,1500,2000,"]  # changed
    rows += ["104,0,0,500,", "104,1,500,700,", "104,0,700,2000,"]  # in no split
    live.write_text("\n".join(rows) + "\n")
    after = build.build(runs, tmp_path / "after")
    for key in ("shot_f1", "interpreter_pool", "example_shots", "interpreter_shot"):
        assert after[key] == before[key], key
    assert after["labels_match"] is True, "the copy is the evaluation's"
    shas = after["labels_sha256"]
    assert shas["live"] != shas["scored"] == shas["ae_evaluation"]
    table = (tmp_path / "after" / "table_datasets.tex").read_text().splitlines()
    assert table[5].startswith("AE & 4 & 4 & 2.6 & 1 / 0 / 2 & 1 & "), table[5]


AE_PRODUCTS = (
    "fig_scores",
    "fig_mhd",
    "table_ae_scores",
    "fig_segmentation",
    "table_seg_scores",
    "table_differences",
    "fig_interpreter",
    "fig_examples",
)


FRAME_PRODUCTS = (
    "fig_scores",
    "fig_mhd",
    "table_ae_scores",
    "fig_interpreter",
    "fig_examples",
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
        "fig_segmentation",
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
    assert manifest["skipped"]["fig_interpreter"]["missing"] == [
        str(evaluation),
        str(chosen),
    ]
    assert manifest["skipped"]["fig_scores"]["missing"] == [str(evaluation)]
    assert manifest["partial"]["table_differences"] == [_missing(evaluation)]
    assert {"reason": build.NO_CHOSEN, "missing": [str(chosen)]} in manifest["partial"][
        "fig_coverage"
    ], "the owner's labels have no version"
    frame = [k for k in manifest["inputs"] if not k.startswith(("seg_", "poi"))]
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
    assert manifest["skipped"] == {}
    assert set(AE_PRODUCTS) <= set(manifest["products"])
    assert manifest["labels_match"] is True
    pinned = manifest["inputs"]
    for key in ("ae_evaluation", "ae_chosen", "ae_model", "ae_split"):
        assert pinned[key]["path"].startswith(str(models)), key
    assert pinned["ae_scored_labels"]["path"] == str(
        models / "band80-mhd3" / "review" / "labels.csv"
    )
    assert manifest["interpreter_shot"] == 102


def _seg_products_of(manifest: dict) -> dict:
    return {
        k: manifest["partial"].get(k)
        for k in ("fig_segmentation", "table_seg_scores", "table_differences")
    }


def test_the_segmentation_keeps_its_own_version(runs, tmp_path):
    """v2 changes the frame model only: SegNet v1 is drawn beside it, pinned and
    checked against its own record's labels, not the frame model's."""
    sha = tree.seg_record(runs, "v1")
    models = tree.as_version(runs, "v2", keep=True)
    live = tree.scored_labels(runs, "v2")
    live.write_text(live.read_text() + "104,0,0,2000,\n")  # v2 has more labels
    frame = tree.record_labels(runs, "v2")
    assert frame != sha
    out = tmp_path / "paper"
    argv = ["--out", str(out), "--version", "v2", "--seg-version", "v1"]
    assert build.main(argv) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert (manifest["version"], manifest["seg_version"]) == ("v2", "v1")
    assert manifest["skipped"] == {}
    assert {"fig_segmentation", "table_seg_scores", "table_differences"} <= set(
        manifest["products"]
    )
    assert _seg_products_of(manifest) == dict.fromkeys(_seg_products_of(manifest))
    seg = runs.root / "models" / "ae_seg" / "v1"
    pinned = manifest["inputs"]
    assert pinned["seg_evaluation"]["path"] == str(seg / "evaluation.json")
    assert pinned["seg_labels"]["path"] == str(seg / "review" / "labels.csv")
    assert pinned["poi"]["path"].endswith("/ae_seg-v1/poi.csv")
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
    assert manifest["interpreter_pool"]["102"]["poi"] == 1, "the v1 POI boxes"
    for product in ("fig_interpreter", "fig_examples"):
        assert product not in manifest["partial"], manifest["partial"]


def test_the_segmentation_is_checked_against_its_own_copy(runs, tmp_path):
    tree.seg_record(runs)
    seg = runs.root / "models" / "ae_seg" / "v1"
    copy = seg / "review" / "labels.csv"
    copy.write_text(copy.read_text() + "104,0,0,2000,\n")
    manifest = build.build(runs, tmp_path / "paper")
    assert manifest["seg_version"] == "v1"
    assert manifest["labels_match"] is True
    assert manifest["seg_labels_match"] is False
    for product in ("fig_segmentation", "table_seg_scores"):
        assert product in manifest["products"], "the scores are the record's own"
        (entry,) = manifest["partial"][product]
        assert entry["reason"].startswith(build.SEG_LABELS_DIFFER), entry
    copy.unlink()
    manifest = build.build(runs, tmp_path / "paper")
    assert manifest["seg_labels_match"] is None
    no_copy = {"reason": build.NO_SEG_COPY, "missing": [str(copy)]}
    assert manifest["partial"]["fig_segmentation"] == [no_copy]


def test_a_seg_version_not_run_skips_only_the_segmentation(runs, tmp_path):
    tree.seg_record(runs)
    tree.as_version(runs, "v2")
    out = tmp_path / "paper"
    argv = ["--out", str(out), "--version", "v2", "--seg-version", "v9"]
    assert build.main(argv) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    seg = str(build.inputs(runs, "v2", "v9")["seg_evaluation"])
    assert "/ae_seg/v9/" in seg
    assert manifest["skipped"] == {
        "fig_segmentation": _missing(seg),
        "table_seg_scores": _missing(seg),
    }
    assert manifest["partial"]["table_differences"] == [_missing(seg)]
    assert {"fig_scores", "fig_interpreter", "fig_examples"} <= set(
        manifest["products"]
    )
