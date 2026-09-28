"""The paper's build: what the inputs allow, its manifest, the manuscript's copy."""

from __future__ import annotations

import builtins
import hashlib
import io
import json
import os
from pathlib import Path

import h5py
import pytest

from labeler.ae import xpower
from labeler.ae.xpower import gallery
from labeler.paper import build, shots

from . import ae_tree
from . import paper_tree as tree


@pytest.fixture
def runs(tmp_path, monkeypatch):
    paths = ae_tree.build(tmp_path, {101: "train", 102: "valid", 103: "valid"})
    models = ae_tree.chosen(paths, {101: "train", 102: "test", 103: "test"})
    (models / "evaluation.json").write_text(json.dumps(tree.ae_evaluation()))
    gallery.write_index(
        gallery.gallery_dir(paths) / "index.csv",
        [
            {"shot": 101, "group": "reviewed", "split": "train", "f1_vs_owner": 0.99},
            {"shot": 102, "group": "reviewed", "split": "test", "f1_vs_owner": 0.8},
            {"shot": 103, "group": "reviewed", "split": "test", "f1_vs_owner": 0.6},
        ],
    )
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
    missing = f"missing {build.inputs(runs)['seg_evaluation']}"
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
        "ae_split",
        "gallery_index",
        "store_102",
        "store_103",
    ]
    for name, files in manifest["products"].items():
        assert all((out / f).is_file() for f in files), name


def test_without_the_chosen_model_the_shot_figures_wait(runs, tmp_path):
    (xpower.model_dir(runs) / "chosen.json").unlink()
    manifest = build.build(runs, tmp_path / "paper")
    missing = f"missing {build.inputs(runs)['ae_chosen']}"
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
    no_summary, no_poi = [str(found["summary"])], [str(found["poi"])]
    assert manifest["partial"] == {
        "fig_coverage": no_summary,
        "table_datasets": no_summary,
        "fig_interpreter": no_poi,
        "fig_examples": no_poi,
        "table_differences": [str(found["seg_evaluation"])],
    }
    assert set(manifest["partial"]) <= set(manifest["products"])
    (xpower.model_dir(runs) / "chosen.json").unlink()
    manifest = build.build(runs, tmp_path / "paper")
    no_split = [str(found["ae_chosen"]), *no_summary]
    assert manifest["partial"]["fig_coverage"] == no_split
    assert manifest["partial"]["table_datasets"] == no_split


def _spy_reads(monkeypatch, opened: set[str]) -> None:
    """Record every file opened for reading through Python, pathlib, pandas,
    torch and h5py."""

    def wrap(real):
        def spy(file, mode="r", *args, **kwargs):
            if isinstance(file, str | os.PathLike) and not set(mode) & set("wax+"):
                opened.add(str(Path(file).resolve()))
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
    opened: set[str] = set()
    _spy_reads(monkeypatch, opened)
    out = tmp_path / "paper"
    manifest = build.build(runs, out)
    tree_root = str(tmp_path.resolve())
    read = {p for p in opened if p.startswith(tree_root) and not p.startswith(str(out))}
    pinned = {str(Path(v["path"]).resolve()) for v in manifest["inputs"].values()}
    assert read and read <= pinned, sorted(read - pinned)
    assert {"ae_model", "ae_split", "store_102", "store_103"} <= set(manifest["inputs"])
    for entry in manifest["inputs"].values():
        data = Path(entry["path"]).read_bytes()
        assert entry["sha256"] == hashlib.sha256(data).hexdigest()


def test_the_manifest_records_the_commit_and_the_labels_check(runs, tmp_path):
    manifest = build.build(runs, tmp_path / "paper")
    assert manifest["git_sha"] == "unknown" or len(manifest["git_sha"]) == 40
    assert manifest["git_dirty"] in (True, False, None)
    assert manifest["version"] == "v1"
    assert manifest["labels_match"] is None, "no record names the labels' sha256"
    evaluation = xpower.model_dir(runs) / "evaluation.json"
    record = json.loads(evaluation.read_text())
    live = build.inputs(runs)["ae_labels"]
    record["meta"]["labels_sha256"] = hashlib.sha256(live.read_bytes()).hexdigest()
    evaluation.write_text(json.dumps(record))
    assert build.build(runs, tmp_path / "paper")["labels_match"] is True
    record["meta"]["labels_sha256"] = "0" * 64
    evaluation.write_text(json.dumps(record))
    manifest = build.build(runs, tmp_path / "paper")
    assert manifest["labels_match"] is False, "recorded, not refused"
    assert "fig_scores" in manifest["products"]


def test_the_version_names_the_models_drawn(runs, tmp_path, capsys):
    out = tmp_path / "paper"
    assert build.main(["--out", str(out), "--version", "v2"]) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["version"] == "v2"
    assert "/ae_xpower/v2/" in manifest["skipped"]["fig_scores"]
    assert "fig_coverage" in manifest["products"], "the owner's labels have no version"
