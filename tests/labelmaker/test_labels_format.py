"""Raw adapters preserve originals and reproduce the committed format bytes."""
import hashlib
import json
import runpy
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest
import yaml

from labelmaker.events import databases as db

REPO = Path(__file__).resolve().parents[2]


def converter():
    return runpy.run_path(str(REPO / "scripts/labelmaker/labels_format.py"))["main"]


def test_rwm_raw_bytes_match_original_commit_and_format_regenerates(tmp_path):
    root = tmp_path / "labels"
    shutil.copytree(REPO / "data/labels", root)
    snapshots = {}
    for spec in db.load_manifest(root):
        raw = spec.raw_path(root)
        original = subprocess.check_output([
            "git", "show", f"6de489d:data/labels/{spec.dir}/{spec.raw_file}",
        ], cwd=REPO)
        assert raw.read_bytes() == original
        formatted = spec.path(root)
        for path in (formatted, formatted.with_suffix(".meta.json")):
            snapshots[path] = path.read_bytes()
            path.unlink()
    assert converter()(["--root", str(root)]) == 0
    assert {p: p.read_bytes() for p in snapshots} == snapshots
    assert converter()(["--root", str(root)]) == 0
    assert {p: p.read_bytes() for p in snapshots} == snapshots
    for spec in db.load_manifest(root):
        raw = spec.raw_path(root)
        formatted = spec.path(root)
        meta = json.loads(formatted.with_suffix(".meta.json").read_text())
        assert meta["made_from"] == {
            "raw_file": str(raw.relative_to(root)),
            "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        }
        assert meta["n_rows"] in (30, 26)
        assert meta["n_shots"] in (20, 13)


def raw_fixture(root, *, kind="point", **over):
    entry = {"stem": "fixture", "dir": "resistive_wall_mode", "phenomenon": "rwm",
                 "kind": kind, "shot_col": "SHOT", "t_col": "TIME", "t0_col": "TIME",
                 "t1_col": "END", "t_units": "ms", "attr_cols": ["NTOR", "NOTE"],
                 "attr_types": {"NTOR": "int", "NOTE": "str"}, "provenance": "test",
                 "raw_file": "original.csv", "format_stem": "normalized",
                 "converter": "csv", "made_at": "2026-09-13T00:00:00Z"}
    entry.update(over)
    (root / "tables.yaml").write_text(yaml.safe_dump({"version": 1,
                                                     "tables": [entry]}))
    raw = root / entry["dir"] / "raw" / entry["raw_file"]
    raw.parent.mkdir(parents=True)
    raw.write_text("SHOT,TIME,END,NTOR,NOTE\n158015,2613,3000,2,NA\n"
                   "156785,856,1000,1,rwm\n158015,2613,3000,2,NA\n")
    return raw, root / entry["dir"] / "format/normalized.csv"


@pytest.mark.parametrize("kind", ["point", "interval"])
def test_adapter_sorts_preserves_duplicates_attributes_and_raw(tmp_path, kind):
    raw, out = raw_fixture(tmp_path, kind=kind)
    before = raw.read_bytes()
    assert converter()(["--root", str(tmp_path)]) == 0
    assert raw.read_bytes() == before
    frame = pd.read_csv(out)
    db.validate_format(frame)
    assert frame.shot.tolist() == [156785, 158015, 158015]
    assert frame.t0_s.tolist() == [0.856, 2.613, 2.613]
    assert frame.t1_s.tolist() == ([1, 3, 3] if kind == "interval"
                                  else [0.856, 2.613, 2.613])
    assert frame.confidence.isna().all()
    assert json.loads(frame["attrs"].iloc[1]) == {
        "NTOR": 2, "NOTE": "NA", "table": "fixture",
    }


@pytest.mark.parametrize("over, match", [
    ({"raw_file": "../escape.csv"}, "raw_file"),
    ({"format_stem": "../escape"}, "format_stem"),
    ({"converter": ""}, "converter"),
    ({"made_at": "yesterday"}, "made_at"),
])
def test_manifest_validates_converter_fields(tmp_path, over, match):
    raw_fixture(tmp_path)
    path = tmp_path / "tables.yaml"
    manifest = yaml.safe_load(path.read_text())
    manifest["tables"][0].update(over)
    path.write_text(yaml.safe_dump(manifest))
    with pytest.raises(db.DatabaseError, match=match):
        db.load_manifest(tmp_path)


@pytest.mark.parametrize("text, match", [
    ("SHOT,WRONG,NTOR,NOTE\n158015,1,2,x\n", "TIME"),
    ("SHOT,TIME,NTOR,NOTE\n158015,soon,2,x\n", "TIME"),
    ("SHOT,TIME,NTOR,NOTE\n158015.5,1000,2,x\n", "shot"),
    ("SHOT,TIME,NTOR,NOTE\n158015,1000,1.5,x\n", "NTOR"),
])
def test_bad_raw_data_is_rejected_without_outputs(tmp_path, text, match):
    raw, out = raw_fixture(tmp_path)
    raw.write_text(text)
    with pytest.raises(db.DatabaseError, match=match):
        converter()(["--root", str(tmp_path)])
    assert raw.read_text() == text
    assert not out.exists()
