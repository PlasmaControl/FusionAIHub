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

from labeler.events import databases as db
from labeler.events.interval_tables import validate_intervals

REPO = Path(__file__).resolve().parents[2]


def converter():
    return runpy.run_path(str(REPO / "scripts/labeler/labels_format.py"))["main"]


def test_rwm_raw_bytes_match_original_commit_and_format_regenerates(tmp_path):
    root = tmp_path / "labels"
    root.mkdir()
    manifest = yaml.safe_load((REPO / "data/events/events.yaml").read_text())
    manifest["format_datasets"] = [r for r in manifest["format_datasets"]
                                   if r["name"] == "resistive_wall_mode"]
    manifest["raw_datasets"] = [r for r in manifest["raw_datasets"]
                                if r["stem"].startswith("rwm_onsets_")]
    (root / "events.yaml").write_text(yaml.safe_dump(manifest))
    for row in manifest["raw_datasets"]:
        raw = root / row["path"]
        raw.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / "data/events" / row["path"], raw)
        original = subprocess.check_output([
            "git", "show", f"6de489d:data/labels/resistive_wall_mode/{raw.name}",
        ], cwd=REPO)
        assert raw.read_bytes() == original
    assert converter()(["--root", str(root)]) == 0
    (spec,) = db.load_manifest(root)
    out = spec.path(root)
    snapshots = out.read_bytes(), out.with_suffix(".meta.json").read_bytes()
    assert converter()(["--root", str(root)]) == 0
    assert snapshots == (out.read_bytes(), out.with_suffix(".meta.json").read_bytes())
    meta = json.loads(snapshots[1])
    assert meta["n_rows"] == 56 and meta["n_shots"] == 33
    for source in meta["made_from"]:
        assert source["sha256"] == hashlib.sha256((root/source["raw_file"]).read_bytes()).hexdigest()


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
    validate_intervals(frame)
    assert frame.shot.tolist() == [156785, 158015, 158015]
    assert frame.t_start.tolist() == [856, 2613, 2613]
    assert frame.t_end.tolist() == ([1000, 3000, 3000] if kind == "interval"
                                      else [856, 2613, 2613])
    assert frame.confidence.isna().all()
    assert list(frame.columns) == ["shot", "category", "t_start", "t_end", "confidence"]


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
