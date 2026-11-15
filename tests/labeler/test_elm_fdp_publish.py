"""`elm_fdp_label.py publish` of an instance run: the span layer and the individual
layer (`individual.parquet`), matched to each shot's labeled job, and the guards that
keep two model runs out of one folder."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts/labeler"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "elm_fdp_label", SCRIPTS / "elm_fdp_label.py"
)
E = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = E
spec.loader.exec_module(E)

HEADER = (
    "shot,status,window_start_ms,window_end_ms,present_ms,n_spans,plasma_end_ms,"
    "fs_hz,density_hz,mean_prob,job,error\n"
)


def _models(path: Path) -> Path:
    for k in range(5):
        (path / f"fold{k}").mkdir(parents=True)
        fold = {"occ_threshold": 0.4 + 0.05 * k, "onset_threshold": 0.5}
        (path / f"fold{k}/fold.json").write_text(json.dumps(fold))
        (path / f"fold{k}/model.pt").write_bytes(bytes([k]))
    decoding = {"name": "elm-inst2-sized", "record": "key2_record.json", "gate": None}
    decoding |= {"onset_threshold": 0.5, "min_sep_ms": 3.0, "event_duration_ms": 1.1}
    (path / "decoding.json").write_text(json.dumps(decoding))
    (path / "run.json").write_text(json.dumps({"name": "elm-inst2"}))
    return path


@pytest.fixture
def run(tmp_path, monkeypatch):
    root = tmp_path / "root"
    (root / "results").mkdir(parents=True)
    (root / "logs").mkdir()
    res = root / "results"
    # job A labels 1 and 2; job B relabels 2 (B wins), its onsets file cut by a kill
    (res / "A.shots.csv").write_text(
        HEADER + "190001,labeled,100,4000,300,2,5000,5e4,1e5,0.2,A,\n"
        "190002,labeled,100,4000,100,1,5000,5e4,1e5,0.1,A,\n"
    )
    (res / "A.spans.csv").write_text(
        "shot,t_start_ms,t_end_ms\n190001,1000,1200\n190001,2000,2100\n190002,500,600\n"
    )
    (res / "A.onsets.csv").write_text(
        "shot,t_ms\n190001,1000.05\n190001,1199.95\n190001,1200.05\n190002,550.15\n"
    )
    (res / "B.shots.csv").write_text(
        HEADER + "190002,labeled,100,4000,50,1,5000,5e4,1e5,0.1,B,\n"
        '190003,infer_failed,,,,,,,,,B,"boom"\n'
    )
    (res / "B.spans.csv").write_text("shot,t_start_ms,t_end_ms\n190002,700,750\n")
    (res / "B.onsets.csv").write_text("shot,t_ms\n190002,720.35\n190002,7")
    cands = tmp_path / "cands.csv"
    pd.DataFrame({"shot": [190001, 190002, 190003], "end_ms": 5000}).to_csv(
        cands, index=False
    )
    monkeypatch.setattr(E, "ROOT", root)  # main() moves both; undone after the test
    monkeypatch.setattr(E.ae, "ROOT", root)
    monkeypatch.setattr(E.ae, "snapshot_range", lambda c: (190000, 190005))
    monkeypatch.setattr(E, "_git_sha", lambda: "test")
    return root, cands, _models(tmp_path / "models"), tmp_path / "out"


def _publish(root, cands, out, *extra):
    E.main(
        [
            "--root",
            str(root),
            "publish",
            "--candidates",
            str(cands),
            "--out",
            str(out),
            *extra,
        ]
    )


def test_publish_writes_the_span_and_the_individual_layer(run):
    root, cands, models, out = run
    _publish(root, cands, out, "--models", str(models))
    ind = pd.read_parquet(out / "individual.parquet")
    # 2's onsets come from job B only; the cut line (7 ms, outside the window) is dropped
    assert ind.shot.tolist() == [190001] * 3 + [190002]
    assert ind.t_ms.tolist() == [1000.05, 1199.95, 1200.05, 720.35]
    assert ind.in_span.tolist() == [True, True, False, True]  # spans are [start, end)
    shots = pd.read_parquet(out / "shots.parquet").set_index("shot")
    assert shots.n_elms[190001] == 3 and shots.n_elms[190002] == 1
    assert pd.isna(shots.n_elms[190003])
    meta = json.loads((out / "meta.json").read_text())
    assert (meta["n_elms"], meta["n_elms_in_span"]) == (4, 3)
    assert meta["model"]["individual"]["decoding"] == "elm-inst2-sized"
    assert meta["model"]["threshold"] == pytest.approx(0.5)
    assert "individual.parquet" in meta["files"]
    assert pd.read_parquet(out / "spans.parquet").shot.tolist() == [
        190001,
        190001,
        190002,
    ]


def test_one_folder_takes_one_model_run(run, tmp_path):
    root, cands, models, out = run
    with pytest.raises(SystemExit, match="pass its folder with --models"):
        _publish(root, cands, out)
    _publish(root, cands, out, "--models", str(models))
    other = _models(tmp_path / "other")
    with pytest.raises(SystemExit, match="another model run"):
        _publish(root, cands, out, "--models", str(other))


def test_infer_stops_on_work_files_of_another_model(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    np.savez(work / "190001.npz", models=str(tmp_path / "student"))
    with pytest.raises(SystemExit, match="move"):
        E.check_work_model(work, tmp_path / "inst2")
    E.check_work_model(work, tmp_path / "student")
