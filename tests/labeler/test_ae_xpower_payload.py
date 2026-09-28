"""Merge validates the payload it will publish, including failure accounting."""

import json

import numpy as np
import pandas as pd
import pytest

from labeler.ae.xpower import extend, gallery
from labeler.config import Paths
from labeler.events.review.rows import Grid

from . import ae_tree
from .test_ae_xpower_extend import _three_shots


@pytest.mark.parametrize(
    "damage",
    [
        "missing_rows",
        "extra_rows",
        "window",
        "frames",
        "present_frames",
        "not_observable_frames",
        "present_runs",
        "missing_array",
        "extra_array",
        "array_length",
        "array_first",
    ],
)
def test_merge_refuses_a_damaged_shot_payload(tmp_path, monkeypatch, capsys, damage):
    paths, models = _three_shots(tmp_path, monkeypatch)
    ae_tree.env(monkeypatch, paths)
    for k in (0, 1):
        extend.run_shard(paths, models=models, k=k, of=2, pictures=False)
    shards = extend.suggestions_dir(paths) / "shards"
    shot = 203
    if damage.endswith("rows") or damage == "window":
        file = shards / "0.csv"
        rows = pd.read_csv(file)
        if damage == "missing_rows":
            rows = rows[rows.shot != shot]
        elif damage == "extra_rows":
            shot = 999
            rows.loc[len(rows)] = [shot, 1, 0, 600, 0.9]
        else:
            rows.loc[rows.shot == shot, "t_end"] = 590
        rows.to_csv(file, index=False)
    elif "array" in damage:
        file = shards / "0.npz"
        with np.load(file) as archive:
            arrays = dict(archive)
        if damage == "missing_array":
            del arrays[f"p{shot}"]
        elif damage == "extra_array":
            shot = 999
            arrays[f"p{shot}"] = np.zeros(60)
        elif damage == "array_length":
            arrays[f"p{shot}"] = arrays[f"p{shot}"][:-1]
        else:
            arrays[f"f{shot}"] = np.int64(1)
        np.savez_compressed(file, **arrays)
    else:
        file = shards / "0.summary.csv"
        summary = pd.read_csv(file)
        summary.loc[summary.shot == shot, damage] += 1
        summary.to_csv(file, index=False)
    with pytest.raises(ValueError) as error:
        extend.merge(paths, models=models, of=2)
    assert "0." in str(error.value) and str(shot) in str(error.value)
    with pytest.raises(SystemExit) as error:
        extend.main(["--merge", "--of", "2"])
    assert error.value.code != 0
    stderr = capsys.readouterr().err
    assert "0." in stderr and str(shot) in stderr and "Traceback" not in stderr
    assert not list(extend.suggestions_dir(paths).glob("*_suggest_*.csv"))


@pytest.mark.parametrize("failures", [1, 2])
def test_failure_json_lines_roundtrip_and_two_percent_gate(
    tmp_path, monkeypatch, failures
):
    paths, models = _three_shots(tmp_path, monkeypatch)
    ae_tree.env(monkeypatch, paths)
    shots = list(range(201, 251))
    population = paths.catalog / "population.csv"
    template = pd.read_csv(population).iloc[0].to_dict()
    pd.DataFrame([{**template, "shot": s} for s in shots]).to_csv(
        population, index=False
    )
    work = extend.run_all

    def failing(*args):
        for job, outcome in work(*args):
            if job[0] in shots[:failures]:
                outcome = RuntimeError("first line\nsecond line\twith a tab")
            yield job, outcome

    monkeypatch.setattr(extend, "run_all", failing)
    extend.run_shard(paths, models=models, k=0, of=1, pictures=False)
    out = extend.suggestions_dir(paths)
    failure_file = out / "shards/0.failed.jsonl"
    assert failure_file.is_file()
    records = [json.loads(line) for line in failure_file.read_text().splitlines()]
    assert records == [
        {"shot": s, "error": "RuntimeError: first line\nsecond line\twith a tab"}
        for s in shots[:failures]
    ]
    if failures == 1:
        result = extend.merge(paths, models=models, of=1)
        assert result["shots"] == 49 and result["failed"] == 1
        assert (out / "failed.jsonl").read_bytes() == failure_file.read_bytes()
    else:
        with pytest.raises(ValueError, match="2 %"):
            extend.merge(paths, models=models, of=1)
        assert not list(out.glob("*_suggest_*.csv"))


def test_extension_picture_marks_the_scored_window(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    paths = Paths(
        root=tmp_path / "root",
        label_tables=tmp_path / "events",
        corpus=tmp_path / "corpus",
    )
    ae_tree.env(monkeypatch, paths)
    figures = []

    def savefig(fig, path, **kwargs):
        figures.append(fig)
        path.write_bytes(b"picture")

    monkeypatch.setattr(Figure, "savefig", savefig)
    gallery.draw(
        paths.root / "extension.jpg",
        title="extension",
        grid=Grid(0, 10, 300),
        values=np.zeros((3, 257, 300), dtype=np.uint8),
        y0=0,
        dy=500 / 512,
        first=0,
        prob=np.zeros(300),
        threshold=0.5,
    )
    for ax in figures[0].axes:
        lines = [line for line in ax.lines if line.get_label() == "scored: 0-2 s"]
        assert len(lines) == 1
        assert list(lines[0].get_xdata()) == [2000, 2000]
        assert lines[0].get_linestyle() == "--"
        assert any(t.get_text() == "scored: 0-2 s" for t in ax.texts)
