"""D2e: a cold Thomson plateau is excluded, with a reproducible input record."""

import hashlib
import json

import h5py
import numpy as np
import pandas as pd
import pytest

from labeler.config import Paths
from labeler.events.catalog import population as pop
from labeler.events.catalog import window


def _module():
    from labeler.events.catalog import runaway

    return runaway


def _h5(path, x, y, scale=1.0):
    with h5py.File(path, "w") as f:
        for name, values in {
            "ts_core_temp": y,
            "pinj": [[1000, 2000, 3000, 4000, 5000], [0, 1000, 0, 1000, 0]],
            "neutron_rate": [[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]],
        }.items():
            g = f.create_group(name)
            g["xdata"] = np.asarray(x) * scale
            g["ydata"] = values


@pytest.mark.parametrize("scale", [1.0, 0.001], ids=["ms", "seconds"])
def test_statistic_uses_positive_finite_channels_then_window_median(tmp_path, scale):
    path = tmp_path / "1_processed.h5"
    _h5(
        path,
        [0, 100, 200, 300, 400],
        [
            [10000, 10, 20, -1, 10000],
            [10000, 30, np.nan, 0, 10000],
            [10000, 0, 40, np.inf, 10000],
        ],
        scale,
    )
    row, digest = _module().assess(path, 1, 100, 300)
    # p90(10,30)=28; p90(20,40)=38; no valid value at 300 ms.
    assert row["te_p90_ev"] == 33.0
    assert row["n_thomson"] == 2 and row["runaway"] is True
    assert row["n_profile"] == 2
    assert row["pinj_kw"] == pytest.approx(11 / 3)
    assert row["neutron_rate_mean"] == "3.0;30.0"
    assert len(digest) == 64


@pytest.mark.parametrize("te,marked", [(59.999, True), (60.0, False), (60.001, False)])
def test_threshold_is_strictly_below_60_ev(tmp_path, te, marked):
    path = tmp_path / "1_processed.h5"
    _h5(path, [0, 100, 200, 300, 400], [[te] * 5])
    row, _ = _module().assess(path, 1, 100, 300)
    assert row["runaway"] is marked


@pytest.mark.parametrize("channels,valid", [(4, 1), (4, 2), (5, 2), (5, 3)])
def test_profile_requires_at_least_half_the_channels(tmp_path, channels, valid):
    path = tmp_path / "1_processed.h5"
    values = np.full((channels, 5), np.nan)
    values[:valid] = 10
    _h5(path, [0, 100, 200, 300, 400], values)
    row, _ = _module().assess(path, 1, 100, 300)
    usable = (channels, valid) in ((4, 2), (5, 3))
    assert row["runaway"] is usable
    assert row["n_thomson"] == 3
    assert row["n_profile"] == (3 if usable else 0)
    assert row["te_p90_ev"] == (10.0 if usable else None)


def test_one_profile_sample_keeps_the_statistic_over_all_valid_samples(tmp_path):
    path = tmp_path / "1_processed.h5"
    _h5(
        path,
        [0, 100, 200, 300, 400],
        [
            [1000, 40, 10, 50, 1000],
            [1000, np.nan, 10, np.inf, 1000],
            [1000, 0, 0, -1, 1000],
            [1000, -1, np.nan, 0, 1000],
        ],
    )
    row, _ = _module().assess(path, 1, 100, 300)
    assert row["runaway"] is True
    assert row["n_thomson"] == 3 and row["n_profile"] == 1
    # Only 200 ms is a profile, but all three samples still set the median.
    assert row["te_p90_ev"] == 40.0


@pytest.mark.parametrize("missing", ["group", "window", "invalid", "sentinel"])
def test_no_thomson_is_unmarked_and_blank(tmp_path, missing):
    path = tmp_path / "1_processed.h5"
    _h5(path, [0, 100, 200, 300, 400], [[np.nan, 0, -1, np.inf, 0]])
    if missing in ("group", "sentinel"):
        with h5py.File(path, "a") as f:
            del f["ts_core_temp"]
            if missing == "sentinel":
                g = f.create_group("ts_core_temp")
                g["xdata"], g["ydata"] = [0.0], [[0.0]]
    bounds = (500, 600) if missing == "window" else (100, 300)
    row, _ = _module().assess(path, 1, *bounds)
    assert row["te_p90_ev"] is None and row["n_thomson"] == 0
    assert row["n_profile"] == 0
    assert row["runaway"] is False


@pytest.mark.parametrize("sparse", [False, True], ids=["missing", "sparse"])
def test_command_scans_only_rule4_shots_and_records_inputs(
    tmp_path, monkeypatch, sparse
):
    cat, corpus, out = tmp_path / "catalog", tmp_path / "corpus", tmp_path / "out"
    cat.mkdir()
    corpus.mkdir()
    pool = pd.DataFrame(
        [
            {"shot": s, "year": 2024, "reasons": "heating" if s == 4 else ""}
            for s in (1, 2, 3, 4)
        ],
        columns=pop.POOL_COLUMNS,
    )
    pool.to_csv(cat / "pool.csv", index=False)
    lines = [
        {
            "shot": s,
            "status": "ok",
            "window_start_ms": 100,
            "window_end_ms": 2100,
            "flattop_s": 0.5 if s == 3 else 1.5,
            "ip_peak_ma": 1.0,
            "dt_ms": 0.05,
            "version": 3,
            "ip_sha256": "a" * 64,
            "run": "b" * 32,
        }
        for s in (1, 2, 3)
    ]
    (cat / "ip.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
    (cat / "ip_runs.jsonl").write_text(json.dumps({"run": "b" * 32}) + "\n")
    for shot, te in ((1, 10), (2, np.nan)):
        values = [[te] * 5]
        if shot == 2 and sparse:
            values = [[10] * 5, [np.nan] * 5, [0] * 5, [np.inf] * 5]
        _h5(corpus / f"{shot}_processed.h5", [0, 100, 200, 300, 400], values)
    paths = Paths(root=tmp_path, corpus=corpus, label_tables=out)
    monkeypatch.setattr(Paths, "from_env", classmethod(lambda cls: paths))
    assert _module().main([]) == 0
    csv = out / "catalog" / "runaway.csv"
    result = pd.read_csv(csv)
    assert result.shot.tolist() == [1, 2]
    assert result.runaway.tolist() == [True, False]
    assert result.columns.tolist() == [
        "shot",
        "te_p90_ev",
        "n_thomson",
        "n_profile",
        "pinj_kw",
        "neutron_rate_mean",
        "runaway",
    ]
    assert result.n_profile.tolist() == [4, 0]
    assert result.n_thomson.tolist() == [4, 4 if sparse else 0]
    assert pd.isna(result.loc[1, "te_p90_ev"])
    meta = json.loads(csv.with_suffix(".meta.json").read_text())
    assert meta["counts"] == {"shots": 2, "marked": 1, "no_thomson": 1}
    assert meta["threshold_ev"] == 60.0
    assert "no usable Thomson profile in the window" in meta["statistic"]
    assert meta["inputs"]["ip_log"]["version"] == window.LOG_VERSION
    for key, name in (("pool", "pool.csv"), ("ip_log", "ip.jsonl")):
        assert (
            meta["inputs"][key]["sha256"]
            == hashlib.sha256((cat / name).read_bytes()).hexdigest()
        )
    assert (
        meta["outputs"]["runaway.csv"] == hashlib.sha256(csv.read_bytes()).hexdigest()
    )
    assert len(meta["git_sha"]) == 40 and isinstance(meta["git_dirty"], bool)
    assert "labeler.events.catalog.runaway" in meta["command"]
    assert meta["inputs"]["corpus_files"]["count"] == 2


def test_corpus_digest_binds_measured_values(tmp_path):
    path = tmp_path / "1_processed.h5"
    _h5(path, [0, 100, 200, 300, 400], [[10] * 5])
    before, digest = _module().assess(path, 1, 100, 300)
    with h5py.File(path, "a") as f:
        f["ts_core_temp/ydata"][0, 1:4] = 100
    after, changed = _module().assess(path, 1, 100, 300)
    assert before["runaway"] and not after["runaway"] and changed != digest
