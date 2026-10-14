"""``scripts/labeler/confinement_ours_shots.py``: the grid of a shot from disk inputs."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from labeler.confinement import zerod

SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts/labeler/confinement_ours_shots.py"
)
spec = importlib.util.spec_from_file_location("confinement_ours_shots", SCRIPT)
shots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shots)


def _write(path: Path, **groups) -> None:
    """A corpus-style file: each group is ``(time in s, rows)``."""
    with h5py.File(path, "w") as f:
        for name, (t, y) in groups.items():
            f.create_group(name)
            f[name]["xdata"] = np.asarray(t, dtype=np.float64)
            f[name]["ydata"] = np.atleast_2d(np.asarray(y, dtype=np.float32))


def _slow(value: float):
    """An EFIT-like trace: one point every 50 ms over two seconds, within 1 % of
    ``value`` (a constant trace counts as no record)."""
    t = np.arange(0.0, 2.001, 0.05)
    return t, value * (1 + 0.01 * np.sin(t * 5.0))


def _raw_cache(tmp_path: Path, *, extra: dict | None = None) -> Path:
    """Two seconds of D-alpha (FS01), a 3 MW beam from 0.5 s and a CO2 trace whose
    rows differ, so the V2 row is the only one that gives the density below."""
    raw = tmp_path / "raw"
    raw.mkdir()
    t = np.arange(0, 2.0, 1e-3)
    da = 1.0 + np.sin(t * 50.0) ** 2
    beam = np.where(t >= 0.5, 3e6 / 8, 0.0)
    co2 = np.stack([np.full_like(t, 9e13 + k * 1e12) for k in range(4)])
    co2[2] = 6e13 * (1 + 1e-3 * np.sin(t * 7.0))  # V2
    _write(
        raw / "123_processed.h5",
        filterscopes=(t, np.tile(da, (8, 1))),
        pinj=(t, np.tile(beam, (8, 1))),
        co2=(t, co2),
        ip=(t, -1.2e6 * (1 + 0.01 * np.sin(t * 3.0))),  # amperes, negative sign
        **(extra or {}),
    )
    return raw


def test_disk_record_reads_the_v2_chord_and_prefers_the_feature_store(tmp_path):
    raw = _raw_cache(
        tmp_path,
        extra={"betan": _slow(1.0), "wmhd": _slow(3e5)},
    )
    features = tmp_path / "features"
    features.mkdir()
    _write(features / "123_features.h5", betan=_slow(2.0))
    arrays, source = shots.disk_record(123, raw, tmp_path / "corpus", features)
    # row 2, not R0, V1 or V3 (9e13, 9.1e13, 9.3e13)
    assert np.allclose(arrays["dens"][:2000], 6e13, rtol=2e-3)
    assert float(arrays["dens_t0_ms"]) == 0.0
    # the feature store (2.0) wins over the raw cache (1.0)
    assert np.allclose(arrays["betan"], 2.0, rtol=0.02)
    assert np.allclose(arrays["betan_t_ms"][[0, -1]], [0.0, 2000.0])
    assert np.allclose(arrays["wmhd"], 3e5, rtol=0.02)
    assert source == {
        "dens": "raw:co2 V2 (DENUF), 1 ms mean",
        "betan": "features:betan",
        "wmhd": "raw:wmhd",
    }


def test_disk_record_leaves_out_what_the_disk_lacks(tmp_path):
    raw = _raw_cache(tmp_path)
    arrays, source = shots.disk_record(
        123, raw, tmp_path / "corpus", tmp_path / "features"
    )
    assert set(source) == {"dens"}
    assert not {"betan", "wmhd"} & set(arrays)
    flat = tmp_path / "flat"
    flat.mkdir()
    _write(flat / "9_processed.h5", co2=([0.0, 1.0], [[5.0, 5.0]] * 4))
    assert shots.disk_record(9, flat, tmp_path, tmp_path) == ({}, {})  # a flat trace


def test_assemble_disk_masks_the_channels_it_could_not_fill(tmp_path):
    raw = _raw_cache(tmp_path, extra={"betan": _slow(1.5)})
    result = shots.assemble_disk(
        123,
        tmp_path / "zerod",
        raw_dir=raw,
        corpus=tmp_path / "corpus",
        features=tmp_path / "features",
    )
    assert result["inputs"] == {
        "dens": "raw:co2 V2 (DENUF), 1 ms mean",
        "betan": "raw:betan",
        "wmhd": "masked",
        "pinj": "corpus/raw",
    }
    grid = result["grid"]
    index = zerod.CHANNELS.index
    assert np.isnan(grid[index("wmhd")]).all()
    assert np.isclose(np.nanmedian(grid[index("dens")]), 0.6, rtol=2e-3)  # / 1e14
    assert np.isclose(np.nanmedian(grid[index("betan")]), 1.0, rtol=0.02)  # 1.5 / 1.5
    assert np.isclose(np.nanmax(grid[index("pinj")]), 1.0)  # 3 MW / 3e6
    # a stale record from an earlier run does not leak into a shot that has none
    stale = tmp_path / "zerod" / "124.npz"
    np.savez(stale, dens=np.ones(10), dens_t0_ms=np.array(0.0))
    none = shots.assemble_disk(
        124, tmp_path / "zerod", raw_dir=raw, corpus=tmp_path, features=tmp_path
    )
    assert none is None and not stale.exists()


def test_assemble_disk_uses_a_fetched_record_as_it_stands(tmp_path):
    raw = _raw_cache(tmp_path)  # its co2 trace would give a density of 0.6
    fetched = tmp_path / "fetched"
    fetched.mkdir()
    t, one = _slow(1.0)
    np.savez(
        fetched / "123.npz",
        dens=np.full(2000, 2e14, dtype=np.float32),
        dens_t0_ms=np.array(0.0),
        betan_t_ms=t * 1000.0,
        betan=one,
        wmhd_t_ms=t * 1000.0,
        wmhd=one * 5e5,
    )
    result = shots.assemble_disk(
        123, tmp_path / "zerod", raw_dir=raw, corpus=tmp_path, fetched_dir=fetched
    )
    index = zerod.CHANNELS.index
    assert np.isclose(np.nanmedian(result["grid"][index("dens")]), 2.0)
    assert np.isclose(np.nanmedian(result["grid"][index("wmhd")]), 1.0, rtol=0.02)
    assert result["inputs"] == {
        "dens": "fetched",
        "betan": "fetched",
        "wmhd": "fetched",
        "pinj": "corpus/raw/fetched",
    }
    assert shots.missing_reason(result) is None
    assert not (tmp_path / "zerod" / "123.npz").exists()  # the disk path never ran


def test_missing_reason_names_the_channels_the_network_cannot_do_without():
    ok = {"dens": "raw:co2", "betan": "features:betan", "wmhd": "raw:wmhd"}
    assert shots.missing_reason({"inputs": ok}) is None
    got = shots.missing_reason({"inputs": {**ok, "wmhd": "masked", "dens": "masked"}})
    assert got.startswith("no dens, wmhd record on disk")


def test_estimate_wmhd_scales_beta_n_by_the_plasma_current():
    t = np.array([0.0, 1000.0, 2000.0])
    w = shots.estimate_wmhd(
        t, np.array([1.0, 2.0, np.nan]), np.array([0.0, 2000.0]), np.array([1.0, 2.0])
    )
    k = shots.WMHD_PER_BETAN_MA
    assert np.allclose(w[:2], [k * 1.0 * 1.0, k * 2.0 * 1.5])  # I_p: 1, 1.5, 2 MA
    assert np.isnan(w[2])


def test_read_ip_gives_the_absolute_current_in_megaamperes(tmp_path):
    raw = _raw_cache(tmp_path)
    t_ms, ip = shots.read_ip(123, raw, tmp_path / "corpus", tmp_path / "features")
    assert t_ms[0] == 0.0 and np.allclose(ip, 1.2, rtol=0.02)
    assert shots.read_ip(7, raw, tmp_path, tmp_path) is None


def test_disk_record_estimates_w_mhd_only_when_asked_and_only_if_it_has_none(tmp_path):
    raw = _raw_cache(tmp_path, extra={"betan": _slow(2.0)})
    args = (123, raw, tmp_path / "corpus", tmp_path / "features")
    arrays, source = shots.disk_record(*args)
    assert "wmhd" not in arrays and "wmhd" not in source
    arrays, source = shots.disk_record(*args, estimate=True)
    assert np.allclose(arrays["wmhd"], 357e3 * 2.0 * 1.2, rtol=0.02)
    assert np.allclose(arrays["wmhd_t_ms"], arrays["betan_t_ms"])
    assert source["wmhd"] == "estimated: 357 kJ x beta_N x I_p[MA]"
    # a recorded W_MHD is never replaced, and no beta_N means nothing to estimate from
    (tmp_path / "two").mkdir()
    raw2 = _raw_cache(tmp_path / "two", extra={"betan": _slow(2.0), "wmhd": _slow(4e5)})
    arrays, source = shots.disk_record(
        123, raw2, tmp_path / "corpus", tmp_path / "f", estimate=True
    )
    assert source["wmhd"] == "raw:wmhd" and np.allclose(arrays["wmhd"], 4e5, rtol=0.02)
    (tmp_path / "three").mkdir()
    raw3 = _raw_cache(tmp_path / "three")
    arrays, _ = shots.disk_record(123, raw3, tmp_path, tmp_path, estimate=True)
    assert "wmhd" not in arrays


def test_segment_grid_names_why_a_shot_has_no_segment():
    n = 300
    grid = np.full((len(zerod.CHANNELS), n), np.nan, dtype=np.float32)
    prob = np.zeros((4, n), dtype=np.float32)
    prob[1] = 1.0
    assert shots.segment_grid(grid, prob) == ([], "no beam power record")
    grid[zerod.CHANNELS.index("pinj")] = 0.0
    assert shots.segment_grid(grid, prob) == (
        [],
        "beam power never reaches the threshold",
    )
    grid[zerod.CHANNELS.index("pinj")] = 1.0
    found, why = shots.segment_grid(grid, prob)
    assert why is None and [(c, lo, hi) for c, lo, hi, _ in found] == [(1, 0, n)]


def test_summarise_counts_the_floor_in_segments_and_in_seconds():
    frame = pd.DataFrame(
        {
            "category": [1, 5, 2],
            "predicted": [1, 3, 2],
            "t_start": [0.0, 1000.0, 2000.0],
            "t_end": [1000.0, 1500.0, 4000.0],
        }
    )
    got = shots.summarise(frame)
    assert got["segments"] == 3 and got["label_seconds"] == 3.5
    assert got["segments_under_floor"] == 1
    assert got["fraction_segments_under_floor"] == 0.333
    assert got["fraction_seconds_under_floor"] == round(0.5 / 3.5, 3)
    assert got["by_category"]["uncertain"] == {"segments": 1, "seconds": 0.5}
    assert got["by_predicted_class"]["QH"] == {"segments": 1, "seconds": 0.5}


def test_extend_table_passes_the_validator_and_keeps_the_roster_columns_in_attrs():
    from labeler.events.interval_tables import parse_attrs, validate_intervals

    frame = pd.DataFrame(
        {
            "shot": [1, 1],
            "category": [1, 5],
            "t_start": [0.0, 500.0],
            "t_end": [500.0, 800.0],
            "confidence": [0.9, 0.6],
            "predicted": [1, 3],
            "source": ["ensemble", "held_out"],
            "tier": ["model", "unreviewed"],
            "extrapolated": [False, True],
        }
    )
    table = shots.to_extend_table(frame)
    assert list(table.columns) == [*frame.columns[:5], "attrs"]
    assert len(validate_intervals(table)) == 2  # the 9-column roster layout fails
    assert parse_attrs(table["attrs"].iloc[1]) == {
        "predicted": 3,
        "source": "held_out",
        "tier": "unreviewed",
        "extrapolated": True,
    }


def test_input_summary_says_whether_any_channel_is_not_a_fetched_record():
    fetched = {"dens": "fetched", "betan": "fetched", "wmhd": "fetched"}
    got = shots.input_summary({1: fetched, 2: fetched})
    assert got["shots_not_fetched"] == []
    assert got["sources"]["dens"] == {"fetched": 2}
    assert "no surrogate input" in got["statement"]
    mixed = shots.input_summary({1: fetched, 2: {**fetched, "wmhd": "masked"}})
    assert mixed["shots_not_fetched"] == [2]
    assert mixed["sources"]["wmhd"] == {"fetched": 1, "masked": 1}
    assert "1 of 2 shots" in mixed["statement"]


def test_select_shots_drops_the_labelled_ones_but_keeps_what_is_included():
    listed = [1, 2, 3, 4]
    assert shots.select_shots(listed, None) == [1, 2, 3, 4]
    assert shots.select_shots(listed, {2, 3}) == [1, 4]
    assert shots.select_shots(listed, {2, 3}, [3, 9]) == [1, 3, 4]  # 9 is not listed
