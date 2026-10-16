"""``scripts/labeler/confinement_ours_shots.py``: the grid of a shot from disk inputs."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

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


def test_predict_averages_a_folds_seeds_then_the_folds(monkeypatch):
    # a "network" here is the probabilities it gives; predict_mean averages them
    monkeypatch.setattr(
        shots.ours, "predict_mean", lambda nets, x, device: np.mean(nets, axis=0)
    )
    x = np.zeros((zerod.N_INPUT, 3), dtype=np.float32)
    three = [[np.full((4, 3), 10.0 * f + s) for s in range(3)] for f in range(5)]
    assert np.allclose(shots.predict(x, three, "cpu", fold=3), 31.0)  # its fold only
    assert np.allclose(shots.predict(x, three, "cpu"), 21.0)  # folds 1, 11, ..., 41
    one = [[np.full((4, 3), float(f))] for f in range(5)]  # run_day: a seed a fold
    assert np.allclose(shots.predict(x, one, "cpu", fold=4), 4.0)
    assert np.allclose(shots.predict(x, one, "cpu"), 2.0)


def test_checkpoint_record_names_every_seed_of_every_fold(tmp_path, monkeypatch):
    three = shots.ours.Variant(tmp_path, seeds=3)
    monkeypatch.setitem(shots.ours.VARIANTS, "three", three)
    for fold in range(shots.ours.FOLDS):
        for seed in range(3):
            stem = shots.ours.seed_stem(fold, seed)
            (tmp_path / f"{stem}_model.pt").write_bytes(b"%d%d" % (fold, seed))
    got = shots.checkpoint_record("three")
    assert len(got) == 15 and list(got)[:3] == ["fold0", "fold0_s1", "fold0_s2"]
    assert got["fold4_s2"] == {
        "path": str(tmp_path / "fold4_s2_model.pt"),
        "sha256": hashlib.sha256(b"42").hexdigest(),
    }


def test_only_seven_channel_run_day_variants_are_offered_and_base3_sorts_first():
    assert {"run_day", "base3"} <= set(shots.INPUT_VARIANTS)
    assert not {"beams", "base_no_da", "shot"} & set(shots.INPUT_VARIANTS)
    assert shots.default_out_dir("run_day") == shots.ours.EXTEND
    base3 = shots.default_out_dir("base3")
    assert base3 == shots.ours.EXTEND.parent / "extend_confine_base3"
    # the review page ranks extend tables of one kind by sorted path
    assert min(shots.ours.EXTEND.name, base3.name) == base3.name


def test_label_takes_a_named_shot_list_and_a_variant(monkeypatch):
    seen = {}
    monkeypatch.setattr(shots, "label", lambda args: seen.update(vars(args)))
    shots.main(
        ["label", "--shots-file", "s.csv", "--name", "silver", "--variant", "base3"]
    )
    assert seen["shots"] == Path("s.csv") and seen["variant"] == "base3"
    assert seen["out_dir"] is None  # label() resolves it by the variant
    shots.main(["label", "--shots", "gold.csv", "--name", "gold_shots"])
    assert seen["shots"] == Path("gold.csv") and seen["variant"] == "run_day"
    # the gold run's one-shot-at-a-time path stays the default
    assert (seen["workers"], seen["batch"], seen["check"]) == (0, 0, 0)
    shots.main(
        ["label", "--shots", "s.csv", "--name", "n"]
        + ["--workers", "8", "--batch", "64", "--check", "20"]
    )
    assert (seen["workers"], seen["batch"], seen["check"]) == (8, 64, 20)


def _grids(rng, lengths):
    """Shots' 1 ms grids of different lengths: D-alpha rows inside and outside the
    clipping, sparse channels with gaps, an infinity and values past 8."""
    out = []
    for n in lengths:
        grid = rng.normal(0.0, 2.0, (len(zerod.CHANNELS), n)).astype(np.float32)
        grid[3:, : n // 5] = np.nan  # no density, beta_N, W_MHD or beams early on
        grid[zerod.CHANNELS.index("wmhd"), n // 2 :] = np.nan
        grid[0, 3], grid[4, -2], grid[6, n // 3] = np.inf, -np.inf, 12.0
        out.append(grid)
    return out


def _network(seed: int):
    """A small U-Net in eval mode whose batch norms shift and scale, so a padded bin
    past a shot's end comes out of a block far from zero."""
    import torch

    from labeler.confinement import unet

    torch.manual_seed(seed)
    net = unet.UNet1d()
    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            m.running_mean.uniform_(-1.0, 1.0)
            m.running_var.uniform_(0.5, 2.0)
            m.weight.data.uniform_(0.5, 1.5)
            m.bias.data.uniform_(-0.5, 0.5)
    return net.eval()


def test_to_input_torch_is_to_input_on_a_batch():
    import torch

    grids = _grids(np.random.default_rng(0), [300, 300])
    got = shots.to_input_torch(torch.from_numpy(np.stack(grids))).numpy()
    for g, x in zip(grids, got, strict=True):
        assert np.array_equal(x, zerod.to_input(g))


def test_forward_alone_gives_each_shot_what_it_gets_alone():
    import torch

    from labeler.confinement import unet

    net = _network(1)
    n = [130, 515, 384]  # padded to 256, 640 and 384 bins
    lengths = [k + (-k) % unet.STRIDE for k in n]
    batch = torch.full((3, len(zerod.CHANNELS), max(lengths)), float("nan"))
    for i, g in enumerate(_grids(np.random.default_rng(2), n)):
        batch[i, :, : n[i]] = torch.from_numpy(g)
    x = shots.to_input_torch(batch)
    with torch.no_grad():
        logits = shots.forward_alone(net, x, torch.tensor(lengths))
        naive = net(x)
    moved = 0.0
    for i, k in enumerate(n):
        alone = unet.predict_shot(net, x[i, :, :k].numpy(), "cpu")
        got = torch.softmax(logits, 1)[i, :, :k].numpy()
        assert np.abs(got - alone).max() < 1e-5
        moved = max(
            moved, np.abs(torch.softmax(naive, 1)[i, :, :k].numpy() - alone).max()
        )
    # padding the shorter shots without the masks changes what they get
    assert moved > 1e-3


def test_predict_batch_is_predict_shot_by_shot():
    import torch

    models = [[_network(10 * f + s) for s in range(2)] for f in range(shots.ours.FOLDS)]
    n = [200, 450, 129, 384]
    grids = _grids(np.random.default_rng(3), n)
    folds = [None, 3, None, 0]  # two shots a fold held out, two for the ensemble
    got = shots.predict_batch(grids, folds, models, torch.device("cpu"))
    for g, f, p in zip(grids, folds, got, strict=True):
        want = shots.predict(zerod.to_input(g), models, "cpu", f)
        assert p.shape == want.shape == (4, g.shape[1])
        assert np.abs(p - want).max() < 1e-5


def test_read_shots_keeps_the_order_and_reads_here_without_workers(monkeypatch):
    seen = []

    def fake(shot, zerod_dir, *, fetched_dir=None):
        seen.append((shot, zerod_dir, fetched_dir))
        return {"shot": shot}

    monkeypatch.setattr(shots, "assemble_disk", fake)
    got = list(shots.read_shots([5, 2, 9], 0, Path("fetched")))
    assert got == [(5, {"shot": 5}), (2, {"shot": 2}), (9, {"shot": 9})]
    assert seen[0] == (5, shots.ZEROD_DISK, Path("fetched"))


class _FakeProcess:
    killed = False

    def kill(self):
        self.killed = True


def _fake_pool(monkeypatch, dies_on=None):
    """A stand-in process pool whose futures are done at once; reading ``dies_on``
    raises what a pool does after the kernel kills one of its workers."""
    import concurrent.futures as cf
    from concurrent.futures.process import BrokenProcessPool

    made: list = []

    class Pool:
        def __init__(self, workers, mp_context=None):
            self._processes = {1: _FakeProcess(), 2: _FakeProcess()}
            self.submitted, self.stopped = [], None
            made.append(self)

        def submit(self, fn, shot):
            self.submitted.append(shot)
            future = cf.Future()
            if shot == dies_on:
                future.set_exception(BrokenProcessPool("a worker was killed"))
            else:
                future.set_result(fn(shot))
            return future

        def shutdown(self, wait=True, *, cancel_futures=False):
            self.stopped = (wait, cancel_futures)
            if self._processes is not None:  # as CPython 3.11, a second call finds None
                self.workers, self._processes = self._processes, None

    monkeypatch.setattr(cf, "ProcessPoolExecutor", Pool)
    return made


def test_read_shots_runs_a_bounded_number_ahead_of_the_caller(monkeypatch):
    pools = _fake_pool(monkeypatch)
    reads = shots.read_shots(list(range(20)), 2, None, read=lambda s: s, ahead=3)
    assert next(reads) == (0, 0)
    assert pools[0].submitted == [0, 1, 2, 3]  # 3 in flight, one more per result
    assert [s for s, _ in reads] == list(range(1, 20))
    assert pools[0].stopped == (True, False)


def test_read_shots_stops_at_once_when_a_worker_dies(monkeypatch):
    pools = _fake_pool(monkeypatch, dies_on=3)
    seen = []
    with pytest.raises(SystemExit, match="stopping at once"):
        for shot, _ in shots.read_shots(list(range(9)), 2, None, read=lambda s: s):
            seen.append(shot)
    assert seen == [0, 1, 2]
    made = pools[0]
    assert made.stopped == (False, True)  # no wait, futures cancelled
    assert made._processes is None and len(made.workers) == 2
    assert all(p.killed for p in made.workers.values())
