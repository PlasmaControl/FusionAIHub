"""Tests of the unit transform and the per-channel statistics.

Nothing here needs `tokamark` or data: the shot loader is monkeypatched where statistics are fitted.
"""

import json

import numpy as np
import pytest

from tokamak_foundation_model.ignite.mast import tasks, transform


def _spec(name, shape=(1,), dt=0.001):
    return tasks.SignalSpec(name, dt, shape, 0.0, 1.0)


def test_roundtrip_none_and_asinh():
    rng = np.random.default_rng(0)
    for compress in ("none", "asinh"):
        st = transform.ChannelStats(compress, scale=np.full((3,), 0.5), mean=np.array([0.1, -2.0, 3.0]),
                                    std=np.array([1.0, 0.2, 5.0]))
        x = rng.standard_normal((3, 100)) * 10
        np.testing.assert_allclose(transform.to_physical(transform.to_model(x, st), st), x,
                                   rtol=1e-9, atol=1e-9)


def test_roundtrip_two_dimensional_signal():
    """A map: stats of values_shape (2, 3), data (2, 3, T)."""
    rng = np.random.default_rng(1)
    st = transform.ChannelStats("asinh", scale=rng.uniform(0.5, 2, (2, 3)), mean=rng.standard_normal((2, 3)),
                                std=rng.uniform(0.5, 2, (2, 3)))
    x = rng.standard_normal((2, 3, 7)) * 100
    z = transform.to_model(x, st)
    assert z.shape == x.shape
    # channel (1, 2) is transformed with its own constants
    ref = (np.arcsinh(x[1, 2] / st.scale[1, 2]) - st.mean[1, 2]) / st.std[1, 2]
    np.testing.assert_allclose(z[1, 2], ref)
    np.testing.assert_allclose(transform.to_physical(z, st), x, rtol=1e-9, atol=1e-9)


def test_nan_stays_nan_and_sign_is_kept():
    st = transform.ChannelStats("none", np.ones((1,)), np.zeros((1,)), np.ones((1,)))
    z = transform.to_model(np.array([[np.nan, -3.0, 3.0]]), st)
    assert np.isnan(z[0, 0]) and z[0, 1] == -z[0, 2]
    st = transform.ChannelStats("asinh", np.ones((1,)), np.zeros((1,)), np.ones((1,)))
    z = transform.to_model(np.array([[np.nan, -3.0, 3.0]]), st)
    assert np.isnan(z[0, 0]) and z[0, 1] == -z[0, 2] and z[0, 2] > 0


@pytest.mark.parametrize("fn", ["to_model", "to_physical"])
@pytest.mark.parametrize("compress", ["asinh_centred", "ASINH", "", "log"])
def test_an_unknown_compress_is_refused_not_taken_for_the_identity(fn, compress):
    """A third value is expected in stage 1; code that does not know it must not return unstandardized values."""
    st = transform.ChannelStats(compress, np.ones((1,)), np.zeros((1,)), np.ones((1,)))
    with pytest.raises(ValueError, match="compress"):
        getattr(transform, fn)(np.ones((1, 4)), st)


def test_load_stats_refuses_an_unknown_compress(tmp_path):
    """The file is read once and used for the whole cache: a value this code does not know stops there."""
    path = tmp_path / "stats.npz"
    transform.save_stats({"a-x": transform.ChannelStats("asinh_centred", np.ones((1,)), np.zeros((1,)),
                                                        np.ones((1,)))}, path)
    with pytest.raises(ValueError, match="asinh_centred"):
        transform.load_stats(path)


def test_the_two_known_compress_values_are_read_back(tmp_path):
    path = tmp_path / "stats.npz"
    transform.save_stats({n: transform.ChannelStats(n, np.ones((1,)), np.zeros((1,)), np.ones((1,)))
                          for n in ("none", "asinh")}, path)
    assert {k: v.compress for k, v in transform.load_stats(path).items()} == {"none": "none", "asinh": "asinh"}


def _fake_loader(n_t=600, spiky_fraction=0.01):
    """A loader of two scalar signals, deterministic per shot: one gaussian, one with 1% of samples x1000."""
    def load(shot, signals):
        out = {}
        for s in signals:
            rng = np.random.default_rng([1000 + shot, len(s.name)])   # per signal: the loader ignores their order
            x = rng.standard_normal(n_t) * 2.0 + 5.0
            if s.name == "spiky":
                x = rng.standard_normal(n_t)
                x[rng.random(n_t) < spiky_fraction] *= 1000.0
            out[s.name] = {"time": np.arange(n_t) * s.dt, "values": x[None, :]}
        return out
    return load


def test_fit_stats_picks_asinh_for_heavy_tails(monkeypatch):
    monkeypatch.setattr(transform, "load_shot", _fake_loader())
    signals = [_spec("gauss"), _spec("spiky")]
    stats = transform.fit_stats(list(range(60)), signals, samples_per_channel=64, seed=0, workers=1)
    assert stats["gauss"].compress == "none" and stats["spiky"].compress == "asinh"
    assert abs(stats["gauss"].std[0] - 2.0) / 2.0 < 0.05 and abs(stats["gauss"].mean[0] - 5.0) < 0.1
    assert stats["gauss"].scale[0] == 1.0
    # asinh scale is the scaled MAD of the sample (the core is N(0, 1) with a few outliers)
    assert abs(stats["spiky"].scale[0] - 1.0) < 0.1
    # mean / std are those of the compressed values
    assert stats["spiky"].std[0] > 0 and np.isfinite(stats["spiky"].mean[0])


def test_fit_stats_is_reproducible_whatever_the_order(monkeypatch):
    monkeypatch.setattr(transform, "load_shot", _fake_loader())
    signals = [_spec("gauss"), _spec("spiky")]
    shots = list(range(40))
    a = transform.fit_stats(shots, signals, samples_per_channel=64, seed=3, workers=1)
    b = transform.fit_stats(shots[::-1], signals[::-1], samples_per_channel=64, seed=3, workers=1)
    c = transform.fit_stats(shots, signals, samples_per_channel=64, seed=4, workers=1)
    for name in ("gauss", "spiky"):
        assert a[name].compress == b[name].compress
        for f in ("scale", "mean", "std"):
            np.testing.assert_array_equal(getattr(a[name], f), getattr(b[name], f))
    assert not np.array_equal(a["gauss"].mean, c["gauss"].mean)   # another seed, another sample


def test_degenerate_channels_get_mean_0_std_1(monkeypatch, capsys):
    """Constant channel, channel with too few finite samples, and a channel with MAD 0 but a spread."""
    def load(shot, signals):
        rng = np.random.default_rng(shot)
        n = 600
        x = rng.standard_normal((4, n))
        x[0] = 7.0                                  # constant: zero spread
        x[1, 10:] = np.nan                          # 10 finite samples per shot: < 1000 in all
        x[2, : int(n * 0.9)] = 0.0                  # MAD is 0 (90% zeros) but the spread is not
        x[2, int(n * 0.9):] = rng.standard_normal(n - int(n * 0.9)) * 3.0
        return {"c": {"time": np.arange(n) * 0.001, "values": x}}
    monkeypatch.setattr(transform, "load_shot", load)
    sig = _spec("c", (4,))
    st = transform.fit_stats(list(range(30)), [sig], workers=1)["c"]
    assert (st.mean[0], st.std[0]) == (0.0, 1.0)
    assert (st.mean[1], st.std[1]) == (0.0, 1.0)
    assert st.std[2] > 0 and st.std[3] > 0.5
    # the degenerate channels are listed in the printed summary, not dropped silently
    assert "c: degenerate channels [0, 1]" in capsys.readouterr().out


def test_zero_mad_channel_uses_the_sample_std_as_the_asinh_scale(monkeypatch):
    """A heavy-tailed channel whose MAD is 0 takes the sample standard deviation as its asinh scale."""
    def load(shot, signals):
        rng = np.random.default_rng(shot)
        n = 600
        x = np.zeros(n)
        k = rng.choice(n, 12, replace=False)
        x[k] = rng.standard_normal(12) * 100.0       # 98% zeros, 2% large: kurtosis far above 10
        return {"c": {"time": np.arange(n) * 0.001, "values": x[None, :]}}
    monkeypatch.setattr(transform, "load_shot", load)
    st = transform.fit_stats(list(range(40)), [_spec("c")], workers=1)["c"]
    assert st.compress == "asinh"
    assert st.scale[0] > 1.0 and np.isfinite(st.scale[0])


def test_missing_and_failing_shots_are_counted_not_dropped(monkeypatch, capsys):
    good = _fake_loader()

    def load(shot, signals):
        if shot % 10 == 7:
            raise OSError("no such shot")
        out = good(shot, signals)
        if shot % 10 == 3:                      # TokaMark returns an empty signal for a missing one
            out["gauss"] = {"time": np.zeros((0,)), "values": np.zeros((0,))}
        return out
    monkeypatch.setattr(transform, "load_shot", load)
    stats = transform.fit_stats(list(range(60)), [_spec("gauss"), _spec("spiky")], workers=1)
    assert set(stats) == {"gauss", "spiky"}
    out = capsys.readouterr().out
    assert "6 shots failed to load" in out
    assert "gauss: 6 shots without the signal" in out


def test_no_shot_loading_is_an_error_not_identity_constants(monkeypatch):
    """Zero loaded shots must not return mean 0 / std 1 for everything (a silent unit change)."""
    def load(shot, signals):
        raise OSError(f"cannot read shot {shot}")
    monkeypatch.setattr(transform, "load_shot", load)
    with pytest.raises(RuntimeError, match=r"no shot could be loaded.*5 shots.*OSError: cannot read shot 0"):
        transform.fit_stats(list(range(5)), [_spec("a")], workers=1)
    with pytest.raises(RuntimeError, match="no shot could be loaded"):
        transform.fit_stats_with_summary([], [_spec("a")], workers=1)


def test_an_import_error_propagates_and_is_not_counted(monkeypatch):
    def load(shot, signals):
        raise ModuleNotFoundError("No module named 'tokamark'")
    monkeypatch.setattr(transform, "load_shot", load)
    with pytest.raises(ModuleNotFoundError, match="tokamark"):
        transform.fit_stats(list(range(5)), [_spec("a")], workers=1)


def _summary_loader():
    good = _fake_loader()

    def load(shot, signals):
        if shot == 5:
            raise RuntimeError("x" * 300)
        out = good(shot, signals)
        if shot == 8:
            del out["spiky"]
        return out
    return load


def test_summary_counts(monkeypatch):
    monkeypatch.setattr(transform, "load_shot", _summary_loader())
    signals = [_spec("gauss"), _spec("spiky")]
    stats, summary = transform.fit_stats_with_summary(list(range(40)), signals, workers=1)
    assert set(summary) == {"n_shots", "n_loaded", "failed", "missing", "degenerate"}
    assert summary["n_shots"] == 40 and summary["n_loaded"] == 39
    assert list(summary["failed"]) == [5]
    assert summary["failed"][5].startswith("RuntimeError: x") and len(summary["failed"][5]) == len("RuntimeError: ") + 200
    assert summary["missing"] == {"gauss": 0, "spiky": 1}
    assert summary["degenerate"] == {"gauss": [], "spiky": []}
    json.dumps(summary)                                   # serialisable as it is
    assert stats["gauss"].compress == "none" and stats["spiky"].compress == "asinh"
    assert abs(stats["gauss"].std[0] - 2.0) / 2.0 < 0.05   # fitted from the good shots
    again = transform.fit_stats(list(range(40)), signals, workers=1)    # fit_stats returns the same statistics
    for name in stats:
        np.testing.assert_array_equal(stats[name].mean, again[name].mean)


def test_summary_does_not_depend_on_the_shot_order(monkeypatch):
    monkeypatch.setattr(transform, "load_shot", _summary_loader())
    signals = [_spec("gauss"), _spec("spiky")]
    _, a = transform.fit_stats_with_summary(list(range(40)), signals, workers=1)
    _, b = transform.fit_stats_with_summary(list(range(39, -1, -1)), signals[::-1], workers=1)
    assert a == b and json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_a_shot_outside_the_splits_is_an_error_not_a_miss(monkeypatch):
    def load(shot, signals):
        raise ValueError("shots [1] are not in any split of TokaMark's split file; they are never loaded")
    monkeypatch.setattr(transform, "load_shot", load)
    with pytest.raises(ValueError, match="never loaded"):
        transform.fit_stats([1], [_spec("a")], workers=1)


def test_save_load_stats_roundtrip_and_hash(tmp_path):
    rng = np.random.default_rng(0)
    stats = {
        "a-x": transform.ChannelStats("none", np.ones((3,)), rng.standard_normal(3), rng.uniform(1, 2, 3)),
        "b-y": transform.ChannelStats("asinh", rng.uniform(1, 2, (2, 2)), rng.standard_normal((2, 2)),
                                      rng.uniform(1, 2, (2, 2))),
    }
    p1, p2 = tmp_path / "s1.npz", tmp_path / "s2.npz"
    transform.save_stats(stats, p1)
    transform.save_stats(stats, p2)
    back = transform.load_stats(p1)
    assert set(back) == set(stats)
    for k in stats:
        assert back[k].compress == stats[k].compress
        for f in ("scale", "mean", "std"):
            np.testing.assert_array_equal(getattr(back[k], f), getattr(stats[k], f))
    assert transform.stats_sha256(p1) == transform.stats_sha256(p2)
    # numpy reads the file too
    with np.load(p1) as z:
        assert len(z.files) == 8
    stats["a-x"].mean[0] += 1.0
    transform.save_stats(stats, p2)
    assert transform.stats_sha256(p1) != transform.stats_sha256(p2)
    assert not list(tmp_path.glob("*.tmp"))


def test_excess_kurtosis_matches_the_definition():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 200000))
    assert np.all(np.abs(transform._excess_kurtosis(x)) < 0.05)
    lap = rng.laplace(size=(1, 400000))          # excess kurtosis 3
    assert abs(transform._excess_kurtosis(lap)[0] - 3.0) < 0.15
    assert np.isnan(transform._excess_kurtosis(np.full((1, 100), 2.0))[0])


@pytest.mark.real_data
def test_load_shot_and_a_process_pool_on_real_shots():
    """TokaMark's loader through `load_shot`, and a 2-worker pool giving the statistics of the calling process."""
    shots = tasks.split_shots("train")[:5]
    sigs = [tasks.SignalSpec("summary-ip", 0.00025, (1,), 0.0, 1.0),
            tasks.SignalSpec("pf_active-coil_current", 0.00025, (10,), 0.0, 1.0)]
    d = transform.load_shot(shots[0], sigs)
    assert d["summary-ip"]["values"].shape == (1, d["summary-ip"]["time"].shape[0])
    assert d["pf_active-coil_current"]["values"].shape[0] == 10
    one = transform.fit_stats(shots, sigs, samples_per_channel=400, seed=0, workers=1)
    two = transform.fit_stats(shots, sigs, samples_per_channel=400, seed=0, workers=2)
    for s in sigs:
        assert one[s.name].mean.shape == s.values_shape
        for f in ("scale", "mean", "std"):
            np.testing.assert_array_equal(getattr(one[s.name], f), getattr(two[s.name], f))
        assert one[s.name].compress == two[s.name].compress
    assert (one["summary-ip"].std[0], one["summary-ip"].mean[0]) != (1.0, 0.0)   # not degenerate
