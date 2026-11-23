"""Streams, their identity features and the store of examples (plan 1c)."""

import numpy as np
import pandas as pd
import pytest

from tokamak_foundation_model.ignite.mast import tasks, tokenizers, transform
from tokamak_foundation_model.ignite.mast.tokenizers import assign, codecs, learned
from tokamak_foundation_model.ignite.mast.tokenizers.learned import identity, store

RNG = np.random.default_rng(0)


def rows(signal, n, family="slow", dt=0.0002, units="T", compress="none", r=np.nan, z=np.nan):
    return pd.DataFrame({"signal": signal, "channel": np.arange(n), "channel_name": [f"c{i}" for i in range(n)],
                         "family": family, "dt": dt, "units": units, "compress": compress, "r": r, "z": z})


def series_front(n_ch=2, k=16):
    shape = (n_ch,)
    st = transform.ChannelStats(compress="none", scale=np.ones(shape), mean=np.zeros(shape), std=np.ones(shape))
    return codecs.FrameDCT("m", 0.00025, shape, st, k)


def test_the_code_hash_covers_this_sub_package_only():
    assert len(learned.code_sha256()) == 64 and learned.code_sha256() == learned.code_sha256()
    assert learned.code_sha256() != tokenizers.code_sha256()


def test_features_of_a_probe_array():
    f = identity.stream_features(rows("m", 5), "frame_dct", 5)
    assert f.shape == (5, identity.N_FEAT) and f.dtype == np.float32
    assert np.array_equal(f[:, :4], np.tile([1, 0, 0, 0], (5, 1))) and np.allclose(f[:, 4], np.log10(0.0002) + 3)
    assert np.array_equal(f[:, 5:12], np.tile([0, 1, 0, 0, 0, 0, 0], (5, 1)))        # not asinh; unit T
    assert np.allclose(f[:, 12], [0, 0.25, 0.5, 0.75, 1]) and not f[:, 13:].any()
    g = identity.stream_features(rows("m", 2, units="eV", r=[0.2, 0.3], z=[-1.0, 1.0]), "frame_dct", 2)
    assert np.array_equal(g[:, 11], [1, 1]) and np.array_equal(g[:, 13], [1, 1])
    assert np.allclose(g[:, 14], [0.2, 0.3]) and np.allclose(g[:, 15], [-1, 1])
    with pytest.raises(ValueError):
        identity.stream_features(rows("m", 5), "frame_dct", 4)


def test_features_of_a_spectrogram_signal_are_per_probe_and_band():
    f = identity.stream_features(rows("s", 768, family="high_rate", dt=0.001024, units="", compress="asinh"), "band_dct", 24)
    assert f.shape == (24, identity.N_FEAT) and f[:, 1].all() and f[:, 5].all() and f[:, 11].all() and f[:, 16].all()
    assert np.allclose(f[:, 17], np.tile(np.arange(8) / 8, 3)) and np.allclose(f[:, 12], np.repeat([0, 0.5, 1], 8))


def test_a_profile_is_one_stream_without_a_position():
    f = identity.stream_features(rows("p", 120, family="profile", dt=0.005, units="eV", r=1.0, z=0.0), "linear", 1)
    assert f.shape == (1, identity.N_FEAT) and f[0, 2] == 1 and not f[0, 12:].any()


def test_the_table_gives_each_signal_its_rows():
    reg = pd.concat([rows("a", 3), rows("b", 2)])
    t = identity.build_table(reg, [("a", "frame_dct", 3), ("b", "frame_dct", 2)])
    assert len(t) == 5 and t.features.shape == (5, identity.N_FEAT) and t.signals == {"a": (0, 3), "b": (3, 2)}
    assert t.index_of("b", 2).tolist() == [3, 4] and t.index_of("c", 4).tolist() == [-1, -1, -1, -1]
    with pytest.raises(ValueError):
        t.index_of("a", 2)
    with pytest.raises(ValueError, match="zz"):
        identity.build_table(reg, [("zz", "frame_dct", 1)])


def test_the_front_end_is_the_top_rung_of_the_fixed_codec():
    ip = tasks.SignalSpec("summary-ip", 0.00025, (1,), 0.0, 1.0)
    assert store.front_setting(ip, "frame_dct", "linear") == assign.Setting("frame_dct", (16,), "linear")
    psi = tasks.SignalSpec("equilibrium-psi", 0.005, (65, 65), 0.0, 1.0)
    assert store.front_setting(psi, "linear", "signal", 2) == assign.Setting("linear", (96, 2), "signal")
    small = tasks.SignalSpec("p", 0.005, (30,), 0.0, 1.0)
    assert store.front_setting(small, "linear", "signal", 4).k == (28, 4)
    mirnov = tasks.SignalSpec("s", 0.001024, (3, 256), 0.0, 1.0)
    assert store.front_setting(mirnov, "band_dct", "band") == assign.Setting("band_dct", (32,) * 8, "band")


def test_stream_values_of_a_series_keep_whole_frames_only():
    front = series_front()
    z = RNG.standard_normal((2, 50)).astype(np.float32)             # two whole frames of 20 samples, and 10 more
    v = store.stream_values(front, z)
    assert v.shape == (2, 2, 16) and v.dtype == np.float32
    assert np.allclose(v[1, 0, 0], z[0, 20:40].mean(), atol=1e-6)   # value 0 is the frame mean (1b)
    z[1, 20:40] = np.nan
    holed = store.stream_values(front, z)
    assert np.isnan(holed[1, 1]).all() and np.isfinite(holed[1, 0]).all() and np.isfinite(holed[0]).all()


def test_collect_skips_and_counts_a_shot_without_the_signal():
    front = series_front()
    shots = [(30000, RNG.standard_normal((2, 40)).astype(np.float32)), (30001, None),
             (30002, RNG.standard_normal((2, 60)).astype(np.float32))]
    s = store.collect("m", front, shots)
    assert s.values.shape == (5, 2, 16) and s.n_skipped == 1 and s.shots.tolist() == [30000, 30000, 30002, 30002, 30002]
    empty = store.collect("m", front, [(30001, None)])
    assert empty.values.shape == (0, 2, 16) and empty.n_skipped == 1


def test_a_store_is_saved_once_and_read_back(tmp_path):
    s = store.collect("m", series_front(), [(30000, RNG.standard_normal((2, 40)).astype(np.float32))])
    store.save_store({"m": s}, tmp_path / "train", {"shots": [30000]})
    back, meta = store.load_store(tmp_path / "train")
    assert np.array_equal(back["m"].values, s.values) and back["m"].shots.tolist() == [30000, 30000]
    assert meta["shots"] == [30000] and store.load_store(tmp_path / "train", names=[])[0] == {}
    with pytest.raises(FileExistsError):
        store.save_store({"m": s}, tmp_path / "train", {})


def test_the_norm_gives_each_stream_unit_scale_and_survives_missing_and_constant_streams():
    v = RNG.standard_normal((500, 3, 4)).astype(np.float32) * 7.0 + 2.0
    v[:, 1] = np.nan
    v[:, 2] = 5.0
    v[0, 0] = np.nan
    n = store.fit_norm(v)
    e = n.apply(v)
    assert n.center.shape == (3, 4) and n.scale.shape == (3,)
    assert np.sqrt(np.nanmean(e[:, 0] ** 2)) == pytest.approx(1.0, abs=1e-4) and n.scale[0] == pytest.approx(7.0, rel=0.1)
    assert n.scale[1] == 1.0 and not n.center[1].any() and n.scale[2] == 1.0 and not e[:, 2].any()
    assert np.isnan(e[0, 0]).all() and np.isnan(e[:, 1]).all()
    assert np.allclose(n.invert(e)[1:, [0, 2]], v[1:, [0, 2]], atol=1e-4)
