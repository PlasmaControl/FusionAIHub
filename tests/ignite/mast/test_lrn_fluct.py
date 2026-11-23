"""The band above the trend of a 50 kHz channel, as a statistic (plan 1c)."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tokamak_foundation_model.ignite.mast import cache, registry, tasks, transform
from tokamak_foundation_model.ignite.mast.tokenizers import assign, codecs, fit
from tokamak_foundation_model.ignite.mast.tokenizers.learned import fluct

I = np.arange(250)


def wave(k, amplitude):
    return amplitude * np.cos(np.pi * (I + 0.5) * k / 250)          # cosine coefficient k of a 5 ms frame


def test_the_band_power_adds_up_to_the_frames_mean_square():
    u = np.random.default_rng(0).standard_normal((3, 250))
    assert np.allclose(fluct.fluct_power(u, 0).sum(-1), (u ** 2).mean(-1))
    u = wave(2, 2.0) + wave(37, 0.5)                                 # a trend and a tone at 3.7 kHz
    p = fluct.fluct_power(u, 8)
    assert p.shape == (25,) and p[3] == pytest.approx(0.125) and np.delete(p, 3).max() < 1e-12
    assert fluct.fluct_power(u, 0)[0] == pytest.approx(2.0)          # without a trend path the trend is in the band
    assert np.isnan(fluct.fluct_power(np.where(I == 7, np.nan, u), 8)).all()
    with pytest.raises(ValueError):
        fluct.fluct_power(np.zeros(240), 8)


def test_persistence_tells_a_slow_level_from_noise():
    rng = np.random.default_rng(1)
    assert fluct.frame_persistence(10.0 ** np.cumsum(0.1 * rng.standard_normal(2000))) > 0.9
    assert abs(fluct.frame_persistence(10.0 ** rng.standard_normal(2000))) < 0.2
    assert np.isnan(fluct.frame_persistence(np.array([1.0, np.nan, 2.0, 0.0])))
    assert np.isnan(fluct.frame_persistence(np.ones(10)))


def test_the_summary_of_white_noise_is_flat_and_skips_frames_with_gaps():
    z = np.random.default_rng(2).standard_normal((2, 2500 + 30))    # ten whole frames per channel
    z[0, :250] = np.nan
    s = fluct.summarize([z], 250, 10)
    assert s["n_frames"] == 19 and s["share_by_bin"].sum() == pytest.approx(1.0) and s["share_by_bin"][0] == 0.0
    assert s["share_below_5khz"] == pytest.approx(4 / 24, abs=0.03)
    assert s["band_to_trend"] == pytest.approx(240 / 9, rel=0.4)
    none = fluct.summarize([np.full((2, 500), np.nan)], 250, 10)
    assert none["n_frames"] == 0 and np.isnan(none["share_below_5khz"]) and np.isnan(none["persistence"])


# ------------------------------------------------------------------------- edges the brief's tests leave open

def test_the_band_power_is_the_dct_values_of_1b_to_1e_12():
    # fluct.py takes the cosine values from scipy directly (importing `codecs` loads torch, which the module must not);
    # this is the identity with 1b's values that makes the two the same numbers
    x = np.random.default_rng(3).standard_normal((4, 250)) * 3.0 + 1.0
    assert np.abs(fluct.cosine_values(x) - codecs.dct_values(x, 250)).max() < 1e-12
    assert np.abs(fluct.fluct_power(x, 5) - _band_from_values(codecs.dct_values(x, 250), 5)).max() < 1e-12


def _band_from_values(values, k_trend, n_bins=25):
    p = values ** 2
    p[..., :k_trend] = 0.0
    return p.reshape(p.shape[:-1] + (n_bins, -1)).sum(-1)


def test_a_trend_of_one_value_has_no_ratio_but_the_other_keys_are_computed():
    z = np.random.default_rng(4).standard_normal((3, 1000))
    s = fluct.summarize([z], 250, 1)
    assert np.isnan(s["band_to_trend"])
    assert s["n_frames"] == 12 and s["share_by_bin"].sum() == pytest.approx(1.0)
    assert s["share_below_5khz"] == pytest.approx(5 / 25, abs=0.04) and np.isfinite(s["persistence"])
    s0 = fluct.summarize([z], 250, 0)                                # no trend path at all: the mean is in the band
    assert np.isnan(s0["band_to_trend"]) and s0["share_by_bin"][0] > 0
    with pytest.raises(ValueError):
        fluct.summarize([z], 250, -1)


def test_only_whole_frames_from_the_first_sample_are_used():
    rng = np.random.default_rng(5)
    z = rng.standard_normal((1, 749))                                # two whole frames and 249 samples
    assert fluct.summarize([z], 250, 4)["n_frames"] == 2
    assert fluct.summarize([z[:, :249]], 250, 4)["n_frames"] == 0
    z2 = z.copy()
    z2[0, 700:] = 1e6                                                # in the partial frame only: not read
    a, b = fluct.summarize([z], 250, 4), fluct.summarize([z2], 250, 4)
    assert np.array_equal(a["share_by_bin"], b["share_by_bin"])
    z3 = z.copy()
    z3[0, 300] = np.nan                                              # in the second whole frame: it is left out
    assert fluct.summarize([z3], 250, 4)["n_frames"] == 1


def test_persistence_is_nan_for_a_constant_without_a_warning_and_the_median_is_over_finite_ones():
    assert np.isnan(fluct.frame_persistence(np.array([1.0, 2.0, 2.0, 2.0, 2.0])))   # one side constant
    assert np.isnan(fluct.frame_persistence(np.array([1.0, 3.0])))                   # fewer than three pairs
    assert np.isnan(fluct.frame_persistence(np.array([1.0, 3.0, 2.0])))               # two pairs
    assert np.isnan(fluct.frame_persistence(np.array([-1.0, -2.0, -3.0, 0.0, 0.0])))  # nothing positive
    assert np.isnan(fluct.frame_persistence(np.array([1.0, np.inf, 2.0, 5.0, 3.0, np.inf])))   # too few pairs left
    assert fluct.frame_persistence(np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])) == pytest.approx(1.0)
    rng = np.random.default_rng(6)
    slow = 10.0 ** np.cumsum(0.1 * rng.standard_normal(40))
    # one channel of a shot is constant (dead, 0 everywhere in its band power: no persistence), the other is slow
    # white noise of a changing level, built as frames of a changing amplitude
    amp = np.repeat(np.sqrt(slow), 250)
    z = np.vstack([np.zeros(40 * 250), amp * rng.standard_normal(40 * 250)])
    s = fluct.summarize([z], 250, 4)
    assert s["n_frames"] == 80 and s["persistence"] > 0.8            # the dead channel counts its frames, not a persistence


# ------------------------------------------------------------------------------------------- the command

def test_fluct_py_imports_no_torch():
    code = ("import sys; from tokamak_foundation_model.ignite.mast.tokenizers.learned import fluct; "
            "sys.exit(1 if 'torch' in sys.modules else 0)")
    done = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-500:]


SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "mast"
STATS = transform.ChannelStats(compress="none", scale=np.ones(2), mean=np.zeros(2), std=np.ones(2))
A = tasks.SignalSpec("magnetics-toy_a", 2e-5, (2,), 0.0, 1.0)              # 50 kHz, K 3, present in every shot
B = tasks.SignalSpec("magnetics-toy_b", 2e-5, (2,), 0.0, 1.0)              # 50 kHz, K 1, absent in the validation shots
SLOW = tasks.SignalSpec("magnetics-toy_slow", 5e-3, (1,), 0.0, 1.0)         # one sample per frame: not a 50 kHz signal
SHOTS = (1, 2, 3, 4, 5, 6)                                                 # 1-3 train (the fit's shots), 4-6 val_select


def load_script():
    spec = importlib.util.spec_from_file_location("train_tokenizer_fluct_under_test", SCRIPTS / "train_tokenizer.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def toy_world(tmp_path, monkeypatch):
    monkeypatch.setenv("IGNITE_MAST_ROOT", str(tmp_path))
    root = tmp_path / "cache" / "v1"
    (root / "shots").mkdir(parents=True)
    stats = {s.name: transform.ChannelStats(compress="none", scale=np.ones(s.values_shape[0]),
                                            mean=np.zeros(s.values_shape[0]), std=np.ones(s.values_shape[0]))
             for s in (A, B, SLOW)}
    transform.save_stats(stats, root / "stats.npz")
    sha = transform.stats_sha256(root / "stats.npz")
    (root / "manifest.json").write_text(json.dumps({"stats_sha256": sha, "tokamark_revision": "146b0a0f8738"}))
    rows = []
    for s in (A, B, SLOW):
        for c in range(s.values_shape[0]):
            rows.append({"device": "mast", "signal": s.name, "channel": c, "channel_name": f"c{c}", "family": "slow",
                         "dt": s.dt, "units": "A", "imas_path": "", "r": np.nan, "z": np.nan, "tm_mean": s.mean,
                         "tm_std": s.std, "roles": "inp+out", "sat_abs": np.nan, "valid_fraction": 1.0,
                         "dead_fraction": 0.0, "compress": "none", "scale": 1.0, "mean": 0.0, "std": 1.0})
    reg = pd.DataFrame(rows)
    registry.save_registry(reg, root / "registry.csv")
    zs = {}
    for shot in SHOTS:
        rng = np.random.default_rng(shot)
        n = 250 * 12 + 100                                                  # twelve whole frames and a partial one
        raw_a = {"time": 0.1 + A.dt * np.arange(n), "values": rng.standard_normal((2, n))}
        raw = {A.name: raw_a, SLOW.name: {"time": 0.1 + SLOW.dt * np.arange(40), "values": rng.standard_normal((1, 40))}}
        if shot <= 3:
            raw[B.name] = {"time": 0.1 + B.dt * np.arange(n), "values": rng.standard_normal((2, n))}
        zs[shot] = {k: v["values"].astype(np.float32) for k, v in raw.items()}
        cache.write_shot(root / "shots" / f"{shot}.h5", shot, "train", raw, [A, B, SLOW], stats, reg, stats_sha=sha)
    fits = {s.name: fit.fit_signal(s, stats[s.name], [zs[k].get(s.name) for k in (1, 2, 3)], kind="frame_dct")
            for s in (A, B, SLOW)}
    fit.save_fit(fits, tmp_path / "stage1" / "fit" / "v1", {"stats_sha256": sha, "shots": [1, 2, 3]})
    assign.save_assignment({A.name: assign.Setting("frame_dct", (3,)), B.name: assign.Setting("frame_dct", (1,)),
                            SLOW.name: assign.Setting("frame_dct", (1,))}, tmp_path / "stage1" / "assignments" / "v1.json")
    return root


def toy_script(monkeypatch):
    module = load_script()
    monkeypatch.setattr(module.registry, "all_signals", lambda: [A, B, SLOW])
    monkeypatch.setattr(module.tasks, "split_shots", lambda split: [4, 5, 6] if split == "val_select" else [1, 2, 3])
    return module


def test_the_command_measures_the_50_khz_signals_writes_the_csv_and_prints_no_data(tmp_path, monkeypatch, capsys):
    root = toy_world(tmp_path, monkeypatch)
    tt = toy_script(monkeypatch)
    out_csv = tmp_path / "stage1" / "compare" / "fluct.csv"
    assert tt.main(["fluct", "--n-shots", "3", "--workers", "1", "--cache-dir", str(root)]) == 0
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if "magnetics-toy" in ln]
    assert len(lines) == 2 and "2 signals with 250 samples per frame" in out         # the slow signal is not one of them
    assert "toy_slow" not in out
    a_line = next(ln for ln in lines if "toy_a" in ln)
    assert "no data" not in a_line and "bin_" not in a_line
    assert "no data" in next(ln for ln in lines if "toy_b" in ln)
    df = pd.read_csv(out_csv)
    assert list(df.columns[:6]) == ["signal", "k_trend", "n_frames", "band_to_trend", "share_below_5khz", "persistence"]
    assert list(df.columns[6:]) == [f"bin_{i:02d}" for i in range(25)]
    a = df.set_index("signal").loc[A.name]
    assert a["k_trend"] == 3 and a["n_frames"] == 3 * 2 * 12 and a["band_to_trend"] > 1
    assert 0 < a["share_below_5khz"] < 1 and -1 <= a["persistence"] <= 1
    assert a[[f"bin_{i:02d}" for i in range(25)]].sum() == pytest.approx(1.0)
    b = df.set_index("signal").loc[B.name]
    assert b["k_trend"] == 1 and b["n_frames"] == 0 and np.isnan(b["band_to_trend"]) and np.isnan(b["bin_00"])
    assert not list(out_csv.parent.glob("*.tmp"))                                        # written atomically
    # a second run replaces the file; two workers (spawned processes) give the same numbers as one
    assert tt.main(["fluct", "--n-shots", "3", "--workers", "2", "--cache-dir", str(root)]) == 0
    again = pd.read_csv(out_csv)
    assert np.allclose(again.drop(columns="signal").to_numpy(float), df.drop(columns="signal").to_numpy(float),
                       equal_nan=True)


def test_the_command_refuses_before_reading_or_writing(tmp_path, monkeypatch, capsys):
    root = toy_world(tmp_path, monkeypatch)
    tt = toy_script(monkeypatch)
    compare = tmp_path / "stage1" / "compare"
    assert tt.main(["fluct", "--n-shots", "0", "--cache-dir", str(root)]) == 2
    assert tt.main(["fluct", "--workers", "0", "--cache-dir", str(root)]) == 2
    assert tt.main(["fluct", "--cache-dir", str(tmp_path / "nowhere")]) == 2               # no manifest
    assert "manifest" in capsys.readouterr().err
    (tmp_path / "stage1" / "assignments" / "v1.json").rename(tmp_path / "stage1" / "assignments" / "gone.json")
    assert tt.main(["fluct", "--cache-dir", str(root)]) == 2
    assert "assignment" in capsys.readouterr().err
    assert not (compare / "fluct.csv").exists() and not compare.exists()
    monkeypatch.setattr(tt.registry, "all_signals", lambda: [SLOW])
    (tmp_path / "stage1" / "assignments" / "gone.json").rename(tmp_path / "stage1" / "assignments" / "v1.json")
    assert tt.main(["fluct", "--cache-dir", str(root)]) == 2                                # no 50 kHz signal at all
    assert "250" in capsys.readouterr().err and not compare.exists()
