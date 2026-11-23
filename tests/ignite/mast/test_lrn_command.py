"""The arms of plan 1c, recoded assignments and their provenance."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tokamak_foundation_model.ignite.mast import cache, registry, tasks, transform
from tokamak_foundation_model.ignite.mast.tokenizers import assign, fit, grid
from tokamak_foundation_model.ignite.mast.tokenizers.learned import build, identity, recode, store

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "mast"
IP = tasks.SignalSpec("summary-ip", 0.00025, (1,), 3.5e5, 3.0e5)


def load(name):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def loaded(concat, n_tok, n_val, signals):
    recoder = types.SimpleNamespace(kind="linear", concat=concat, n_tok=n_tok, n_val=n_val, n_values=n_tok * n_val)
    meta = {"signals": {n: dict(zip(("kind", "domain", "n_streams", "values_per_token"), v)) for n, v in signals.items()}}
    return recode.Loaded(recoder, {}, None, meta, {})


def test_the_arms_cover_the_three_sides_of_each_comparison():
    tt = load("train_tokenizer")
    series = list(tt.MAGNETICS) + ["summary-ip"]
    streams = {**dict(zip(tt.MAGNETICS, (40, 18, 18, 15))), "summary-ip": 1, **{n: 24 for n in tt.BANDS},
               "equilibrium-lcfs_r": 1, "equilibrium-psi": 1}
    a = tt.arms(series, ["equilibrium-lcfs_r"], ["equilibrium-psi"], streams)
    assert a["klt.series"].signals == tuple(series) and not a["klt.series"].concat
    assert (a["klt.series"].n_tok, a["klt.series"].n_val) == (1, 16) and a["klt.mag"].signals == tuple(tt.MAGNETICS)
    assert a["pca.ccbv"].concat and a["pca.ccbv"].signals == (tt.SHORT["ccbv"],)
    assert (a["pca.ccbv"].n_tok, a["pca.ccbv"].n_val) == (160, 4) and (a["net.ccbv"].n_tok, a["net.ccbv"].n_val) == (40, 8)
    assert (a["pca.band.cc"].n_tok, a["pca.band.cc"].n_val) == (192, 4) and a["klt.band"].signals == tuple(tt.BANDS)
    # ruling C4: v1's 3 x (32 + 7 x 16) = 432 values per Mirnov signal, in tokens of 8
    assert (a["net.bandcat.omv"].n_tok, a["net.bandcat.omv"].n_val, a["net.bandcat.omv"].concat) == (54, 8, True)
    assert (a["net.bandcat.cc"].n_tok, a["net.bandcat.cc"].n_val) == (54, 8)
    assert (a["net.profile"].n_tok, a["net.profile"].n_val) == (24, 4) and (a["net.map"].n_tok, a["net.map"].n_val) == (48, 2)
    assert sum(not x.net for x in a.values()) == 11 and sum(x.net for x in a.values()) == 13
    assert tt.net_name("net.series", 2) == "net.series.s2" and tt.net_name("net.mag", 0, "lr1e4") == "net.mag.s0.lr1e4"
    # ruling C5: obr, obv and flux use tokens of 6, so that v1's 18, 36 and 30 values per frame divide exactly
    for short, linear_tokens, net_tokens in (("obr", 48, 24), ("obv", 48, 24), ("flux", 40, 20)):
        assert (a[f"pca.{short}"].n_tok, a[f"pca.{short}"].n_val) == (linear_tokens, 6)
        assert (a[f"net.{short}"].n_tok, a[f"net.{short}"].n_val) == (net_tokens, 6)
        assert a[f"pca.{short}"].concat and a[f"net.{short}"].concat and a[f"net.{short}"].net
    assert [x.n_tok * x.n_val for x in (a["pca.obr"], a["pca.obv"], a["pca.flux"])] == [288, 288, 240]
    assert [x.n_tok * x.n_val for x in (a["net.obr"], a["net.obv"], a["net.flux"])] == [144, 144, 120]
    assert a["net.series"].signals == tuple(series) and not a["net.series"].concat and a["net.series"].net


def test_recoding_an_assignment_keeps_the_values_per_frame():
    base = {"m": assign.Setting("frame_dct", (2,)), "s": assign.Setting("band_dct", (16, 2, 2, 2, 2, 2, 2, 2), "band"),
            "p": assign.Setting("linear", (32, 4), "signal"), "x": assign.Setting("frame_dct", (4,))}
    per = loaded(False, 1, 16, {"m": ("frame_dct", "linear", 40, 1)})
    got = build.recode_assignment(base, per, "klt.series")
    assert got["m"] == assign.Setting("recoded", (2,), "linear", source="klt.series") and got["x"] == base["x"]
    cat = loaded(True, 160, 4, {"m": ("frame_dct", "linear", 40, 1)})
    assert build.recode_assignment(base, cat, "pca.ccbv")["m"] == assign.Setting(
        "recoded_concat", (80,), "linear", source="pca.ccbv")                      # 40 channels x K = 2
    assert build.recode_assignment(base, cat, "pca.ccbv", scale=0.25)["m"].k == (20,)
    assert build.recode_assignment(base, cat, "pca.ccbv", scale=0.01)["m"].k == (4,)   # at least one token
    band = loaded(False, 1, 32, {"s": ("band_dct", "band", 24, 1)})
    assert build.recode_assignment(base, band, "klt.band")["s"].k == (16, 2, 2, 2, 2, 2, 2, 2)
    bandcat = loaded(True, 24, 8, {"s": ("band_dct", "band", 24, 1)})
    assert build.recode_assignment(base, bandcat, "net.bandcat.cc.s0")["s"].k == (88,)  # 3 x 30 = 90, down to tokens of 8
    profile = loaded(False, 24, 4, {"p": ("linear", "signal", 1, 4)})
    assert build.recode_assignment(base, profile, "net.profile.s0")["p"].k == (32,)
    assert build.recode_assignment(base, loaded(False, 1, 16, {"m": ("frame_dct", "linear", 40, 1),
                                                               "x": ("frame_dct", "linear", 3, 1)}),
                                   "klt.series", signals=["x"])["m"] == base["m"]
    for match, args, kw in (("scale", (base, per, "klt.series"), {"scale": 0.5}),
                            ("domain", ({"m": assign.Setting("frame_dct", (2,), "compressed")}, per, "klt.series"), {}),
                            ("recoded", (got, per, "klt.series"), {}),
                            ("holds", ({"m": assign.Setting("frame_dct", (16,))},
                                       loaded(True, 40, 8, {"m": ("frame_dct", "linear", 40, 1)}), "net.ccbv.s0"), {})):
        with pytest.raises(ValueError, match=match):
            build.recode_assignment(*args, **kw)


def provenance_fixture(tmp_path):
    cache_root, fit_dir = tmp_path / "cache", tmp_path / "fit"
    cache_root.mkdir()
    fit_dir.mkdir()
    (cache_root / "stats.npz").write_bytes(b"statistics")
    sha = transform.stats_sha256(cache_root / "stats.npz")
    (cache_root / "manifest.json").write_text(json.dumps({"stats_sha256": sha, "tokamark_revision": "146b0a0f8738"}))
    st = transform.ChannelStats(compress="none", scale=np.ones(1), mean=np.zeros(1), std=np.ones(1))
    shot = np.random.default_rng(0).standard_normal((1, 400)).astype(np.float32)
    fit.save_fit({"summary-ip": fit.fit_signal(IP, st, [shot], kind="frame_dct")}, fit_dir, {"stats_sha256": sha})
    return cache_root, fit_dir


def test_provenance_of_a_plain_assignment_is_unchanged(tmp_path):
    rt = load("run_tokenizer")
    cache_root, fit_dir = provenance_fixture(tmp_path)
    assign.save_assignment({"summary-ip": assign.Setting("frame_dct", (4,))}, tmp_path / "v1.json")
    p = rt.provenance_of(tmp_path / "v1.json", fit_dir, cache_root, recoders_root=tmp_path / "recoders")
    assert list(p) == ["tokenizer_code_sha256", "assignment_sha256", "fit_sha256", "stats_sha256", "tokamark_revision",
                       "data_root"]


def test_provenance_names_the_recoders_bytes(tmp_path):
    rt = load("run_tokenizer")
    cache_root, fit_dir = provenance_fixture(tmp_path)
    table = identity.StreamTable({"summary-ip": (0, 1)}, np.zeros((1, identity.N_FEAT), dtype=np.float32))
    norm = store.Norm(np.zeros((1, 16)), np.ones(1))
    meta = {"signals": {"summary-ip": {"kind": "frame_dct", "domain": "linear", "n_streams": 1, "values_per_token": 1}}}
    for name, shift in (("klt.x", 0.0), ("klt.y", 1.0)):
        recode.save_recoder(tmp_path / "recoders" / name, recode.LinearRecoder(np.full(16, shift), np.eye(16)),
                            {"summary-ip": norm}, table, meta)
        assign.save_assignment({"summary-ip": assign.Setting("recoded", (4,), source=name)}, tmp_path / f"{name}.json")
    px, py = (rt.provenance_of(tmp_path / f"{n}.json", fit_dir, cache_root, recoders_root=tmp_path / "recoders")
              for n in ("klt.x", "klt.y"))
    assert list(px) == ["tokenizer_code_sha256", "assignment_sha256", "fit_sha256", "stats_sha256",
                        "learned_code_sha256", "sources_sha256", "tokamark_revision", "data_root"]
    assert px["sources_sha256"] != py["sources_sha256"] and len(px["learned_code_sha256"]) == 64
    assign.save_assignment({"summary-ip": assign.Setting("recoded", (4,), source="gone")}, tmp_path / "gone.json")
    with pytest.raises(FileNotFoundError):
        rt.provenance_of(tmp_path / "gone.json", fit_dir, cache_root, recoders_root=tmp_path / "recoders")


def test_left_over_falls_to_zero_with_all_values():
    tt = load("train_tokenizer")
    e = np.random.default_rng(0).standard_normal((500, 16))
    left = tt.left_over(recode.fit_linear(e, 16), e, np.zeros(500, dtype=np.int64), [1, 4, 16])
    assert 1.0 > left[0] > left[1] > left[2] and left[2] < 1e-8


# ------------------------------------------------------------------ the commands, on one toy signal

STATS = transform.ChannelStats(compress="none", scale=np.ones(1), mean=np.zeros(1), std=np.ones(1))
COIL_SHOTS = (1, 2, 3, 4, 5)                   # 1-3 train (the fit's shots), 4-5 val_select


def toy_world(tmp_path, monkeypatch):
    """A cache of five shots of `summary-ip` (20 frames of 16 values each), a fit on shots 1-3 and the
    assignment ``v1`` with K = 4. The trial root is `tmp_path`; `registry.all_signals` and the split are patched."""
    monkeypatch.setenv("IGNITE_MAST_ROOT", str(tmp_path))
    root = tmp_path / "cache" / "v1"
    (root / "shots").mkdir(parents=True)
    transform.save_stats({"summary-ip": STATS}, root / "stats.npz")
    sha = transform.stats_sha256(root / "stats.npz")
    (root / "manifest.json").write_text(json.dumps({"stats_sha256": sha, "tokamark_revision": "146b0a0f8738"}))
    reg = pd.DataFrame([{"device": "mast", "signal": "summary-ip", "channel": 0, "channel_name": "ip", "family": "slow",
                         "dt": IP.dt, "units": "A", "imas_path": "", "r": np.nan, "z": np.nan, "tm_mean": IP.mean,
                         "tm_std": IP.std, "roles": "inp+out", "sat_abs": np.nan, "valid_fraction": 1.0,
                         "dead_fraction": 0.0, "compress": "none", "scale": 1.0, "mean": 0.0, "std": 1.0}])
    registry.save_registry(reg, root / "registry.csv")
    zs = {}
    for shot in COIL_SHOTS:
        z = np.random.default_rng(shot).standard_normal((1, 400)).cumsum(axis=-1).astype(np.float32) * 0.2
        zs[shot] = z
        raw = {"time": 0.1 + IP.dt * np.arange(400), "values": z.astype(np.float64)}
        cache.write_shot(root / "shots" / f"{shot}.h5", shot, "train", {"summary-ip": raw}, [IP], {"summary-ip": STATS},
                         reg, stats_sha=sha)
    fits = {"summary-ip": fit.fit_signal(IP, STATS, [zs[s] for s in (1, 2, 3)], kind="frame_dct")}
    fit_dir = tmp_path / "stage1" / "fit" / "v1"
    fit.save_fit(fits, fit_dir, {"stats_sha256": sha, "shots": [1, 2, 3]})
    assign.save_assignment({"summary-ip": assign.Setting("frame_dct", (4,))}, tmp_path / "stage1" / "assignments" / "v1.json")
    return root, fit_dir, fits, zs, sha


def toy_script(monkeypatch, name, **extra):
    module = load(name)
    monkeypatch.setattr(module.registry, "all_signals", lambda: [IP])
    monkeypatch.setattr(module.tasks, "split_shots", lambda split: [4, 5] if split == "val_select" else [1, 2, 3])
    for attr, value in extra.items():
        monkeypatch.setattr(module, attr, value)
    return module


def test_store_and_linear_write_once_and_recode_reads_what_they_wrote(tmp_path, monkeypatch, capsys):
    root, _, _, _, sha = toy_world(tmp_path, monkeypatch)
    tt = toy_script(monkeypatch, "train_tokenizer")
    arm = tt.Arm("klt.series", ("summary-ip",), False, False, 1, 16)
    monkeypatch.setattr(tt, "arms", lambda *a: {"klt.series": arm, "pca.x": tt.Arm("pca.x", ("summary-ip",), True, False, 4, 4)})
    argv = ["store", "--n-val", "2", "--workers", "1", "--cache-dir", str(root)]
    assert tt.main(argv) == 0
    out = capsys.readouterr().out
    assert out.count("summary-ip") == 1 and "no data" not in out
    train, meta = store.load_store(tmp_path / "stage1" / "store" / "v1" / "train")
    val, val_meta = store.load_store(tmp_path / "stage1" / "store" / "v1" / "val")
    assert train["summary-ip"].values.shape == (60, 1, 16) and val["summary-ip"].values.shape == (40, 1, 16)
    assert meta["shots"] == [1, 2, 3] and val_meta["shots"] == [4, 5] and meta["stats_sha256"] == sha
    assert meta["signals"]["summary-ip"] == {"kind": "frame_dct", "domain": "linear", "n_streams": 1, "values_per_token": 1}
    assert {"assignment_sha256", "fit_sha256", "learned_code_sha256", "date"} <= set(meta)
    assert tt.main(argv) == 2 and "exists" in capsys.readouterr().err                 # written once

    assert tt.main(["linear", "--arm", "nope"]) == 2
    assert tt.main(["linear", "--arm", "klt.series"]) == 0
    line = [ln for ln in capsys.readouterr().out.splitlines() if "summary-ip" in ln]
    assert len(line) == 1
    fractions = [float(v) for v in line[0].split("left over")[1].replace(",", " ").split()[1::2][:5]]
    assert len(fractions) == 5 and 1 > fractions[0] > fractions[1] > fractions[2] > fractions[3] > fractions[4] >= 0
    recoder = recode.load_recoder(tmp_path / "stage1" / "recoders" / "klt.series")
    assert recoder.recoder.n_values == 16 and "summary-ip" in recoder.ranges and recoder.meta["signals"]["summary-ip"]["n_streams"] == 1
    assert tt.main(["linear", "--arm", "klt.series"]) == 2                              # an existing recoder is refused
    assert tt.main(["arms"]) == 0 and "klt.series" in capsys.readouterr().out

    rt = toy_script(monkeypatch, "run_tokenizer")
    capsys.readouterr()
    assert rt.main(["recode", "--name", "t-klt", "--from", "v1", "--arm", "klt.series"]) == 0
    out = capsys.readouterr().out
    assert "summary-ip" in out and "4 -> 4" in out
    made = assign.load_assignment(tmp_path / "stage1" / "assignments" / "t-klt.json")
    assert made["summary-ip"] == assign.Setting("recoded", (4,), "linear", source="klt.series")
    assert rt.main(["recode", "--name", "t-klt", "--from", "v1", "--arm", "klt.series"]) == 2   # a name is not reused
    assert rt.main(["recode", "--name", "t-two", "--from", "v1", "--arm", "klt.series,klt.series"]) == 2   # overlap
    assert rt.main(["recode", "--name", "t-sc", "--from", "v1", "--arm", "klt.series", "--scale", "0.5"]) == 2
    assert rt.main(["recode", "--name", "t-nope", "--from", "v1", "--arm", "gone"]) == 2
    assert not (tmp_path / "stage1" / "assignments" / "t-two.json").exists()


# ------------------------------------------------------------------ the gate reads a recoded assignment (ruling C2)

def gate_args(tmp_path, root, name):
    return ["gate", "--assignment", str(tmp_path / name), "--n-shots", "2", "--workers", "1", "--cache-dir", str(root),
            "--stage0-dir", str(tmp_path / "stage0")]


def gate_script(monkeypatch, toy_task):
    rt = load("run_tokenizer")
    monkeypatch.setattr(rt.tasks, "ALL_TASKS", ("task_toy",))
    monkeypatch.setattr(rt.tasks, "load_task", lambda name: toy_task)
    monkeypatch.setattr(rt.tasks, "best_published", lambda name: 0.5)
    monkeypatch.setattr(rt.tasks, "split_shots", lambda split: [4, 5])
    monkeypatch.setattr(rt, "_stored_floors", lambda stage1, t, who, mode, label, prov: (
        {"summary-ip": {"unquantized": 0.1, "ranged": 0.11, "nearest": 0.2}.get(mode, 0.3)}, 0.3, ""))
    return rt


def save_toy_recoder(tmp_path, fits, zs):
    front = store.front_codec(IP, STATS, fits["summary-ip"], "frame_dct", "linear")
    fv = np.concatenate([store.stream_values(front, zs[s]) for s in (1, 2, 3)])        # (60, 1, 16)
    norm = store.fit_norm(fv)
    e = norm.apply(fv).reshape(-1, 16)
    r = recode.fit_linear(e, 16)
    ranges = grid.fit_ranges(r.encode(e, np.zeros(len(e), dtype=np.int64)).reshape(-1, 1, 16))
    table = identity.StreamTable({"summary-ip": (0, 1)}, np.zeros((1, identity.N_FEAT), dtype=np.float32))
    meta = {"signals": {"summary-ip": {"kind": "frame_dct", "domain": "linear", "n_streams": 1, "values_per_token": 1}}}
    recode.save_recoder(tmp_path / "stage1" / "recoders" / "klt.ip", r, {"summary-ip": norm}, table, meta,
                        {"summary-ip": ranges})


def test_the_gate_measures_the_cache_side_of_a_recoded_assignment(tmp_path, monkeypatch, capsys, toy_task):
    root, _, fits, zs, _ = toy_world(tmp_path, monkeypatch)
    save_toy_recoder(tmp_path, fits, zs)
    assign.save_assignment({"summary-ip": assign.Setting("recoded", (4,), "linear", source="klt.ip")}, tmp_path / "r1.json")
    rt = gate_script(monkeypatch, toy_task)
    assert rt.main(gate_args(tmp_path, root, "r1.json")) in (0, 1)
    out = capsys.readouterr().out
    assert "refused" not in out and "1 rows" in out
    row = pd.read_csv(tmp_path / "stage1" / "gate" / "r1.csv").iloc[0]
    assert row["codec"] == "recoded" and row["values_per_frame"] == 4
    assert np.isfinite(row["worst_channel_error"]) and np.isfinite(row["bins_used_min"])   # the recoded codec was built and read


def test_the_gate_refuses_a_concatenated_band_codec_and_names_the_signal(tmp_path, monkeypatch, capsys, toy_task):
    import dataclasses
    root, _, _, _, _ = toy_world(tmp_path, monkeypatch)
    cc = tasks.SignalSpec("magnetics-b_field_tor_probe_cc_field", 0.00002, (3, 256), 0.0, 1.0)
    assign.save_assignment({cc.name: assign.Setting("recoded_concat", (88,), "band", source="bc")}, tmp_path / "bc.json")
    rt = gate_script(monkeypatch, dataclasses.replace(toy_task, outputs=(cc,)))
    assert rt.main(gate_args(tmp_path, root, "bc.json")) == 2
    err = capsys.readouterr().err
    assert "refused" in err and cc.name in err and "concatenated" in err
    assert not (tmp_path / "stage1" / "gate").exists()


def test_floor_still_refuses_the_gpu_flags_for_a_recoded_assignment(tmp_path, monkeypatch, capsys):
    toy_world(tmp_path, monkeypatch)
    assign.save_assignment({"summary-ip": assign.Setting("recoded", (4,), "linear", source="klt.ip")}, tmp_path / "r1.json")
    rt = load("run_tokenizer")
    assert rt.main(["score", "--task", "task_1-1", "--split", "val_select", "--assignment", str(tmp_path / "r1.json"),
                    "--mode", "unquantized", "--gpu", "0"]) == 2                              # ruling C7
    assert "uses no GPU" in capsys.readouterr().err
