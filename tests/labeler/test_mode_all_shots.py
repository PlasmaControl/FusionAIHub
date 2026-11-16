"""The StellarAI all-shot MHD mode run (`scripts/labeler/mode_all_shots.py`, `labeler.modes.fetch`):
the corpus file omnimode reads, the EFIT record, which view files are staged, what `clean`
frees, and the publish text for the redesign. Synthetic shots only; nothing is fetched."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from labeler.modes import fetch as F

from .modes_synth import make_shot

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts/labeler"
sys.path.insert(0, str(SCRIPTS))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load("mode_all_shots")
SHOT = 100000  # synthetic


def _probe(t0_ms, n, fs=F.FS_HZ, gap=None):
    t = t0_ms + np.arange(n) * (1000.0 / fs)
    y = np.sin(2 * np.pi * 10e3 * t / 1000.0).astype(np.float32)
    if gap is not None:
        keep = (t < gap[0]) | (t > gap[1])
        t, y = t[keep], y[keep]
    return t, y


# ------------------------------------------------------------------ fetch: corpus


def test_a_probe_on_the_grid_comes_back_unchanged_and_a_gap_is_nan():
    ref = _probe(-50.0, 5001)
    gappy = _probe(-50.0, 5001, gap=(-45.0, -40.0))
    times, Y, rows = F.assemble({15: ref, 0: gappy, 22: _probe(-49.0, 2000)})
    assert rows == [0, 15, 22]
    assert times[0] == -50.0 and len(times) == 5001
    np.testing.assert_array_equal(Y[1], ref[1])
    inside = (times > -44.99) & (times < -40.01)
    outside = (times < -45.1) | (times > -39.9)
    assert np.isnan(Y[0][inside]).all() and np.isfinite(Y[0][outside]).all()
    # a shorter probe: NaN outside its own record, its samples where it has them
    assert np.isnan(Y[2][times < -49.0 - 1e-9]).all()
    assert np.isfinite(Y[2][(times >= -48.99) & (times <= -45.01)]).all()


def test_a_probe_at_another_rate_and_no_toroidal_probe_are_refused():
    with pytest.raises(F.RateError):
        F.assemble({15: _probe(0.0, 1000, fs=250_000.0)})
    with pytest.raises(ValueError, match="no toroidal"):
        F.assemble({0: _probe(0.0, 1000)})


def test_the_corpus_file_has_the_rows_omnimode_reads_at_500_khz(tmp_path):
    times, Y, rows = F.assemble({r: _probe(-100.0, 400_001) for r in F.ROWS})
    ip_t = np.arange(-100.0, 700.0, 0.5)
    path = F.corpus_path(tmp_path, SHOT)
    size = F.write_corpus(path, times, Y, rows, ip_t, np.full(ip_t.size, 1.2e6))
    assert size > 0 and not path.with_name(path.name + ".part").exists()
    with h5py.File(path, "r") as h:
        x = h["mirnov/xdata"][:]
        assert x.dtype == np.float32 and h["mirnov/ydata"].shape == (29, x.size)
        # omnimode masks.read_record: (n - 1) / (x[-1] - x[0]) within 1e-6 of 500 kHz
        rate = (x.size - 1) / (float(x[-1]) - float(x[0]))
        assert abs(rate / F.FS_HZ - 1) < 1e-6
        y = h["mirnov/ydata"]
        assert np.isfinite(y[15]).all() and np.isnan(y[4]).all()  # an unread row is NaN
        assert h["ip/ydata"].shape == (1, ip_t.size)
        np.testing.assert_allclose(h["ip/xdata"][:2], [-0.1, -0.0995], rtol=1e-6)


# ------------------------------------------------------------------ fetch: equilibrium


def test_an_efit_record_has_omnimode_fields_with_psirz_as_time_z_r():
    nt, nr, nz = 4, 5, 6
    r, z = np.linspace(1.0, 2.4, nr), np.linspace(-1.2, 1.2, nz)
    ssimag = np.linspace(-0.5, -0.4, nt)
    tzr = np.zeros((nt, nz, nr)) + 1.0
    tzr[:, 3, 2] = ssimag  # the axis at (r[2], z[3])
    g = {
        "gtime": np.array([100.0, 120.0, 140.0, 160.0]),
        "psirz": np.transpose(tzr, (2, 1, 0)),  # stored (r, z, t)
        "ssimag": ssimag,
        "ssibry": np.zeros(nt),
        "rmaxis": np.full(nt, r[2]),
        "zmaxis": np.full(nt, z[3]),
        "r": r,
        "z": z,
        "qpsi": np.tile(np.linspace(1, 4, 7), (nt, 1)).T,  # (psi, t): transposed
        "fpol": np.ones((nt, 7)),
    }
    a = {"atime": g["gtime"], "qmin": np.full(nt, 1.1), "q95": np.full(nt, 4.0)}
    rec = F.eq_record(g, a, "efit02er")
    assert rec["psirz"].shape == (nt, nz, nr) and rec["psirz"].dtype == np.float32
    np.testing.assert_allclose(rec["psirz"][:, 3, 2], ssimag)
    assert rec["qpsi"].shape == (nt, 7) and rec["qpsi"][0, -1] == 4
    assert str(rec["tree"]) == "efit02er" and np.isnan(rec["ece_freq"]).all()
    with pytest.raises(ValueError):
        F.eq_record({**g, "gtime": np.array([100.0])}, a, "efit02er")


# ------------------------------------------------------------------ stage: the view files


def test_staging_takes_the_probe_and_ip_groups_and_one_layout_of_each_tree():
    files = {
        "meta/index.json": 10,
        f"ptdata/{SHOT}.MPI": 900,
        f"ptdata/{SHOT}.MPA": 800,
        f"ptdata/{SHOT}.BCI": 700,
        f"ptdata/{SHOT}.ECE": 600,
        f"mdsplus/codes/efit02er_{SHOT}.tree": 1,
        f"mdsplus/codes/efit02er_{SHOT}.characteristics": 1,
        f"mdsplus/codes/efit02er_{SHOT}.datafile": 5,
        f"mdsplus/shots/zipfit01_{SHOT}.tree": 1,
        f"mdsplus/shots/zipfit01_{SHOT}.characteristics": 1,
        f"mdsplus/shots/zipfit01_{SHOT}.datafile": 5,
        f"mdsplus/usershots/zipfit01_{SHOT}.tree": 1,
        f"mdsplus/usershots/zipfit01_{SHOT}.characteristics": 1,
        f"mdsplus/shots/efit01_{SHOT}.tree": 1,
    }
    pt = {name: ".MPI" for name in F.POINTS.values()}
    pt[F.POINTS[0]] = ".MPA"
    pt[F.IP_POINT.upper()] = (
        "BCI"  # index keys and values in either case, with or without a dot
    )
    todo, missing = M.wanted(files, {"pointname_ext": pt}, SHOT)
    assert missing == []
    assert {f"ptdata/{SHOT}.MPI", f"ptdata/{SHOT}.MPA", f"ptdata/{SHOT}.BCI"} <= set(
        todo
    )
    assert (
        f"ptdata/{SHOT}.ECE" not in todo
        and f"mdsplus/shots/efit01_{SHOT}.tree" not in todo
    )
    # a tree from codes/ lands where MDSplus looks; zipfit01 from shots/, not usershots/
    assert (
        todo[f"mdsplus/codes/efit02er_{SHOT}.datafile"]
        == f"mdsplus/shots/efit02er_{SHOT}.datafile"
    )
    assert f"mdsplus/usershots/zipfit01_{SHOT}.tree" not in todo
    assert (
        todo[f"mdsplus/shots/zipfit01_{SHOT}.tree"]
        == f"mdsplus/shots/zipfit01_{SHOT}.tree"
    )
    _, missing = M.wanted(files, {"pointname_ext": {}}, SHOT)
    assert set(missing) == {*F.POINTS.values(), F.IP_POINT}


# ------------------------------------------------------------------ clean and redo


def _label(store: Path, shot: int, **info):
    store.mkdir(parents=True, exist_ok=True)
    (store / f"{shot}.json").write_text(json.dumps({"shot": shot, **info}))


def test_clean_frees_final_shots_and_keeps_retryable_ones(tmp_path, monkeypatch):
    M._set_root(tmp_path / "root")
    corpus, store = tmp_path / "root/corpus", tmp_path / "store"
    corpus.mkdir(parents=True)
    for s in range(1, 6):
        F.corpus_path(corpus, s).write_bytes(b"x" * 1000)
    _label(store, 1, status="ok")
    _label(store, 2, status="no window: no Ip")
    _label(store, 3, status="no TokEye mask")
    _label(store, 4, status="ok", model_error="boom")
    stage = tmp_path / "root/stage"
    (stage / "1" / "ptdata").mkdir(parents=True)
    (stage / "5" / "ptdata").mkdir(parents=True)  # extracted: its corpus file is there
    (stage / "7" / "ptdata").mkdir(parents=True)  # staged, not extracted yet
    got = M.clean(store)
    assert got["deleted"] == 2 and got["kept_for_redo"] == 2 and got["waiting"] == 1
    assert [p.name for p in sorted(corpus.iterdir())] == [
        "3_processed.h5",
        "4_processed.h5",
        "5_processed.h5",
    ]
    assert sorted(p.name for p in stage.iterdir()) == ["7"]
    assert M.retryable({"status": "read error: x"}) and not M.retryable(
        {"status": "ok"}
    )


# ------------------------------------------------------------------ publish text


def test_the_redesign_publish_text_names_its_inputs_and_counts_equilibrium_sources():
    label = _load("mode_fdp_label")
    ready = [
        (1, make_shot(shot=1, eq_sources=["efit02er", "efit02"])),
        (2, make_shot(shot=2, eq_sources=["efit02er"])),
    ]
    source, _ = label.labeller_text("redesign", ready, ["efit02er"], ["abc"])
    assert "ermap-magrot" in source["labels"] and "no ECE" in source["inputs"]
    assert source["eq_sources_shots"] == {"efit02er": 2, "efit02": 1}
    assert source["status"].startswith("proposed, not checked")
    port, _ = label.labeller_text("port-v6", ready, ["efit02er"], ["abc"])
    assert port["labels"].startswith("omnimode port-v6")
