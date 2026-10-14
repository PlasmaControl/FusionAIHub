"""The q = 1 surface gate on synthetic equilibria."""

from itertools import pairwise

import numpy as np
import pytest

from labeler.sawtooth import qgate


def efit(t0, t1, q, step=20.0):
    t = np.arange(t0, t1 + step / 2, step)
    return t, np.asarray(q(t) if callable(q) else np.full(len(t), q), float)


def test_efit02_high_q_is_no_q1_and_never_extrapolated():
    efit02 = efit(300, 1000, lambda t: np.where(t < 600, 1.5, 0.9))
    gate = qgate.sample([200, 400, 800, 999.5, 1000, 1100], efit02)
    assert list(gate.state) == [
        qgate.UNKNOWN,
        qgate.NO_Q1,
        qgate.Q1_POSSIBLE,
        qgate.Q1_POSSIBLE,
        qgate.UNKNOWN,  # the cell [1000, 1001) lies past the last slice
        qgate.UNKNOWN,
    ]
    assert list(gate.source) == ["", "efit02", "efit02", "efit02", "", ""]
    spans = qgate.gate_spans(efit02)
    assert spans[0][:3] == (300.0, spans[0][1], qgate.NO_Q1)
    assert 580 <= spans[0][1] <= 600  # the linear crossing of 1.2
    assert spans[-1][1] == 1000.0


def test_short_high_q_excursion_is_not_sustained():
    # q >= 1.2 for about 66 ms (three slices plus the crossings).
    efit02 = efit(300, 1000, lambda t: np.where((t >= 500) & (t <= 540), 1.6, 1.0))
    assert not qgate.sample(np.arange(300, 1000), efit02).no_q1.any()
    assert qgate.sample([700], efit02, sustain_ms=20).state[0] == qgate.Q1_POSSIBLE
    assert qgate.sample([520], efit02, sustain_ms=20).state[0] == qgate.NO_Q1


def test_efit01_decides_only_where_efit02_is_silent_with_its_own_cut():
    efit02 = efit(300, 1000, 1.1)
    efit01 = efit(100, 1200, lambda t: np.where(t < 1100, 2.5, 1.6))
    gate = qgate.sample([150, 250, 500, 1050, 1150], efit02, efit01)
    assert list(gate.source) == ["efit01", "efit01", "efit02", "efit01", "efit01"]
    assert list(gate.state) == [
        qgate.NO_Q1,  # EFIT01 2.5 >= 2.0 from 100 to 300 ms (200 ms)
        qgate.NO_Q1,
        qgate.Q1_POSSIBLE,  # EFIT02 1.1 overrides EFIT01's 2.5
        qgate.Q1_POSSIBLE,  # 2.5 from 1000 to ~1090: under 100 ms
        qgate.Q1_POSSIBLE,  # EFIT01 1.6 is below its 2.0 cut
    ]
    # Without EFIT02 the EFIT01 cut alone decides the whole base.
    alone = qgate.sample([500, 1150], None, efit01)
    assert list(alone.state) == [qgate.NO_Q1, qgate.Q1_POSSIBLE]


def test_gaps_and_invalid_slices_are_unknown():
    t, q = efit(300, 1000, 1.5)
    keep = (t < 500) | (t > 700)  # a 240 ms hole between 480 and 720
    q = q.copy()
    q[t == 900] = np.nan  # a failed slice is dropped, its 40 ms gap bridged
    gate = qgate.sample([400, 600, 800, 900], (t[keep], q[keep]))
    assert list(gate.state) == [qgate.NO_Q1, qgate.UNKNOWN, qgate.NO_Q1, qgate.NO_Q1]
    assert qgate.sample([400], None, None).state[0] == qgate.UNKNOWN
    assert qgate.gate_spans(None, None) == []


def test_gate_does_not_depend_on_the_sample_grid():
    efit02 = efit(300, 3000, lambda t: 1.0 + 0.4 * np.sin(t / 150.0))
    fine = qgate.sample(np.arange(3000, 30000) / 10.0, efit02)
    coarse = qgate.sample(np.arange(300, 3000, 1.0), efit02)
    assert np.array_equal(fine.state[::10], coarse.state)
    spans = qgate.gate_spans(efit02)
    assert spans[0][0] == 300.0 and spans[-1][1] == 3000.0
    assert all(a[1] == b[0] for a, b in pairwise(spans))


def test_tree_order_efit02_then_efit02er_then_efit01():
    efit02 = efit(1000, 2000, 1.1)  # q = 1 possible, decides 1000-2000
    efit02er = efit(500, 3000, 1.5)  # MSE-constrained: 1.5 >= 1.2 is no q = 1
    efit01 = efit(100, 4000, lambda t: np.where(t < 3500, 1.8, 2.5))
    gate = qgate.sample(
        [300, 700, 1500, 2500, 3200, 3700],
        efit02,
        efit01,
        efit02er=efit02er,
    )
    assert list(gate.source) == [
        "efit01",
        "efit02er",
        "efit02",
        "efit02er",
        "efit01",
        "efit01",
    ]
    assert list(gate.state) == [
        qgate.Q1_POSSIBLE,  # EFIT01 1.8 is below its 2.0 cut
        qgate.NO_Q1,  # EFIT02ER's 1.2 cut, not EFIT01's 2.0
        qgate.Q1_POSSIBLE,  # EFIT02 1.1 overrides EFIT02ER's 1.5
        qgate.NO_Q1,
        qgate.Q1_POSSIBLE,
        qgate.NO_Q1,  # EFIT01 2.5 from 3500 to 4000 ms
    ]
    spans = qgate.gate_spans(efit02, efit01, efit02er=efit02er)
    assert [s[3] for s in spans if s[0] >= 500 and s[1] <= 3000] == [
        "efit02er",
        "efit02",
        "efit02er",
    ]


def test_readers_return_ms_and_record_every_file(tmp_path):
    import hashlib

    from labeler.config import Paths
    from labeler.features.store import FeatureArray, write_features

    paths = Paths(root=tmp_path, corpus=tmp_path / "corpus")
    (tmp_path / "mhd" / "efit02").mkdir(parents=True)
    (tmp_path / "mhd" / "efit_fallback").mkdir(parents=True)
    t02, q02 = efit(300, 1000, 1.5)
    np.savez(tmp_path / "mhd/efit02/1.npz", atime=t02, a_qmin=q02)
    t_er, q_er = efit(320, 900, 1.3)
    np.savez(
        tmp_path / "mhd/efit_fallback/2.npz",
        atime=t_er,
        a_qmin=q_er,
        tree=np.array("efit02er"),
    )
    seconds = np.arange(0.1, 1.2001, 0.02)  # EFIT01 in a store: seconds
    write_features(
        paths.features_file(1),
        1,
        {
            "qmin": FeatureArray(
                seconds, np.full((1, len(seconds)), 2.5), {"resolver": "fdp"}
            )
        },
        {},
    )
    t01, q01 = qgate.read_efit01(1, paths)
    assert t01[0] == 100.0 and abs(t01[-1] - 1200.0) < 1e-6
    assert np.allclose(q01, 2.5)
    assert np.array_equal(qgate.read_efit02(1, paths)[0], t02)
    assert qgate.read_fallback(2, paths)[0] == "efit02er"
    assert qgate.read_efit02(2, paths) is None

    gate = qgate.q1_gate(1, paths, [200, 500, 1100])
    assert list(gate.source) == ["efit01", "efit02", "efit01"]
    assert [f["tree"] for f in gate.files] == ["efit02", "efit01"]
    first = gate.files[0]
    assert first["path"] == "mhd/efit02/1.npz"
    raw = (tmp_path / "mhd/efit02/1.npz").read_bytes()
    assert first["sha256"] == hashlib.sha256(raw).hexdigest()
    assert first["series_sha256"] == qgate.series_sha256((t02, q02))
    spans, files = qgate.shot_gate(2, paths)
    assert {s[3] for s in spans} == {"efit02er"}
    assert [f["tree"] for f in files] == ["efit02er"]
    assert qgate.shot_gate(3, paths) == ([], ())

    (tmp_path / "mhd/efit02/3.npz").write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="3.npz"):
        qgate.shot_spans(3, paths)


def _parked_paths(tmp_path):
    from labeler.config import Paths

    paths = Paths(root=tmp_path, corpus=tmp_path / "corpus")
    (tmp_path / "mhd" / "efit02").mkdir(parents=True, exist_ok=True)
    (tmp_path / "mhd" / "efit_fallback").mkdir(parents=True, exist_ok=True)
    return paths


def _store_efit01(paths, shot, q):
    from labeler.features.store import FeatureArray, write_features

    seconds = np.arange(0.1, 1.2001, 0.02)
    array = FeatureArray(seconds, np.full((1, len(seconds)), q), {"resolver": "fdp"})
    write_features(paths.features_file(shot), shot, {"qmin": array}, {})


def test_a_one_slice_efit02_covers_nothing_so_efit01_decides(tmp_path):
    paths = _parked_paths(tmp_path)
    # one EFIT02 slice reading q = 1 possible: it covers no cell, so it overrides nothing
    np.savez(tmp_path / "mhd/efit02/4.npz", atime=[500.0], a_qmin=[0.9])
    t01, q01 = efit(100, 1200, 2.5)
    fallback = tmp_path / "mhd/efit_fallback/4.npz"
    np.savez(fallback, atime=t01, a_qmin=q01, tree=np.array("efit01"))
    gate = qgate.q1_gate(4, paths, [300, 500, 1000])
    assert list(gate.source) == ["efit01"] * 3 and gate.no_q1.all()
    assert [f["tree"] for f in gate.files] == ["efit02", "efit01"]
    # the same with EFIT01 from the stores when no fallback file holds it
    fallback.unlink()
    _store_efit01(paths, 4, 2.5)
    gate = qgate.q1_gate(4, paths, [300, 500, 1000])
    assert list(gate.source) == ["efit01"] * 3 and gate.no_q1.all()
    assert gate.files[1]["path"] == str(paths.features_file(4).relative_to(tmp_path))


def test_an_efit01_fallback_file_is_preferred_over_the_stores(tmp_path):
    paths = _parked_paths(tmp_path)
    t01, q01 = efit(100, 1200, 2.5)
    fallback = tmp_path / "mhd/efit_fallback/5.npz"
    np.savez(fallback, atime=t01, a_qmin=q01, tree=np.array("efit01"))
    _store_efit01(paths, 5, 1.0)  # the store would say q = 1 possible
    gate = qgate.q1_gate(5, paths, [300, 1000])
    assert gate.no_q1.all() and list(gate.source) == ["efit01", "efit01"]
    assert [f["path"] for f in gate.files] == ["mhd/efit_fallback/5.npz"]
    # an EFIT02ER fallback file leaves EFIT01 to the stores, below it in the order
    t_er, q_er = efit(500, 800, 1.0)
    np.savez(fallback, atime=t_er, a_qmin=q_er, tree=np.array("efit02er"))
    gate = qgate.q1_gate(5, paths, [300, 600])
    assert list(gate.source) == ["efit01", "efit02er"]
    assert list(gate.state) == [qgate.Q1_POSSIBLE, qgate.Q1_POSSIBLE]
    assert [f["tree"] for f in gate.files] == ["efit02er", "efit01"]


def test_shot_spans_hashes_no_file(tmp_path, monkeypatch):
    paths = _parked_paths(tmp_path)
    np.savez(
        tmp_path / "mhd/efit02/6.npz",
        atime=efit(300, 1000, 1.5)[0],
        a_qmin=efit(300, 1000, 1.5)[1],
    )
    _store_efit01(paths, 6, 2.5)
    hashed = []
    real = qgate.file_record
    monkeypatch.setattr(
        qgate, "file_record", lambda *a, **k: hashed.append(a[0]) or real(*a, **k)
    )
    spans = qgate.shot_spans(6, paths)
    assert hashed == [] and {s[3] for s in spans} == {"efit02", "efit01"}
    assert qgate.shot_gate(6, paths)[0] == spans and len(hashed) == 2


def test_only_a_store_without_q_min_means_no_efit01(tmp_path, monkeypatch):
    from labeler.events import equilibrium
    from labeler.events.verify import NoDataError

    paths = _parked_paths(tmp_path)

    def missing(*args, **kwargs):
        raise NoDataError("no qmin")

    def broken(*args, **kwargs):
        raise ValueError("a broken store")

    monkeypatch.setattr(equilibrium, "signal", missing)
    assert qgate.read_efit01(7, paths) is None and qgate.shot_spans(7, paths) == []
    monkeypatch.setattr(equilibrium, "signal", broken)
    with pytest.raises(ValueError, match="broken store"):
        qgate.shot_spans(7, paths)
