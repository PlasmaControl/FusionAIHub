"""``scripts/labeler/silver_tm_saw_rwm.py``'s tm-features stage reads a shot's Mirnov
from the corpus or, for a shot before the corpus, from the raw cache."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest

from labeler.events import raw
from labeler.tearing import magfeatures

from . import editor_tree as tree

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/labeler/silver_tm_saw_rwm.py"
spec = importlib.util.spec_from_file_location("silver_tm_saw_rwm", SCRIPT)
silver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(silver)

SHOTS = [101, 102, 103, 104, 105]


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Silver shots 101-105 on a temporary disk: 101 in the corpus with a Mirnov group,
    102 in the corpus without one, 103 in the raw cache with Mirnov, 104 in the raw
    cache without it, 105 nowhere."""
    p = tree.paths(tmp_path)
    t = tree.times(0.0, 4000.0, 100_000.0)  # 4 s of Mirnov at 100 kHz
    ip_t = np.arange(0.0, 4000.0, 1.0)
    ip = 1e6 * np.clip(np.minimum(ip_t, 4000.0 - ip_t) / 300.0, 0.0, 1.0)
    noise = tree.noise(len(raw.MIRNOV_ROWS), t)
    mirnov = {"mirnov": (t, tree.noise(raw.MIRNOV_WIDTH, t))}
    tree.write(p.corpus_file(101), mirnov | {"ip": (ip_t, ip)})
    tree.write(p.corpus_file(102), {"ip": (ip_t, ip)})
    tree.write(raw.cache_path(103, paths=p), {"ip": (ip_t, ip)})
    raw.write_group(
        raw.cache_path(103, paths=p),
        "mirnov",
        t,
        noise,
        rows=raw.MIRNOV_ROWS,
        n_rows=raw.MIRNOV_WIDTH,
    )
    tree.write(raw.cache_path(104, paths=p), {"ip": (ip_t, ip)})
    run = tmp_path / "run"
    # as in the SLURM jobs: a group neither tier holds is an error, never a fetch
    monkeypatch.setenv(raw.NO_FETCH_ENV, "1")
    tried = tree.no_fetch(monkeypatch)
    monkeypatch.setattr(silver, "PATHS", p)
    monkeypatch.setattr(silver, "RUN", run)
    monkeypatch.setattr(silver, "TM", tmp_path / "round4_tm")
    monkeypatch.setattr(silver, "silver_shots", lambda: list(SHOTS))
    (tmp_path / "round4_tm" / "magfeatures").mkdir(parents=True)
    (run / "tm" / "magfeatures").mkdir(parents=True)
    yield p
    assert tried == [], "a test must never fetch"


def test_the_plan_takes_a_corpus_file_or_a_cached_mirnov_record(world):
    # 102 has a corpus file without the group: planned, so its reason is reported per
    # shot; 104 (cache, no Mirnov) and 105 (nowhere) are not.
    assert silver.tm_feature_plan() == [101, 102, 103]


def test_the_plan_leaves_out_shots_that_have_features_already(world, tmp_path):
    # round four's features, and this stage's, are both kept as they are
    np.savez(tmp_path / "round4_tm" / "magfeatures" / "101.npz", features=np.zeros(1))
    np.savez(silver.RUN / "tm" / "magfeatures" / "102.npz", features=np.zeros(1))
    assert silver.tm_feature_plan() == [103]


def test_a_cached_mirnov_is_featurised_over_the_shots_ip_window(world):
    shot, status, _ = silver.tm_feature_one(103)
    assert (shot, status) == (103, "ok")
    with np.load(silver.RUN / "tm" / "magfeatures" / "103.npz") as z:
        assert str(z["window_from"]) == "ip"
        w0, w1 = z["window_ms"]
        assert 0.0 < w0 < w1 < 4000.0
        assert z["features"].shape == (len(z["centres_ms"]), len(z["names"]))
        assert list(z["names"]) == list(magfeatures.FEATURE_NAMES)
        assert np.isfinite(z["features"]).any()


def test_a_corpus_shot_is_read_from_the_corpus(world):
    assert silver.tm_feature_one(101)[1] == "ok"
    assert raw.raw_signal(101, "mirnov", channels=[15], paths=world).attrs["tier"] == (
        "corpus"
    )


def test_a_shot_without_mirnov_gets_a_missing_file_naming_why(world):
    for shot in (102, 104):
        assert silver.tm_feature_one(shot)[1] == "missing"
        reason = (silver.RUN / "tm" / "magfeatures" / f"{shot}.missing.json").read_text()
        assert "mirnov" in reason
    assert not (silver.RUN / "tm" / "magfeatures" / "104.npz").exists()


def test_the_cached_group_is_a_corpus_shaped_sparse_one(world):
    with h5py.File(raw.cache_path(103, paths=world), "r") as f:
        assert f["mirnov/ydata"].shape[0] == raw.MIRNOV_WIDTH
        assert np.isnan(f["mirnov/ydata"][1, :10]).all()
