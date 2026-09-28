"""The D-alpha photodiode the ELM review draws: fetched once, then read from the cache."""

from __future__ import annotations

import numpy as np

from labeler.config import Paths
from labeler.events import raw
from labeler.features.store import FeatureArray


def test_pcphd03_is_fetched_from_ptdata_and_cached(tmp_path, monkeypatch):
    paths = Paths(corpus=tmp_path / "corpus", raw_cache=tmp_path / "cache")
    calls = []

    def fake_fdp_signal(shot, exprs, *, tree, via, t_range=None, **kwargs):
        calls.append((shot, list(exprs), via))
        return FeatureArray(
            x=np.arange(0.0, 100.0, 0.05),
            y=np.ones((1, 2000), dtype="float32"),
            attrs={"units": "ms"},
        )

    monkeypatch.setattr(raw, "fdp_signal", fake_fdp_signal)
    got = raw.raw_signal(200001, "pcphd03", paths=paths)
    assert calls == [(200001, ["PCPHD03"], "ptdata")]
    assert got.attrs["tier"] == "fetch" and got.y.shape == (1, 2000)
    again = raw.raw_signal(200001, "pcphd03", t_range=(10.0, 20.0), paths=paths)
    assert len(calls) == 1 and again.attrs["tier"] == "cache"
    assert again.x[0] >= 10.0 and again.x[-1] <= 20.0
