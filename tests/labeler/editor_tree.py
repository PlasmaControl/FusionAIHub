"""Synthetic shots for the review editors' tests: corpus files, the cache, features."""

from __future__ import annotations

import h5py
import numpy as np

from labeler.config import Paths
from labeler.events import raw
from labeler.events.verify import NoDataError
from labeler.features.store import FeatureArray, write_features

RNG = np.random.default_rng(7)


def paths(tmp_path) -> Paths:
    return Paths(
        root=tmp_path / "root",
        corpus=tmp_path / "corpus",
        text_root=tmp_path / "text",
        logs_jsonl=tmp_path / "logs.jsonl",
        label_tables=tmp_path / "events",
        raw_cache=tmp_path / "raw",
    )


def write(path, groups: dict) -> None:
    """A corpus-format file: each group's `xdata` in seconds, `ydata` `(C, T)`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "a") as f:
        for name, (t_ms, y) in groups.items():
            g = f.create_group(name)
            g.create_dataset("xdata", data=np.asarray(t_ms, "float64") / 1000.0)
            g.create_dataset("ydata", data=np.atleast_2d(np.asarray(y, "float32")))


def times(t0_ms: float, t1_ms: float, rate_hz: float) -> np.ndarray:
    return t0_ms + np.arange(round((t1_ms - t0_ms) * rate_hz / 1000)) * 1000 / rate_hz


def noise(n_channels: int, t_ms, scale: float = 1.0) -> np.ndarray:
    return scale * RNG.standard_normal((n_channels, len(t_ms)))


def features(p: Paths, shot: int, t_ms, betan) -> None:
    arrays = {
        "betan": FeatureArray(
            x=np.asarray(t_ms) / 1000.0, y=np.atleast_2d(betan), attrs={}
        )
    }
    write_features(p.features_file(shot), shot, arrays, {})


def no_fetch(monkeypatch) -> list:
    """Every live fetch fails, as it does off the login node; returns the tries."""
    tried = []

    def refuse(shot, exprs, **kwargs):
        tried.append((shot, list(exprs)))
        raise NoDataError("no fdp here")

    monkeypatch.setattr(raw, "fdp_signal", refuse)
    monkeypatch.setattr(raw, "RETRY_DELAY_S", 0.0)
    return tried
