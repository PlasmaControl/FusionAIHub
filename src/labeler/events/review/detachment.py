"""Detachment rows and lazy camera previews on a shared discharge timeline."""

from __future__ import annotations

import math
import os
from pathlib import Path

import h5py
import numpy as np

from ...config import sha256_of
from .. import panels
from . import geometry, panel_rows, video
from .rows import Grid


def plasma_window(shot, paths):
    """Prefer catalog boundaries; otherwise use the producer's cached Ip window."""
    return panels.detachment.plasma_window(shot, paths) or geometry.current_window(
        shot, paths
    )


def context_sources(shot, paths):
    """Exact local inputs for safe resume when the producer writes more outputs."""
    root = Path(
        os.environ.get(
            "LABELER_DETACHMENT_INDICATORS", str(paths.root / "indicators/detachment")
        )
    )
    indicator = root / f"{int(shot)}.npz"
    if not indicator.is_file():
        indicator = root / f"{int(shot)}.csv"
    sources = {"geometry": geometry.source_path(shot, paths), "indicators": indicator}
    fingerprints = {
        key: {"path": str(path), "sha256": sha256_of(path) if path.is_file() else None}
        for key, path in sources.items()
    }
    if panels.detachment.plasma_window(shot, paths) is None:
        window = geometry.current_window(shot, paths)
        fingerprints["plasma_window"] = {
            **fingerprints["geometry"],
            "min_abs_ip_a": geometry.MIN_PLASMA_IP_A,
            "window_ms": list(window) if window else None,
        }
    return fingerprints


def build(event, shot, paths):
    window = plasma_window(shot, paths)
    built = panels.build(event, shot, paths=paths, t_range=window)
    clocks = []
    corpus = paths.corpus_file(shot)
    if corpus.is_file():
        with h5py.File(corpus, "r") as source:
            for camera in video.CAMERAS:
                if camera not in source:
                    continue
                try:
                    times, _, _ = video._layout(source[camera])
                    indices = video.frame_indices(times)
                except ValueError:
                    continue
                if len(indices):
                    clock = times[indices] * 1000
                    if window:
                        clock = clock[(clock >= window[0]) & (clock <= window[1])]
                    if len(clock):
                        clocks.append(clock)
    xs = [np.asarray(p.x, dtype=float) for p in built if len(p.x)]
    if window:
        # Exact window boundaries; do not extend to post-plasma diagnostic tails.
        base = panel_rows._grid(xs + clocks) if xs or clocks else Grid(0, 50, 1)
        n = max(1, math.ceil((window[1] - window[0]) / base.dt_ms))
        grid = Grid(window[0], (window[1] - window[0]) / n, n)
    elif xs or clocks:
        grid = panel_rows._grid(xs + clocks)
    else:
        # A missing-camera shot can still be labelled on a blank 10 s timeline.
        grid = Grid(0.0, 50.0, 200)
    rows = [
        panel_rows._trace(f"p{i}", p, np.asarray(p.x), grid)
        for i, p in enumerate(built)
        if len(p.x)
    ]
    return (
        grid,
        rows,
        {
            "params": {
                "context_only": True,
                "plasma_window_ms": window,
                "plasma_window_source": (
                    "catalog"
                    if panels.detachment.plasma_window(shot, paths)
                    else "cached |Ip| >= 300 kA"
                    if window
                    else "unavailable"
                ),
                "camera_max_fps": video.MAX_FPS,
                "detachment_geometry": geometry.load(shot, paths, window),
                "context_sources": context_sources(shot, paths),
                "panel_metadata": {
                    f"p{i}": p.metadata for i, p in enumerate(built) if p.metadata
                },
            },
            "video_corpus": corpus,
        },
    )
