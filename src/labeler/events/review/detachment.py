"""Detachment rows and lazy camera previews on a shared discharge timeline."""

from __future__ import annotations

import h5py
import numpy as np

from .. import panels
from . import panel_rows, video
from .rows import Grid


def build(event, shot, paths):
    built = panels.build(event, shot, paths=paths)
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
                    clocks.append(times[indices] * 1000)
    xs = [np.asarray(p.x, dtype=float) for p in built if len(p.x)]
    if xs or clocks:
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
                "camera_max_fps": video.MAX_FPS,
                "panel_metadata": {
                    f"p{i}": p.metadata for i, p in enumerate(built) if p.metadata
                },
            },
            "video_corpus": corpus,
        },
    )
