"""A synthetic port-v6 shot for the `labeler.modes` tests: tracks, frames and an info record.

Tracks over a 0-1000 ms window (frames every 4 ms, n as written):

  1  n=1  100-300  labelled 2/1 (P(resonant) 0.97)
  2  n=2  100-300  4/2 at 2f of track 1: `harm` true, labelled 4/2 by the model (a harmonic)
  3  n=2  400-500  an independent 4/2: `harm` false, labelled 4/2 (0.93)
  4  n=2  600-700  labelled 3/2 (0.95)
  5  n=4  150-250  outside |n| 1-3 (labelled 5/4 by a model that emits it)
  6  n=1  500-700  a fixed-frequency line
  7  n=1  800-860  an abstention (uncertain, 90% set 2/1|3/1)
  8  n=1  900-960  labelled 1/1 (0.99)
  9  n=3  700-780  labelled 4/3 (0.91)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from labeler.mhd import pipeline

SHOT = 100000  # synthetic: no test reads a real shot
WINDOW = (0.0, 1000.0)

SPEC = [
    (1, 1, (100, 300), {"status": "m/n", "label": "2/1", "p_res": 0.97}),
    (2, 2, (100, 300), {"status": "m/n", "label": "4/2", "p_res": 0.9, "harm": True}),
    (3, 2, (400, 500), {"status": "m/n", "label": "4/2", "p_res": 0.93}),
    (4, 2, (600, 700), {"status": "m/n", "label": "3/2", "p_res": 0.95}),
    (5, 4, (150, 250), {"status": "m/n", "label": "5/4", "p_res": 0.95}),
    (6, 1, (500, 700), {"line": True, "why": "line", "modelled": False}),
    (7, 1, (800, 860), {"status": "uncertain", "set90": "2/1|3/1"}),
    (8, 1, (900, 960), {"status": "m/n", "label": "1/1", "p_res": 0.99}),
    (9, 3, (700, 780), {"status": "m/n", "label": "4/3", "p_res": 0.91}),
]


def make_shot(
    *, shot=SHOT, status="ok", eq_tree="efit02er", gaps=(), spec=SPEC, **info_extra
):
    T, frames = [], []
    for fid, n, (a, b), kw in spec:
        row = {c: np.nan for c in pipeline.TRACK_COLUMNS}
        row |= {"shot": shot, "family_id": fid, "n": n, "t0": a, "t1": b, "coh": 0.9}
        row |= {"line": False, "harm": False, "modelled": True, "why": ""}
        row |= {"status": "", "label": "", "set90": "", "p_2/1": 0.6, "p_3/1": 0.3}
        T.append(row | kw)
        frames += [(shot, fid, t, 10.0, n) for t in np.arange(a, b + 1, 4.0)]
    info = {
        "window": list(WINDOW),
        "window_source": "test",
        "status": status,
        "mirnov_ms": list(WINDOW),
        "mirnov_gaps": [list(g) for g in gaps],
        "inputs": {"tree": eq_tree, "rotation": True},
        "eq_t": list(np.arange(0, 1000, 20.0)) if eq_tree else [],
        "freeze": "f1",
    } | info_extra
    frame = pd.DataFrame(frames, columns=["shot", "family_id", "t_ms", "f0", "n"])
    tracks = pd.DataFrame(T)[pipeline.TRACK_COLUMNS]
    return pipeline.Shot(shot, info, tracks, frame)


def write_shot(directory, got) -> None:
    """The three port-v6 files of the shot, as `tracks.read_shot` reads them."""
    base = Path(directory) / str(got.shot)
    Path(directory).mkdir(parents=True, exist_ok=True)
    base.with_suffix(".json").write_text(json.dumps(got.info))
    got.tracks.to_parquet(f"{base}.tracks.parquet", index=False)
    got.frames.to_parquet(f"{base}.frames.parquet", index=False)
