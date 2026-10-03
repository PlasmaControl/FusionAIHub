"""Read-only source selection for the TokEye interpreter figure.

CONFINEMENT_RUN_DIR defaults to runs/labeler/confinement/v1 in the checkout
containing LABELER_LABEL_TABLES. Both curated and fallback files are required
when needed; a missing file never silently substitutes the H-mode frame model.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Paths, sha256_of
from ..events.catalog.states import ABSENT, NOT_OBSERVABLE, PRESENT
from ..events.verify import corpus_path
from ..labels.store import read_label
from . import label_figure as lf
from . import mode_tags as mt


def dalpha_file(paths: Paths) -> Path:
    return paths.root / "suggestions/dalpha_lh/v1/confinement_suggest_dalpha_lh_v1.csv"


def confinement_track(paths: Paths, shot: int) -> lf.Track:
    """Saved four-class review, curated regime table, then D-alpha H-mode."""
    spec = next(s for s in lf.TRACKS if s.key == "confinement")
    reviewed = paths.label_tables / "confinement/review/labels.csv"
    rows = lf.read_rows(reviewed).get(shot)
    if rows:
        source = lf.Source(
            lf.SILVER, "curated four-class confinement review", lambda p: reviewed
        )
        return lf.Track(replace(spec, title="regime"), source, reviewed, rows)
    default = paths.label_tables.parents[1] / "runs/labeler/confinement/v1"
    run = Path(os.environ.get("CONFINEMENT_RUN_DIR", str(default)))
    curated = run / "merged_intervals.csv"
    if not curated.is_file():
        raise FileNotFoundError(
            f"CONFINEMENT_RUN_DIR={run}: required curated {curated.name} missing"
        )
    rows = lf.read_rows(curated, regimes=True).get(shot)
    if rows:
        source = lf.Source(
            lf.LEGACY, "Gill's and Butt's curated regime intervals", lambda p: curated
        )
        return lf.Track(replace(spec, title="regime"), source, curated, rows)
    fallback = dalpha_file(paths)
    if not fallback.is_file():
        raise FileNotFoundError(f"required D-alpha L-H fallback missing: {fallback}")
    rows = lf.read_rows(fallback).get(shot)
    if not rows:
        raise ValueError(f"shot {shot}: no curated regime or D-alpha L-H coverage")
    source = lf.Source(
        lf.GENERATED, "D-alpha L-H transition detector (dalpha_lh)", lambda p: fallback
    )
    return lf.Track(replace(spec, title="H-mode"), source, fallback, rows)


def stored_ae_track(paths: Paths, shot: int) -> lf.Track | None:
    """Paper ae-ours probabilities, if stored and valid; fixed p>=0.5.

    The adapter's bins are causal (t-25ms, t]; convert them to interval rows,
    preserving invalid bins as not observable. Never run a model or fetch.
    """
    file = paths.labels_file(shot)
    if not file.is_file():
        return None
    slug = "d3d_ae_activity_seldnet"
    try:
        active = read_label(file, slug, "ae_active")
        valid = read_label(file, slug, "ae_active_valid")
    except KeyError:
        return None
    if not np.array_equal(active.x, valid.x):
        raise ValueError(f"{file}: AE activity and validity time grids differ")
    observed = valid.y[0].astype(bool) & np.isfinite(active.y[0])
    if not observed.any():
        return None
    step = float(active.attrs.get("time_step_ms", 25.0))
    rows = []
    for t, p, ok in zip(active.x * 1000, active.y[0], observed, strict=True):
        state = (PRESENT if p >= 0.5 else ABSENT) if ok else NOT_OBSERVABLE
        a, b = float(t - step), float(t)
        if rows and rows[-1].category == state and abs(rows[-1].t_end - a) < 1e-6:
            rows[-1] = replace(rows[-1], t_end=b)
        else:
            rows.append(lf.Row(a, b, state))
    spec = next(s for s in lf.TRACKS if s.key == mt.AE)
    source = lf.Source(
        lf.GENERATED,
        "ae-ours (SELDnet-style), stored CO2 activity, p>=0.5",
        lambda p: file,
    )
    return lf.Track(spec, source, file, tuple(rows))


def reviewed_or_stored_ae(paths: Paths, shot: int) -> lf.Track | None:
    """Paper predictions first, else current expert source; no fallback lookup."""
    stored = stored_ae_track(paths, shot)
    if stored is not None:
        return stored
    spec = next(s for s in lf.TRACKS if s.key == mt.AE)
    source = spec.sources[0]
    path = source.locate(paths)
    rows = lf.read_rows(path).get(shot)
    if rows:
        return lf.Track(spec, source, path, rows)
    return None


def crash_times(paths: Paths, shot: int, spans, source: Path | None = None):
    """Detector crash times within PRESENT label spans, or a replacement CSV.

    A replacement must have shot,t_ms columns. Default: the existing
    ece_sawtooth crash detector's ECE/SXR union, read from local signals only.
    These are detector point times even when the interval label is expert.
    """
    if source is not None:
        frame = pd.read_csv(source)
        times = frame.loc[frame["shot"] == shot, "t_ms"].to_numpy(float)
        record = {"source": str(source), "sha256": sha256_of(source)}
    else:
        from ..events import heuristics
        from ..events import spans as detectors

        found, _, info = detectors._sawtooth_crashes(shot, paths)
        events = heuristics.sawtooth_events_v3(found)
        times = np.array([e.t0_s * 1000 for e in events])
        record = {
            "source": "ece_sawtooth crash detector, ECE/SXR union",
            "implementation": "labeler.events.heuristics.sawtooth_events_v3",
            "diagnostics": info,
            "signals": [
                {"path": str(p), "sha256": sha256_of(p)}
                for p in (
                    paths.corpus_file(shot),
                    corpus_path(shot, corpus=paths.raw_cache),
                )
                if p.is_file()
            ],
        }
    times = np.unique(times[np.isfinite(times)])
    kept = times[mt.present_columns(times, spans)]
    return kept, {
        **record,
        "all_times_ms": times.tolist(),
        "present_times_ms": kept.tolist(),
    }
