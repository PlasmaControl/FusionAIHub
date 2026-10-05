"""Read-only source selection for the TokEye interpreter figure.

CONFINEMENT_RUN_DIR defaults to runs/labeler/confinement/v1 in the checkout
containing LABELER_LABEL_TABLES. The confinement row reads, in order, the saved
four-class review, the curated regime table, the released confine-ours roster
and, last, the D-alpha L-H table. Each file is required when the chain reaches
it; a missing file never silently substitutes the H-mode frame model.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, replace
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Paths, sha256_of
from ..events.catalog.states import ABSENT, NOT_OBSERVABLE, PRESENT, UNCERTAIN
from ..events.interval_tables import INTERVAL_COLUMNS, validate_intervals
from ..events.verify import corpus_path
from ..labels.store import read_label
from . import label_figure as lf
from . import mode_tags as mt

# Paper operating point: scripts/labeler/ae_baselines_evaluate.py, SELDnet.
AE_THRESHOLD = 0.5
NTM_THRESHOLD = 0.63
SAWTOOTH_THRESHOLD = 0.6
ELM_VETO_MS = 5.0
ECE_MATCH_MS = 3.0
SAWTOOTH_DISPLAY_MIN_MS = 0.0
HARMONIC_MIN_SUPPORT_MS = 50.0
#: A ridge pair is called consistent with a harmonic in the caption only where
#: at least this share of the columns that measure both ridges pass the ratio.
HARMONIC_MIN_PASSING_FRACTION = 0.6
#: A ridge passes where |f(n)/f(1) - n| <= this share of n.
HARMONIC_RATIO_TOLERANCE = 0.05
CAPTION_MAX_WORDS = 120
#: The four-class confinement row's title; the binary D-alpha fallback is "H-mode".
CONFINEMENT_TITLE = "confinement"
BINARY_CONFINEMENT_TITLE = "H-mode"
#: The released confine-ours roster: label tables root-relative path, its columns
#: (the five interval columns, then each segment's provenance) and its tiers.
ROSTER_TABLE = "confinement/extend_confine_ours/roster.csv"
ROSTER_COLUMNS = (
    *INTERVAL_COLUMNS,
    "predicted",
    "source",
    "tier",
    "extrapolated",
)
ROSTER_TIERS = ("model", "unreviewed")
ROSTER_WHAT = "released confine-ours roster (1-D U-Net model labels)"
#: Roster classes that are ELM-free by definition.
ELM_FREE_CLASSES = (3, 4)


def tokeye_fingerprints(
    paths, shot, group, row, inference_code, partner_row=None
) -> dict:
    """Identity of the actual model waveform and its preprocessing/inference.

    Hash the trimmed float32 samples that enter prep, with their timing and
    channel identity. Pin local transform code, runtime versions and the
    caller's inference code; presentation changes need not invalidate arrays.
    """
    from ..ae import labels, transform
    from ..events import masks, unet

    y, fs, t0, t1 = masks.read_waveform(paths.corpus_file(shot), group, row)
    waveform = {
        "group": group,
        "row": row,
        "samples": len(y),
        "dtype": str(y.dtype),
        "sample_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
        "fs_hz": fs,
        "start_s": t0,
        "end_s": t1,
    }
    if partner_row is not None:
        y2, fs2, t0_2, t1_2 = masks.read_waveform(
            paths.corpus_file(shot), group, partner_row
        )
        waveform["partner_row"] = partner_row
        waveform["partner_sample_sha256"] = hashlib.sha256(y2.tobytes()).hexdigest()
        waveform["partner_span"] = [fs2, t0_2, t1_2]
    preprocessing = {
        "code_sha256": {
            str(Path(m.__file__).relative_to(Path(__file__).parents[2])): sha256_of(
                Path(m.__file__)
            )
            for m in (masks, unet, transform, labels)
        },
        "runtime_versions": {
            name: version(name) for name in ("numpy", "scipy", "torch")
        },
        "inference_sha256": hashlib.sha256(inference_code.encode()).hexdigest(),
        "zoom_decimation": masks.ZOOM_DECIM,
        "row_lit_threshold": mt.PROB_THRESHOLD,
    }
    return {"waveform": waveform, "preprocessing": preprocessing}


def state_intervals(track: lf.Track, window: tuple[float, float]) -> list[dict]:
    """Clipped source categories with the track's own state/regime names."""
    codes = {ABSENT: "absent", **track.spec.states}
    if track.spec.key != "confinement" or track.spec.title == BINARY_CONFINEMENT_TITLE:
        codes[NOT_OBSERVABLE] = "unassessed"
    return [
        {
            "start_ms": max(r.t_start, window[0]),
            "end_ms": min(r.t_end, window[1]),
            "category": r.category,
            "state": codes.get(r.category, str(r.category)),
        }
        for r in track.rows
        if r.t_end > window[0] and r.t_start < window[1]
    ]


def has_present_time(track: lf.Track, window: tuple[float, float]) -> bool:
    """A source PRESENT interval has positive duration inside the view."""
    return any(
        r.category == PRESENT and min(r.t_end, window[1]) > max(r.t_start, window[0])
        for r in track.rows
    )


def sawtooth_display(track: lf.Track, window: tuple[float, float]):
    """Clip to the view without changing any source category or duration."""
    rows = [
        replace(r, t_start=max(r.t_start, window[0]), t_end=min(r.t_end, window[1]))
        for r in track.rows
        if r.t_end > window[0] and r.t_start < window[1]
    ]
    return replace(track, rows=tuple(rows)), []


def late_untagged_lines(mask, t_ms, f_khz, late_absent) -> dict | None:
    """Frequency bounds from every late untagged pixel in the AE input band."""
    t, f = np.asarray(t_ms), np.asarray(f_khz)
    eligible = mask & np.asarray(late_absent)[None, :]
    eligible &= (f >= mt.BANDS[mt.AE][0])[:, None]
    if not eligible.any():
        return None
    rr, cc = np.nonzero(eligible)
    return {
        "band_khz": [float(f[rr].min()), float(f[rr].max())],
        "pixels": int(eligible.sum()),
        "columns": int(eligible.any(0).sum()),
        "first_time_ms": float(t[cc].min()),
        "last_time_ms": float(t[cc].max()),
        "rule": "all coherent pixels in the AE detector band during late ABSENT "
        "times after the last PRESENT AE interval; no duration cutoff",
        "reason": "AE detector absent; time coincidence is required",
    }


def elm_category(row: lf.Row) -> int:
    """Crowd identifies the annotation lane; category retains its meaning."""
    return row.category


def dalpha_file(paths: Paths) -> Path:
    return paths.root / "suggestions/dalpha_lh/v1/confinement_suggest_dalpha_lh_v1.csv"


def roster_file(paths: Paths) -> Path:
    return paths.label_tables / ROSTER_TABLE


def read_roster(path: Path, shot: int) -> tuple[dict, ...]:
    """The released roster's segments of `shot`, in file order, validated.

    `lf.read_rows` takes only the five interval columns; the roster adds each
    segment's provenance, so it has its own reader. A zero-length segment is
    dropped, as `lf.read_rows` drops it."""
    frame = pd.read_csv(path, index_col=False)
    if tuple(frame.columns) != ROSTER_COLUMNS:
        raise ValueError(f"{path}: expected columns {ROSTER_COLUMNS}")
    core = validate_intervals(frame[list(INTERVAL_COLUMNS)])
    unknown = set(core["category"]) - set(lf.REGIMES)
    if unknown:
        raise ValueError(f"{path}: unknown confinement classes {sorted(unknown)}")
    bad = set(frame["tier"]) - set(ROSTER_TIERS)
    if bad:
        raise ValueError(f"{path}: unknown tiers {sorted(bad)}")
    segments = []
    for i in np.flatnonzero((core["shot"] == shot).to_numpy()):
        a, b = float(core["t_start"].iat[i]), float(core["t_end"].iat[i])
        if b <= a:
            continue
        category = int(core["category"].iat[i])
        segments.append(
            {
                "start_ms": a,
                "end_ms": b,
                "category": category,
                "state": lf.REGIMES[category],
                "confidence": float(core["confidence"].iat[i]),
                "predicted": int(frame["predicted"].iat[i]),
                "source": str(frame["source"].iat[i]),
                "tier": str(frame["tier"].iat[i]),
                "extrapolated": bool(frame["extrapolated"].iat[i]),
            }
        )
    return tuple(segments)


@dataclass(frozen=True)
class RosterTrack(lf.Track):
    """A confinement track read from the roster, with each segment's provenance
    (`segments`, aligned with `rows`)."""

    segments: tuple[dict, ...] = ()


def roster_segments(track: lf.Track, window: tuple[float, float]) -> list[dict]:
    """The roster segments of a roster track that overlap `window`."""
    return [
        s
        for s in getattr(track, "segments", ())
        if s["end_ms"] > window[0] and s["start_ms"] < window[1]
    ]


def confinement_row_source(track: lf.Track, window: tuple[float, float]) -> str | None:
    """The roster row's source text from its own tiers; None for other rows."""
    if not hasattr(track, "segments"):
        return None
    tiers = {s["tier"] for s in roster_segments(track, window)}
    return "model (unreviewed)" if "unreviewed" in tiers else "model"


def confinement_track(paths: Paths, shot: int) -> lf.Track:
    """Saved four-class review, curated regimes, the roster, then D-alpha H-mode."""
    spec = next(s for s in lf.TRACKS if s.key == "confinement")
    reviewed = paths.label_tables / "confinement/review/labels.csv"
    rows = lf.read_rows(reviewed).get(shot)
    if rows:
        source = lf.Source(
            lf.SILVER, "curated four-class confinement review", lambda p: reviewed
        )
        return lf.Track(spec, source, reviewed, rows)
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
        return lf.Track(spec, source, curated, rows)
    roster = roster_file(paths)
    if not roster.is_file():
        raise FileNotFoundError(f"required confinement roster missing: {roster}")
    segments = read_roster(roster, shot)
    if segments:
        source = lf.Source(lf.GENERATED, ROSTER_WHAT, lambda p: roster)
        rows = tuple(
            lf.Row(s["start_ms"], s["end_ms"], s["category"]) for s in segments
        )
        return RosterTrack(spec, source, roster, rows, segments)
    fallback = dalpha_file(paths)
    if not fallback.is_file():
        raise FileNotFoundError(f"required D-alpha L-H fallback missing: {fallback}")
    rows = lf.read_rows(fallback).get(shot)
    if not rows:
        raise ValueError(
            f"shot {shot}: no curated regime, roster or D-alpha L-H coverage"
        )
    source = lf.Source(
        lf.GENERATED, "D-alpha L-H transition detector (dalpha_lh)", lambda p: fallback
    )
    return lf.Track(
        replace(
            spec,
            title=BINARY_CONFINEMENT_TITLE,
            states={ABSENT: "absent", **lf.BINARY, 5: "uncertain"},
        ),
        source,
        fallback,
        rows,
    )


def stored_ae_track(paths: Paths, shot: int) -> lf.Track | None:
    """Paper ae-ours probabilities, if stored and valid; paper operating point.

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
        state = (PRESENT if p >= AE_THRESHOLD else ABSENT) if ok else NOT_OBSERVABLE
        a, b = float(t - step), float(t)
        if rows and rows[-1].category == state and abs(rows[-1].t_end - a) < 1e-6:
            rows[-1] = replace(rows[-1], t_end=b)
        else:
            rows.append(lf.Row(a, b, state))
    spec = next(s for s in lf.TRACKS if s.key == mt.AE)
    source = lf.Source(
        lf.GENERATED,
        f"ae-ours (SELDnet-style), stored CO2 activity, p>={AE_THRESHOLD}",
        lambda p: file,
    )
    return lf.Track(spec, source, file, tuple(rows))


def reviewed_or_stored_ae(
    paths: Paths, shot: int, predictions: Path | None = None
) -> lf.Track | None:
    """Expert review first, then isolated ae-ours inference or stored activity."""
    spec = next(s for s in lf.TRACKS if s.key == mt.AE)
    source = spec.sources[0]
    path = source.locate(paths)
    rows = lf.read_rows(path).get(shot)
    if rows:
        return lf.Track(spec, source, path, rows)
    if predictions is not None:
        rows = lf.read_rows(predictions).get(shot)
        if not rows:
            raise ValueError(f"{predictions}: no AE prediction for shot {shot}")
        source = lf.Source(
            lf.GENERATED,
            f"ae-ours, CO2 activity predictions, p>={AE_THRESHOLD}",
            lambda p: predictions,
        )
        return lf.Track(spec, source, predictions, rows)
    return stored_ae_track(paths, shot)


def local_ece_crashes(paths: Paths, shot: int):
    """Keep the ECE leg before union/merging can replace it with an SXR event."""
    from ..events import heuristics
    from ..events import spans as detectors
    from ..events.verify import NoDataError

    try:
        found, _, info = detectors._sawtooth_crashes(shot, paths)
    except NoDataError as exc:
        return [], {"not_run": str(exc)}
    ece = [c for c in found if c.diag == heuristics.PULSE_DIAG]
    events = heuristics.sawtooth_events_v3(ece)
    return events, info


def saw_source(source: Path, shot: int):
    """Read a physics shot JSON, directory of JSONs/cohort shards, or CSV.

    CSV point rows from the physics export contain inversion/drop evidence.
    For a minimal shot,t_ms CSV, local ECE must independently corroborate it.
    Every file is opened read-only, and all shards used are hashed.
    """
    source = Path(source)
    if source.is_dir():
        jsons = [source / f"{shot}.json", source / "shots" / f"{shot}.json"]
        files = [p for p in jsons if p.is_file()][:1]
        if not files:
            files = sorted(source.glob("cohort-*.csv"))
    else:
        files = [source]
    if not files:
        raise FileNotFoundError(f"{source}: no physics JSON or cohort CSV shards")
    records = [{"path": str(p), "sha256": sha256_of(p)} for p in files]
    if files[0].suffix == ".json":
        data = json.loads(files[0].read_text())
        if int(data["shot"]) != shot:
            raise ValueError(f"{files[0]}: expected shot {shot}")
        points = [
            (float(e["time_s"]) * 1000, e.get("attrs", {}), e.get("confidence"))
            for e in data["crashes"]
        ]
        # The two q-prior states export as uncertain, with the state as the reason.
        codes = {
            "absent": ABSENT,
            "present": PRESENT,
            "uncertain": UNCERTAIN,
            "q_prior_ece_contradicted": UNCERTAIN,
            "q_prior_untested": UNCERTAIN,
            "unassessed": NOT_OBSERVABLE,
        }
        rows = tuple(
            lf.Row(r["start_s"] * 1000, r["end_s"] * 1000, codes[r["state"]])
            for r in data.get("states", [])
        )
        return files[0], points, rows, records
    frame = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    frame = frame.loc[frame["shot"] == shot]
    if frame.empty:
        raise ValueError(f"{source}: no shot {shot}")
    points, rows = [], []
    for r in frame.to_dict("records"):
        attrs = json.loads(r["attrs"]) if isinstance(r.get("attrs"), str) else {}
        if "t_ms" in r:
            points.append((float(r["t_ms"]), attrs, r.get("confidence", 1)))
        elif r["t_start"] == r["t_end"] and int(r["category"]) == PRESENT:
            points.append((float(r["t_start"]), attrs, r.get("confidence", 1)))
        elif r["t_end"] > r["t_start"]:
            rows.append(
                lf.Row(float(r["t_start"]), float(r["t_end"]), int(r["category"]))
            )
    return files[0], points, tuple(rows), records


def ece_evidence(attrs: dict) -> bool:
    """A core drop plus a spatially separate rise/inversion in ECE."""
    if attrs.get("state", "present") != "present":
        return False
    if not attrs.get("core_moves") or not attrs.get("central_relative_drop", 0) > 0:
        return False
    if attrs.get("inversion_channel") is None:
        return False
    a, b = attrs.get("drop_start", 0), attrs.get("drop_stop", 0)
    c, d = attrs.get("rise_start", 0), attrs.get("rise_stop", 0)
    return b > a and d > c and (d <= a or c >= b)


def crash_times(
    paths: Paths,
    shot: int,
    spans=None,
    source: Path | None = None,
    *,
    elm_times=None,
    evidence: Path | None = None,
):
    """ECE-corroborated crashes with an inclusive +/-5 ms D-alpha veto.

    Missing D-alpha cannot establish ELM rejection, so no ticks are shown.
    SXR alone and uncertain physics events never supply a verified tick.
    """
    local = None
    if source is None:
        local, info = local_ece_crashes(paths, shot)
        points = [(e.t0_s * 1000, e.attrs, 1.0) for e in local]
        records = [
            {"path": str(p), "sha256": sha256_of(p)}
            for p in (
                paths.corpus_file(shot),
                corpus_path(shot, corpus=paths.raw_cache),
            )
            if p.is_file()
        ]
        record = {
            "source": "local ECE core-drop and heat-pulse detector",
            "diagnostics": info,
            "files": records,
        }
    else:
        _, points, _, records = saw_source(source, shot)
        if evidence is not None:
            _, proof, _, proof_records = saw_source(evidence, shot)
            records.extend(proof_records)
            enriched = []
            for t, attrs, confidence in points:
                matches = [e for e in proof if abs(e[0] - t) <= 0.5]
                if matches:
                    _, full, proof_confidence = min(
                        matches, key=lambda e: abs(e[0] - t)
                    )
                    attrs = {**attrs, **full}
                    explicit = [
                        c for c in (confidence, proof_confidence) if c is not None
                    ]
                    confidence = min(explicit) if explicit else None
                enriched.append((t, attrs, confidence))
            points = enriched
        record = {
            "source": str(source),
            "files": records,
            "sha256": records[0]["sha256"] if len(records) == 1 else None,
            "evidence_source": str(evidence) if evidence else None,
            "physics_point_match_ms": 0.5,
        }
    verified, rejected = [], []
    for t, attrs, confidence in points:
        ok = ece_evidence(attrs)
        # Original ECE events carry the pulse/block bounds; minimal CSVs
        # must match these local events. Never accept missing evidence.
        if (
            not ok
            and "core_moves" not in attrs
            and attrs.get("state", "present") == "present"
        ):
            if local is None:
                local, _ = local_ece_crashes(paths, shot)
            ok = any(
                abs(e.t0_s * 1000 - t) <= ECE_MATCH_MS
                and e.attrs.get("fall", 0) > 0
                and e.attrs.get("pulse_channel_stop", 0)
                > e.attrs.get("pulse_channel_lo", 0)
                for e in local
            )
        if (
            np.isfinite(t)
            and ok
            and (confidence is None or confidence >= SAWTOOTH_THRESHOLD)
        ):
            verified.append(t)
        else:
            rejected.append(t)
    verified = np.unique(verified)
    elm = np.asarray(elm_times if elm_times is not None else [], float)
    recorded_elm = {t for t, attrs, _ in points if attrs.get("dalpha_coincident")}
    veto = np.array(
        [t in recorded_elm or np.any(np.abs(elm - t) <= ELM_VETO_MS) for t in verified],
        bool,
    )
    kept = verified[~veto] if elm_times is not None else np.array([])
    if spans is not None:
        kept = kept[mt.present_columns(kept, spans)]
    return kept, {
        **record,
        "verified": True,
        "all_times_ms": [p[0] for p in points],
        "ece_times_ms": verified.tolist(),
        "rejected_evidence_times_ms": rejected,
        "rejected_elm_times_ms": verified[veto].tolist(),
        "dalpha_available": elm_times is not None,
        "elm_veto_ms": ELM_VETO_MS,
        "present_times_ms": kept.tolist(),
    }


def sawtooth_track(paths: Paths, shot: int, source: Path | None, verified_times):
    """Expert review first, then physics states; crash candidates are separate."""
    spec = next(s for s in lf.TRACKS if s.key == mt.SAWTOOTH)
    expert = spec.sources[0]
    review = expert.locate(paths)
    rows = lf.read_rows(review).get(shot)
    if rows:
        return lf.Track(spec, expert, review, rows)
    if source is None:
        return lf.Track(spec)
    file, _, rows, _ = saw_source(source, shot)
    times = np.asarray(verified_times)
    name = "physics sawtooth states; separate ECE-supported crash candidates"
    selected = lf.Source(lf.GENERATED, name, lambda p: file)
    if not rows:
        rows = tuple(lf.Row(float(t - 0.5), float(t + 0.5), PRESENT) for t in times)
    return lf.Track(spec, selected, file, rows)


def projection_audit(mask, times, frequencies, raw_spans, band):
    """Independently inspect nonzero pixel coordinates against raw intervals.

    No projection helper, merged-span routine or cached present_columns mask
    is called here; interval ends are explicitly exclusive.
    """
    rr, cc = np.nonzero(mask)
    t, f = np.asarray(times)[cc], np.asarray(frequencies)[rr]
    inside = np.array([any(a <= x < b for a, b in raw_spans) for x in t], bool)
    return {
        "pixels": len(rr),
        "outside_present": int((~inside).sum()),
        "outside_band": int(((f < band[0]) | (f >= band[1])).sum()),
    }


def harmonic_support(n_map, mask, times, frequencies, order=2):
    """Coincident measured n=1 and n=`order` ridges at an `order`:1 frequency ratio.

    This is a plotted-frequency inference, not independent island confirmation.
    A column passes where |f_n/f_1 - order| <= 0.05 * order (the 2:1 tolerance
    is the 0.1 recorded since the first version); the caller decides how much
    passing support, in time and as a share of the jointly measured columns,
    it needs (`harmonic_consistent`).
    """
    t, f = np.asarray(times), np.asarray(frequencies)
    tolerance = HARMONIC_RATIO_TOLERANCE * order
    result = {
        "frequency_ratio_tolerance": tolerance,
        "minimum_support_ms": HARMONIC_MIN_SUPPORT_MS,
        "caption_frequency_step_khz": 1.0,
        "ridge_rule": "per-column pixel-weighted frequency of each measured n",
        "support_ms": 0.0,
        "joint_support_ms": 0.0,
        "joint_columns": 0,
        "passing_columns": 0,
        "passing_fraction": None,
    }
    if n_map is None:
        return result
    means, counts = [], []
    for mode in (1, order):
        measured = mask & (np.asarray(n_map) == mode)
        count = measured.sum(axis=0)
        counts.append(count)
        means.append((measured * f[:, None]).sum(axis=0) / np.maximum(count, 1))
    f1, fn = means
    common = (counts[0] > 0) & (counts[1] > 0) & (f1 > 0)
    ratio = np.divide(fn, f1, out=np.zeros_like(f1), where=common)
    result["joint_columns"] = int(common.sum())
    common &= np.abs(ratio - order) <= tolerance
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0
    result["support_ms"] = float(common.sum() * dt)
    result["joint_support_ms"] = result["joint_columns"] * dt
    result["passing_columns"] = int(common.sum())
    if result["joint_columns"]:
        result["passing_fraction"] = result["passing_columns"] / result["joint_columns"]
    if common.any():
        result["n1_median_khz"] = float(np.median(f1[common]))
        result[f"n{order}_median_khz"] = float(np.median(fn[common]))
    return result


def raster_ae_audit(rgb, bounds, window, frequencies, spans) -> dict:
    """Check pink pixels read from the saved PNG, independently of tag arrays.

    Pixel centres map through the recorded axes rectangle. Allow one raster
    pixel at clip boundaries for rounding/antialiasing; larger leaks fail.
    Only the high-frequency panel is examined, excluding legends and n hues.
    """
    height, width = rgb.shape[:2]
    x0, y0, x1, y1 = bounds
    xx = (np.arange(width) + 0.5) / width
    yy = 1 - (np.arange(height) + 0.5) / height
    rows = np.flatnonzero((yy > y0) & (yy < y1))
    cols = np.flatnonzero((xx > x0) & (xx < x1))
    panel = np.asarray(rgb)[rows][:, cols, :3].astype(float)
    if panel.max(initial=0) > 1:
        panel /= 255
    r, g, b = np.moveaxis(panel, -1, 0)
    pink = (r - g > 0.12) & (b - g > 0.04) & (r > b) & (r > 0.4)
    rr, cc = np.nonzero(pink)
    t = window[0] + (xx[cols[cc]] - x0) / (x1 - x0) * np.diff(window)[0]
    f = frequencies[0] + (yy[rows[rr]] - y0) / (y1 - y0) * np.diff(frequencies)[0]
    dt = np.diff(window)[0] / ((x1 - x0) * width)
    df = np.diff(frequencies)[0] / ((y1 - y0) * height)
    inside = np.array(
        [any(a - dt <= v < z + dt for a, z in spans) for v in t], dtype=bool
    )
    return {
        "pink_pixels": len(rr),
        "outside_present": int((~inside).sum()),
        "below_detector_band": int((f < mt.BANDS[mt.AE][0] - df).sum()),
        "boundary_tolerance_pixels": 1,
        "time_tolerance_ms": float(dt),
        "frequency_tolerance_khz": float(df),
        "rule": "saved PNG pink RGB pixels in processed high-frequency axes",
    }


def _f1_bar(performance: dict) -> float | None:
    """The F1 level of the detector's acceptance bar, from its evaluation."""
    for name, relation, value in (performance.get("bar_criteria") or {}).get("N1", []):
        if name == "f1" and relation == ">=":
            return float(value)
    return None


def ntm_qualification(record: dict, shots: bool = False) -> str:
    """The detector's held-out score against its bar, for a caption.

    The score is the detector's own held-out test shots' (its evaluation.json),
    not the cohort's blind split and not its validation shots.
    """
    performance = record.get("performance") or {}
    parts = []
    score = performance.get("f1")
    if score is not None:
        text = f"held-out F1 {score:.2f}"
        if shots and performance.get("shots"):
            text += f" on {performance['shots']} shots"
        parts.append(text)
    met = (record.get("primary_bars") or {}).get("N1")
    if met is not None:
        bar = _f1_bar(performance)
        target = f"our {bar:g} bar" if bar else "its bar"
        parts.append(("meets " if met else "below ") + target)
    return ", ".join(parts)


def ntm_description(record: dict) -> str:
    """The NTM track's source text: detector suggestions and their score."""
    detail = ntm_qualification(record, shots=True)
    return "detector (suggestions" + (f"; {detail})" if detail else ")")


def sawtooth_row_source(rows: list[dict], guard: dict | None) -> str:
    """The physics sawtooth row's source text; blank time is named, not left silent."""
    unassessed = any(r["state"] == "unassessed" for r in rows)
    if unassessed and (guard or {}).get("cutoff_proxy"):
        return "physics; blank: ECE cut-off"
    return "physics labels"


def sawtooth_caption(record: dict) -> str:
    """Summarise exact source states, for the appendix."""
    rows = record.get("state_intervals_ms", [])
    durations = {
        state: sum(r["end_ms"] - r["start_ms"] for r in rows if r["state"] == state)
        for state in ("present", "absent", "uncertain", "unassessed")
    }
    summary = (
        ", ".join(
            f"{state} {duration:.0f} ms"
            for state, duration in durations.items()
            if duration
        )
        or "unassessed"
    )
    source = "physics labels"
    if record.get("tier") == lf.SILVER:
        source = "expert intervals"
    elif record.get("tier") == lf.LEGACY:
        source = "imported intervals"
    elif not record.get("what", "").startswith("physics"):
        source = "ECE-supported crash detector"
    guard = record.get("density_guard") or {}
    if durations["unassessed"] and guard.get("cutoff_proxy"):
        source += "; ECE density proxy"
        if "bt_missing" in guard.get("status", "").lower():
            source += "; Bt not in local corpus"
    return f"Sawtooth: {summary} ({source})."


def _has_present(record: dict | None) -> bool:
    rows = (record or {}).get("state_intervals_ms", [])
    return any(r["state"] == "present" for r in rows)


def harmonic_consistent(support: dict | None) -> bool:
    """A ridge pair that sits at a multiple of the n=1 frequency for at least the
    minimum time AND for at least `HARMONIC_MIN_PASSING_FRACTION` of the columns
    that measure both ridges (a record without that share is not consistent)."""
    support = support or {}
    floor = support.get("minimum_support_ms", HARMONIC_MIN_SUPPORT_MS)
    share = support.get("passing_fraction")
    return (
        support.get("support_ms", 0) >= floor
        and share is not None
        and share >= HARMONIC_MIN_PASSING_FRACTION
    )


def harmonic_clause(drawn: dict) -> str:
    """Name the ridges whose recorded support puts them at multiples of n=1."""
    ridges = []
    for n, key in ((2, "harmonic_support"), (3, "harmonic3_support")):
        if harmonic_consistent(drawn.get(key)):
            ridges.append(n)
    if ridges == [2, 3]:
        return "n=2 and n=3 ridges are consistent with harmonics of the n=1 mode"
    if ridges:
        return f"the n={ridges[0]} ridge is consistent with a harmonic of the n=1 mode"
    return ""


def _elm_name(records: dict) -> str:
    """Name the ELM source by its tier (expert, imported or detector intervals)."""
    tier = (records.get("edge_localized_mode") or {}).get("tier")
    kind = {lf.SILVER: "Expert", lf.LEGACY: "Imported", lf.GENERATED: "Detected"}
    return f"{kind[tier]} ELM intervals" if tier in kind else "ELM intervals"


def elm_qh_overlaps(
    elm_spans, rows, window: tuple[float, float], four_class: bool = True
) -> list[dict]:
    """ELM present spans that overlap an ELM-free class (QH, WPQH), clipped to the
    window, with the class. Only a four-class confinement row has such classes."""
    if not four_class:
        return []
    out = []
    for a, b in elm_spans:
        for r in rows:
            lo, hi = max(a, r.t_start, window[0]), min(b, r.t_end, window[1])
            if r.category in ELM_FREE_CLASSES and hi > lo:
                out.append({"span_ms": [lo, hi], "category": r.category})
    return out


def _elm_free_names(drawn: dict) -> str:
    names = {3: "QH", 4: "WPQH"}
    found = sorted({o["category"] for o in drawn.get("elm_qh_overlaps_ms", [])})
    return "/".join(names[c] for c in found) or "QH"


def _elm_free_unreviewed(records: dict) -> bool:
    segments = (records.get("confinement") or {}).get("segments") or []
    return any(
        s["tier"] == "unreviewed" and s["category"] in ELM_FREE_CLASSES
        for s in segments
    )


def elm_qh_sentence(records: dict, drawn: dict) -> str:
    """ELM intervals inside an ELM-free class: say whose labels disagree."""
    elm_tier = (records.get("edge_localized_mode") or {}).get("tier")
    name = _elm_free_names(drawn)
    qh_model = (records.get("confinement") or {}).get("segments") is not None
    if elm_tier == lf.GENERATED and qh_model and _elm_free_unreviewed(records):
        return (
            f"ELM intervals overlap the {name} span; {name} is ELM-free by "
            "definition, so the D-alpha ELM boxes inside it are the detector's "
            f"and the {name} label is the model's, both unreviewed."
        )
    return (
        f"{_elm_name(records)} overlap the {name} span; {name} is ELM-free by "
        "definition, so the two sources disagree there."
    )


def caption(shot: int, records: dict, drawn: dict) -> str:
    """Describe what is drawn; sources, thresholds and caveats are in the appendix.

    Shot-specific qualifications are not coded here: they arrive in `drawn` from
    the recorded annotations and measurements (conflicts, review caveats,
    harmonic support).
    """
    tagged = drawn.get("blobs", {}).get("tagged")
    ae = tagged is None or bool(tagged.get(mt.AE))
    ntm = tagged is None or bool(tagged.get(mt.NTM))
    ae_record = records.get(mt.AE) or {}
    ntm_record = records.get(mt.NTM) or {}
    sentences = [
        f"DIII-D shot {shot}.",
        (
            "(a) Raw CO2 interferometer cross-power spectrogram (linear frequency "
            "axis, 0–250 kHz)."
        ),
        (
            "(b) TokEye coherent-mode mask after small-object removal; below "
            "30 kHz coloured by toroidal mode number n (Mirnov array)."
        ),
    ]
    if drawn.get("tokeye_transient_drawn"):
        sentences.append("Red: TokEye's transient channel.")
    if ae:
        if ae_record.get("tier") == lf.GENERATED:
            bin_ms = ae_record.get("temporal_bin_ms") or 25
            trained = ""
            if ae_record.get("what", "").startswith("ae-ours"):
                trained = (
                    "; trained on TokEye-mask-derived targets, so not "
                    "independent of TokEye"
                )
            sentences.append(
                "Pink: mask pixels ≥60 kHz while the CO2 AE detector (80–250 kHz "
                f"input band{trained}) is positive ({bin_ms:g} ms bins)."
            )
        else:
            sentences.append("Pink: mask pixels ≥60 kHz inside labelled AE time.")
    if ntm:
        if ntm_record.get("tier") == lf.GENERATED:
            detector = "the NTM detector"
            qualification = ntm_qualification(ntm_record)
            if qualification:
                detector += f" ({qualification})"
            sentences.append(
                f"Orange outlines: n=1/2 pixels while {detector} is positive."
            )
        else:
            sentences.append("Orange outlines: n=1/2 pixels inside labelled NTM time.")
    if ae or ntm:
        text = "Highlights mark time/band coincidence only"
        clause = harmonic_clause(drawn) if ntm else ""
        sentences.append(text + (f"; {clause}." if clause else "."))
    sentences.append(
        "(c) D-alpha, ELM intervals (red) and confinement regimes. (d) NBI "
        "power. (e) Label rows; sources in the appendix."
    )
    if drawn.get("elm_hmode_conflicts_ms"):
        sentences.append(
            f"{_elm_name(records)} and the H-mode detector disagree in parts of "
            "this window."
        )
    if drawn.get("elm_qh_overlaps_ms"):
        sentences.append(elm_qh_sentence(records, drawn))
    if drawn.get("ae_physical_review_caveat"):
        sentences.append(drawn["ae_physical_review_caveat"])
    text = " ".join(sentences)
    if len(text.split()) > CAPTION_MAX_WORDS:
        raise ValueError(
            f"caption exceeds {CAPTION_MAX_WORDS} words: {len(text.split())}"
        )
    return text


def _source_name(key: str, record: dict) -> str:
    tier = record.get("tier")
    if tier == lf.SILVER:
        return "expert"
    if tier == lf.LEGACY:
        return "imported labels"
    if key == mt.SAWTOOTH and record.get("what", "").startswith("physics"):
        return "physics labels"
    if key == mt.NTM:
        return ntm_description(record)
    if key == mt.AE:
        if record.get("what", "").startswith("ae-ours"):
            return "CO2 neural detector"
        return "CO2 frame detector"
    if key == "confinement":
        if record.get("segments") is not None:
            return "model roster"
        return "D-alpha detector"
    return "detector"


def _model_text(model: str) -> str:
    """The roster's model description without its pointer to a repo document."""
    return re.sub(r";\s*docs/\S+", "", model).strip()


def roster_note(record: dict) -> str | None:
    """What the roster row is, from the record's own roster metadata and segments."""
    segments = record.get("segments")
    if segments is None:
        return None
    meta = record.get("roster") or {}
    segmentation = meta.get("segmentation") or {}
    parts = [
        "Confinement: model labels from the released confinement roster"
        + (f" ({_model_text(meta['model'])})" if meta.get("model") else "")
        + "."
    ]
    if segmentation.get("active"):
        parts.append(f"Active time: {segmentation['active']}.")
    if segmentation.get("confidence_floor"):
        parts.append(
            "Confidence floor: "
            + segmentation["confidence_floor"].replace("`", "")
            + "."
        )
    sources = {s["source"] for s in segments}
    if "ensemble" in sources:
        text = "This shot is read by the ensemble of the fold models"
        if any(s["extrapolated"] for s in segments):
            text += (
                ", and lies past the last curated shot, so the network never saw "
                "its campaign"
            )
        parts.append(text + ".")
    if any(s["tier"] == "unreviewed" for s in segments):
        parts.append(
            "Unreviewed marks a QH or WPQH segment outside the curated set, which "
            "may hold ELM-free or quiescent H-modes the network reads as QH "
            "(unverified)."
        )
    parts.append("Blank marks time outside the roster's beam-on segments.")
    return " ".join(parts)


def _tokeye_threshold(drawn: dict) -> float:
    """The cut the figure applied to TokEye's channels; the network's operating
    point where the record does not say."""
    return drawn.get("tokeye_threshold", mt.PROB_THRESHOLD)


def _chain_note(drawn: dict) -> str:
    """The mask chain's steps, with the persistent-row step and what it did."""
    share = round(mt.PERSISTENT_ROW_SHARE * 100)
    text = (
        f"The mask chain is: TokEye coherent mask at ≥{_tokeye_threshold(drawn):g} "
        "and not transient; persistent-row step (a row lit for over "
        f"{share} % of the record "
        "is a persistent-line candidate, not an identified pickup line, and is "
        "kept only where the rows above and below are both lit); removal of objects "
        f"under {mt.MIN_SIZE['wide']} (wide pass) or {mt.MIN_SIZE['zoom']} (zoom "
        f"pass) pixels; filling of holes under {mt.HOLE_AREA} pixels; components; "
        "time/band tags."
    )
    if drawn.get("tokeye_transient_drawn"):
        text += (
            " Red: TokEye's transient channel at the same cut, after the same "
            "small-object removal; the coherent mask excludes it."
        )
    rows = drawn.get("persistent_line_rows")
    if rows is not None:
        if not any(rows.values()):
            text += " No row reached the persistent share in this shot's record."
        else:
            text += (
                f" {rows['wide']} wide-pass and {rows['zoom']} zoom-pass rows "
                "reached the persistent share in this shot's record."
            )
    return text


def _harmonic_text(support: dict | None, order: int) -> str | None:
    """One ridge's frequency-ratio support in the appendix: where it passes, over
    the time both ridges are measured. Nothing where too little time is measured
    to say anything."""
    support = support or {}
    joint = support.get("joint_support_ms", 0)
    if joint < support.get("minimum_support_ms", HARMONIC_MIN_SUPPORT_MS):
        return None
    share = support.get("passing_fraction")
    percent = f"{HARMONIC_RATIO_TOLERANCE:.0%}"
    text = (
        f"n={order} lies within {percent} of {order}×f(n=1) in "
        f"{support.get('support_ms', 0):.0f} of {joint:.0f} ms where both are "
        "measured"
    )
    detail = []
    if share is not None:
        detail.append(f"{share:.0%}")
    if "n1_median_khz" in support and f"n{order}_median_khz" in support:
        detail.append(
            f"median n=1 {support['n1_median_khz']:.1f} kHz, "
            f"n={order} {support[f'n{order}_median_khz']:.1f} kHz"
        )
    return text + (f" ({'; '.join(detail)})" if detail else "")


def training_note(training: dict | None) -> str:
    """Whether the figure's shot was in each detector's training set."""
    names = {mt.AE: "AE", mt.NTM: "NTM"}
    known = {
        names[key]: bool(record["figure_shot_in_training"])
        for key, record in (training or {}).items()
        if key in names and "figure_shot_in_training" in record
    }
    inside = [name for name, value in known.items() if value]
    outside = [name for name, value in known.items() if not value]
    if not known:
        return ""
    if not inside:
        if len(outside) == 2:
            return "This shot is in neither the AE nor the NTM detector's training set."
        return f"This shot is not in the {outside[0]} detector's training set."
    text = f"This shot is in the {' and '.join(inside)} detector's training set"
    if outside:
        text += f" and not in the {' or '.join(outside)} detector's"
    return text + "."


def appendix_notes(
    shot: int, records: dict, drawn: dict, training: dict | None = None
) -> str:
    """Keep source caveats and measured shot details outside the paper caption.

    `training` is the record's `detector_training`: whether this shot was in each
    detector's training set."""
    notes = [
        f"DIII-D shot {shot}. The raw spectrogram uses one colour scale.",
        (
            "Toroidal mode number n is measured by the Mirnov array. "
            "The frequency axis is linear, 0–250 kHz, in both spectrograms, with "
            "no scale break. The raw spectrogram is the wide-range pass; the "
            "mask is drawn from the higher-resolution pass below 50 kHz and the "
            "wide-range pass above it, because the higher-resolution pass's "
            "decimation filter rolls off above about 50 kHz. The mask is TokEye "
            "run on the same record, the cross-power of CO2 interferometer chords R0 and V1 "
            "(the AE review page's rows, averaged over 8 columns so that a mode both "
            "chords see stands out of the noise each sees alone). Toroidal mode numbers and the NTM candidates come from "
            "the Mirnov array and are drawn on the CO2 mask, so a pixel's n is "
            "not measured on the signal that lit it."
        ),
        _chain_note(drawn),
    ]
    sources = []
    for key, name in (
        (mt.AE, "AE"),
        (mt.NTM, "NTM"),
        ("confinement", "H-mode"),
        ("edge_localized_mode", "ELMs"),
        (mt.SAWTOOTH, "sawtooth"),
    ):
        record = records.get(key)
        if record is None:
            continue
        if key == "confinement":
            name = record.get("title", name).capitalize()
        sources.append(f"{name}: {_source_name(key, record)}")
    if sources:
        notes.append("Tracks: " + "; ".join(sources) + ".")
    ae = records.get(mt.AE) or {}
    ntm = records.get(mt.NTM) or {}
    ae_detector = ae.get("tier") == lf.GENERATED
    ae_ours = ae.get("what", "").startswith("ae-ours")
    thresholds = [f"TokEye {_tokeye_threshold(drawn):g}"]
    if ae_detector:
        thresholds.append(f"AE {ae.get('decision_threshold') or AE_THRESHOLD:g}")
    if ntm.get("tier") == lf.GENERATED:
        thresholds.append(f"NTM {ntm.get('decision_threshold') or NTM_THRESHOLD:g}")
    notes.append("Operating probability thresholds: " + "; ".join(thresholds) + ".")
    tagged = drawn.get("blobs", {}).get("tagged")
    ae_highlighted = tagged is None or bool(tagged.get(mt.AE))
    if ae_detector and ae_highlighted:
        bin_ms = ae.get("temporal_bin_ms", 25 if ae_ours else None)
        text = "AE highlights intersect detector-positive time and mask pixels ≥60 kHz"
        if bin_ms is not None:
            text += f" in {bin_ms:g} ms bins"
        notes.append(text + ".")
        notes.append(
            "The detector's input band is 80–250 kHz; pink starts at 60 kHz (the "
            "AE/NTM split), so mask pixels at 60–80 kHz are "
            "highlighted by time coincidence with the detector, not detected by it."
        )
        if ae_ours:
            notes.append(
                "AE targets used TokEye's mask; highlights are not independent "
                "physical confirmation."
            )
        else:
            notes.append(
                "The frame detector was trained on the owner's reviewed AE labels; "
                "TokEye's mask only up-weights its MHD-absent frames. Highlights "
                "mark coincidence, not independent physical confirmation."
            )
    if ntm.get("tier") == lf.GENERATED:
        notes.append(
            "NTM detector: "
            + ntm_qualification(ntm)
            + "; shared magnetic inputs, not independent confirmation."
        )
    if records.get(mt.NTM) is not None:
        notes.append(
            "NTM outlines require measured and dominant n=1 or 2 at ≤30 kHz. "
            "They mark time coincidence only; they do not establish a 2/1 island."
        )
        display = drawn.get("ntm_outline_display")
        if display:
            notes.append(
                f"Display only: outlines enclosing fewer than {display['min_px']} "
                "print pixels are not drawn "
                f"({display['omitted_fragments']} fragments omitted here); "
                "tags, counts and audits are unchanged."
            )
        dashed = (drawn.get("ntm_outline_display") or {}).get("dashed_regions", 0)
        if dashed:
            notes.append(
                "Dashed orange outlines show the rest of a tagged component's "
                "measured n=1 or 2 support, outside NTM-positive time: the "
                "detector's timing cuts the tag, not the mode."
            )
        harmonics = [
            text
            for order, key in ((2, "harmonic_support"), (3, "harmonic3_support"))
            if (text := _harmonic_text(drawn.get(key), order))
        ]
        if harmonics:
            notes.append(
                "Frequency ratios of the measured-n ridges, in NTM-positive time: "
                + "; ".join(harmonics)
                + ". A frequency ratio cannot separate harmonics of one island "
                "from phase-locked coupled modes; the poloidal number m needs "
                "EFIT q or the poloidal array."
            )
    note = training_note(training)
    if note:
        notes.append(note)
    if drawn.get("lmode_inferred"):
        notes.append(
            "L-mode (inferred) uses pre-transition H-mode-detector absent shading."
        )
    confinement_note = roster_note(records.get("confinement") or {})
    if confinement_note:
        notes.append(confinement_note)
    if records.get(mt.SAWTOOTH) is not None:
        saw = records[mt.SAWTOOTH]
        notes.append(sawtooth_caption(saw))
        notes.append(
            "The sawtooth row preserves all source intervals, including those "
            "shorter than 10 ms, with no state smoothing. Hatching marks "
            "uncertainty; blank marks unassessed time."
        )
        if saw.get("what", "").startswith("physics") and not _has_present(saw):
            notes.append(
                "No sawtooth is labelled present in this window; these physics "
                "labels remain unvalidated."
            )
    late = drawn.get("late_untagged_high_frequency")
    if late:
        lo, hi = late["band_khz"]
        timing = (
            f" after {round(late['first_time_ms'] / 1000, 2):g} s"
            if late.get("first_time_ms") is not None
            else " at late times"
        )
        notes.append(
            f"Mask lines at {lo:.0f}–{hi:.0f} kHz remain visible{timing}, "
            "but stay untagged because the CO2 AE detector is negative. "
            "The AE tags inherit the detector's timing, including its "
            "negative gaps; they do not imply that the lines disappear."
        )
    if drawn.get("sawtooth_strip_shown"):
        notes.append(
            "ECE-supported crash candidates use channel-order geometry and a "
            "±5 ms D-alpha veto; they are unvalidated research evidence."
        )
    if drawn.get("first_large_peak_before_expert_ms") is not None:
        peak = drawn["largest_dalpha_peak_ms"]
        start = drawn["expert_elm_start_ms"]
        notes.append(
            f"The largest D-alpha spike ({int(np.floor(peak + 0.5))} ms) precedes "
            f"the expert span (from {start:.0f} ms)."
        )
    if drawn.get("elm_hmode_conflicts_ms"):
        notes.append(
            f"{_elm_name(records)} overlap H-mode-detector absent time; "
            "sources disagree."
        )
    if drawn.get("elm_qh_overlaps_ms"):
        spans = "; ".join(
            f"{o['span_ms'][0]:.0f}–{o['span_ms'][1]:.0f} ms"
            for o in drawn["elm_qh_overlaps_ms"]
        )
        notes.append(f"{elm_qh_sentence(records, drawn)} Overlaps: {spans}.")
    if drawn.get("ae_physical_review_caveat"):
        notes.append(drawn["ae_physical_review_caveat"])
    marks = []
    if drawn.get("elm_crowd_spans_ms"):
        marks.append("Open circles delimit expert spans containing many ELMs")
    if marks:
        text = "; ".join(marks) + "."
        notes.append(text[0].upper() + text[1:])
    return " ".join(notes)
