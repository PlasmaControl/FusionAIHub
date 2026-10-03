"""Read-only source selection for the TokEye interpreter figure.

CONFINEMENT_RUN_DIR defaults to runs/labeler/confinement/v1 in the checkout
containing LABELER_LABEL_TABLES. Both curated and fallback files are required
when needed; a missing file never silently substitutes the H-mode frame model.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Paths, sha256_of
from ..events.catalog.states import ABSENT, NOT_OBSERVABLE, PRESENT, UNCERTAIN
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
SAWTOOTH_DISPLAY_MIN_MS = 10.0


def tokeye_fingerprints(paths, shot, group, row, inference_code) -> dict:
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
    if track.spec.key != "confinement" or track.spec.title == "H-mode":
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
    """Merge sub-10 ms state slivers into the longer touching neighbour.

    Display only: source rows stay exact. Process shortest sliver first, ties
    left to right; equal neighbour durations prefer the earlier state. Never
    bridge a gap in assessment. Independently verified crash ticks stay exact.
    """
    rows = [
        replace(r, t_start=max(r.t_start, window[0]), t_end=min(r.t_end, window[1]))
        for r in track.rows
        if r.t_end > window[0] and r.t_start < window[1]
    ]
    changes = []
    while True:
        merged = []
        for row in rows:
            if (
                merged
                and merged[-1].category == row.category
                and abs(merged[-1].t_end - row.t_start) < 1e-6
            ):
                merged[-1] = replace(merged[-1], t_end=row.t_end)
            else:
                merged.append(row)
        rows = merged
        candidates = []
        for i, row in enumerate(rows):
            duration = row.t_end - row.t_start
            neighbours = [
                j
                for j in (i - 1, i + 1)
                if 0 <= j < len(rows)
                and abs(
                    min(row.t_end, rows[j].t_end) - max(row.t_start, rows[j].t_start)
                )
                < 1e-6
            ]
            if duration < SAWTOOTH_DISPLAY_MIN_MS and neighbours:
                candidates.append((duration, i, neighbours))
        if not candidates:
            break
        _, i, neighbours = min(candidates)
        j = max(neighbours, key=lambda j: (rows[j].t_end - rows[j].t_start, -j))
        row = rows[i]
        category = rows[j].category
        changes.append(
            {
                "start_ms": row.t_start,
                "end_ms": row.t_end,
                "from": row.category,
                "to": category,
            }
        )
        rows[i] = replace(row, category=category)
    return replace(track, rows=tuple(rows)), changes


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
    return lf.Track(
        replace(
            spec, title="H-mode", states={ABSENT: "absent", **lf.BINARY, 5: "uncertain"}
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
        codes = {
            "absent": ABSENT,
            "present": PRESENT,
            "uncertain": UNCERTAIN,
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
        "pixels_55_to_60": int(((f >= 55) & (f < 60)).sum()),
    }


def harmonic_support(n_map, mask, times, frequencies):
    """Coincident measured n=1/n=2 ridges at a 2:1 frequency ratio.

    This is a plotted-frequency inference, not independent island confirmation.
    Require |f2/f1 - 2| <= 0.1 and at least 50 ms of sampled support.
    """
    t, f = np.asarray(times), np.asarray(frequencies)
    result = {
        "frequency_ratio_tolerance": 0.1,
        "minimum_support_ms": 50.0,
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
    for mode in (1, 2):
        measured = mask & (np.asarray(n_map) == mode)
        count = measured.sum(axis=0)
        counts.append(count)
        means.append((measured * f[:, None]).sum(axis=0) / np.maximum(count, 1))
    f1, f2 = means
    common = (counts[0] > 0) & (counts[1] > 0) & (f1 > 0)
    ratio = np.divide(f2, f1, out=np.zeros_like(f1), where=common)
    result["joint_columns"] = int(common.sum())
    common &= np.abs(ratio - 2) <= 0.1
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0
    result["support_ms"] = float(common.sum() * dt)
    result["joint_support_ms"] = result["joint_columns"] * dt
    result["passing_columns"] = int(common.sum())
    if result["joint_columns"]:
        result["passing_fraction"] = result["passing_columns"] / result["joint_columns"]
    if common.any():
        result.update(
            n1_median_khz=float(np.median(f1[common])),
            n2_median_khz=float(np.median(f2[common])),
        )
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


def ntm_description(record: dict) -> str:
    """Qualify detector suggestions using their recorded evaluation and bar."""
    text = "NTM detector suggestions"
    qualifiers = []
    score = (record.get("performance") or {}).get("f1")
    if score is not None:
        qualifiers.append(f"F1 {score:.2f}")
    if (record.get("primary_bars") or {}).get("N1") is False:
        qualifiers.append("below acceptance bar")
    return text + (f" ({', '.join(qualifiers)})" if qualifiers else "")


def sawtooth_caption(record: dict) -> str:
    """Summarise exact source states, including an omitted row's assessment."""
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
            source += "; Bt unavailable"
    return f"Sawtooth: {summary} ({source})."


def caption(shot: int, records: dict, drawn: dict) -> str:
    """Source-specific paper prose, without internal source identifiers."""
    sources = []
    tagged = drawn.get("blobs", {}).get("tagged")
    for key, name in (
        (mt.AE, "AE"),
        (mt.NTM, "NTM"),
        ("confinement", "Regime"),
        ("edge_localized_mode", "ELMs"),
    ):
        if key in (mt.AE, mt.NTM) and tagged is not None and not tagged.get(key):
            continue
        record = records.get(key)
        if record is None:
            continue
        if key == mt.NTM and record["tier"] == lf.GENERATED:
            sources.append(ntm_description(record))
            continue
        if record["tier"] == lf.SILVER:
            description = "expert"
        elif record["tier"] == lf.LEGACY:
            description = "imported"
        elif key == mt.AE:
            description = (
                "neural detector on CO2 interferometer data"
                if record["what"].startswith("ae-ours")
                else "frame detector on CO2 interferometer data"
            )
            if record["what"].startswith("ae-ours"):
                description += f" (p≥{AE_THRESHOLD}; targets used TokEye's mask)"
            elif record.get("decision_threshold") is not None:
                description += f" (p≥{record['decision_threshold']})"
        elif key == "confinement":
            description = "D-alpha detector"
            if drawn.get("regimes_shown") == []:
                description += " (uncertain here)"
        else:
            description = "detector"
        if key == "confinement":
            name = (
                record["title"].capitalize()
                if record["title"] == "regime"
                else record["title"]
            )
        sources.append(f"{name}: {description}")
    sentences = [
        f"DIII-D shot {shot}. Raw bands normalised separately; TokEye extracts coherent modes.",
    ]
    if sources:
        sentences.append("; ".join(sources) + ".")
    if tagged is None or tagged.get(mt.AE):
        ae = records.get(mt.AE) or {}
        bin_ms = ae.get(
            "temporal_bin_ms", 25 if ae.get("what", "").startswith("ae-ours") else None
        )
        text = "Pink: AE overlap in detector band ≥80 kHz"
        if bin_ms is not None and ae.get("tier") == lf.GENERATED:
            text += f" in {bin_ms:g} ms bins"
        sentences.append(text + ".")
    if tagged is None or tagged.get(mt.NTM):
        sentences.append(
            "Dashed outlines: measured and dominant n=1 or 2, ≤30 kHz (shared inputs)."
        )
    else:
        sentences.append("Measured n: ≤30 kHz.")
    if drawn.get("lmode_inferred"):
        sentences.append("L-mode (inferred): pre-transition H-mode-absent shading.")
    if records.get(mt.SAWTOOTH) is not None:
        text = sawtooth_caption(records[mt.SAWTOOTH])
        if not drawn.get("sawtooth_track_shown", True):
            text = text.removesuffix(".") + "; row omitted."
        sentences.append(text)
    late = drawn.get("late_untagged_high_frequency")
    if late:
        lo, hi = late["band_khz"]
        if shot == 201978:
            sentences.append(
                "Evenly spaced magnetics-only lines after 2.8 s remain unlabelled."
            )
        else:
            sentences.append(
                f"Late magnetics-only {lo:.0f}–{hi:.0f} kHz pixels remain unlabelled."
            )
    if drawn.get("sawtooth_strip_shown"):
        sentences.append(
            "ECE-supported crash candidates: channel-order geometry; ±5 ms D-alpha veto."
        )
    if drawn.get("first_large_peak_before_expert_ms") is not None:
        peak = drawn["largest_dalpha_peak_ms"]
        start = drawn["expert_elm_start_ms"]
        sentences.append(
            f"The largest D-alpha spike ({int(np.floor(peak + 0.5))} ms) precedes "
            f"the expert ELM interval (from {start:.0f} ms)."
        )
    if drawn.get("elm_hmode_conflicts_ms"):
        sentences.append(
            "Expert ELM intervals overlap H-mode-detector absent time; sources disagree."
        )
    keys = []
    states = drawn.get("display_state_keys")
    if states is None or "uncertain" in states:
        keys.append("Hatching: uncertain")
    if states is None or "blank" in states:
        keys.append("Blank: unassessed/unobservable")
    elm = records.get("edge_localized_mode", {})
    if elm.get("tier") == lf.SILVER and drawn.get("elm_crowd_spans_ms", True):
        keys.append("Circles: expert spans containing many ELMs")
    if drawn.get("elm_peaks_in_label", True):
        keys.append("Triangles: threshold D-alpha peaks")
    if keys:
        sentences.append("; ".join(keys) + ".")
    text = " ".join(sentences)
    if len(text.split()) > 150:
        raise ValueError(f"caption exceeds 150 words: {len(text.split())}")
    return text
