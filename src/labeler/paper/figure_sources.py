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
SAWTOOTH_DISPLAY_MIN_MS = 0.0


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
    text = "NTM candidate suggestions"
    qualifiers = []
    score = (record.get("performance") or {}).get("f1")
    if score is not None:
        qualifiers.append(f"detector F1 {score:.2f}")
    if (record.get("primary_bars") or {}).get("N1") is False:
        qualifiers.append("below bar")
    return text + (f" ({', '.join(qualifiers)})" if qualifiers else "")


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


def caption(shot: int, records: dict, drawn: dict) -> str:
    """Explain visible timing and colour choices; full sources are in the appendix."""
    tagged = drawn.get("blobs", {}).get("tagged")
    ae = tagged is None or bool(tagged.get(mt.AE))
    ntm = tagged is None or bool(tagged.get(mt.NTM))
    sentences = [
        (
            f"DIII-D shot {shot}: raw signals → TokEye-processed modes → event labels; "
            "raw bands normalised separately."
        ),
        "Toroidal mode number n: Mirnov array.",
    ]
    ae_record = records.get(mt.AE) or {}
    if ae:
        if ae_record.get("tier") == lf.GENERATED:
            bin_ms = ae_record.get("temporal_bin_ms") or 25
            timing = f"Pink AE tags inherit the CO2 detector's {bin_ms:g} ms timing"
            late = drawn.get("late_untagged_high_frequency")
            if shot == 201978 and late and late.get("first_time_ms") is not None:
                timing += (
                    ": the same lines stay untagged after "
                    f"{round(late['first_time_ms'] / 1000, 2):g} s, "
                    "where it is negative"
                )
            sentences.append(timing + ".")
        else:
            sentences.append("Pink AE: time/band coincidence with source labels.")
        sentences.append("AE floor: 80 kHz input band; 55–80 kHz stays white.")
    if shot == 201978 and drawn.get("first_large_peak_before_expert_ms") is not None:
        peak = int(np.floor(drawn["largest_dalpha_peak_ms"] + 0.5))
        sentences.append(f"The first ELM ({peak} ms) precedes the expert span.")
    if ntm:
        text = "Orange NTM outlines: time coincidence only"
        if shot == 201978:
            text += " (including 3–5 kHz fragments)"
        record = records.get(mt.NTM) or {}
        if record.get("tier") == lf.GENERATED:
            qualification = ntm_description(record).partition(" (")[2].rstrip(")")
            if qualification:
                text += f"; {qualification}"
        sentences.append(text + ".")
    if ae and ae_record.get("tier") == lf.GENERATED:
        sentences.append("AE targets used TokEye's mask (circularity).")
    saw = records.get(mt.SAWTOOTH)
    if saw is not None and not any(
        r["state"] == "present" for r in saw.get("state_intervals_ms", [])
    ):
        sentences.append("Sawtooth: four-state notation only in this window.")
    text = " ".join(sentences)
    if len(text.split()) > 95:
        raise ValueError(f"caption exceeds 95 words: {len(text.split())}")
    return text


def appendix_notes(shot: int, records: dict, drawn: dict) -> str:
    """Keep source caveats and measured shot details outside the paper caption."""
    notes = [
        f"DIII-D shot {shot}. Raw bands are normalised separately.",
        (
            "Toroidal mode number n is measured by the Mirnov array. "
            "The 0–30 kHz range is vertically expanded; 30–55 kHz is compressed. "
            "Both use the higher-resolution spectrogram."
        ),
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
        tier = record.get("tier")
        if tier == lf.SILVER:
            source = "expert"
        elif tier == lf.LEGACY:
            source = "imported labels"
        elif key == mt.SAWTOOTH and record.get("what", "").startswith("physics"):
            source = "physics labels"
        elif key == mt.NTM:
            source = ntm_description(record).removeprefix("NTM ")
        elif key == mt.AE:
            source = (
                "CO2 neural detector"
                if record.get("what", "").startswith("ae-ours")
                else "CO2 frame detector"
            )
        elif key == "confinement":
            source = "D-alpha detector"
        else:
            source = "detector"
        if key == "confinement":
            name = record.get("title", name).capitalize()
        sources.append(f"{name}: {source}")
    if sources:
        notes.append("Tracks: " + "; ".join(sources) + ".")
    notes.append(
        "Operating probability thresholds (for detector sources): "
        f"TokEye {mt.PROB_THRESHOLD:g}; "
        f"AE {(records.get(mt.AE) or {}).get('decision_threshold') or AE_THRESHOLD:g}; "
        f"NTM {(records.get(mt.NTM) or {}).get('decision_threshold') or NTM_THRESHOLD:g}."
    )
    ae = records.get(mt.AE) or {}
    if ae.get("tier") == lf.GENERATED:
        bin_ms = ae.get(
            "temporal_bin_ms", 25 if ae.get("what", "").startswith("ae-ours") else None
        )
        text = (
            "AE highlights intersect detector-positive time and detector band ≥80 kHz"
        )
        if bin_ms is not None:
            text += f" in {bin_ms:g} ms bins"
        notes.append(text + ".")
        notes.append(
            "The 80 kHz AE floor matches the ae-ours input band (80–250 kHz); "
            "55–80 kHz cascade lines stay white even during detector-positive time."
        )
        notes.append(
            "AE targets used TokEye's mask; highlights are not independent "
            "physical confirmation."
        )
    if records.get(mt.NTM, {}).get("tier") == lf.GENERATED:
        notes.append(
            ntm_description(records[mt.NTM])
            + "; shared magnetic inputs, not independent confirmation."
        )
    if records.get(mt.NTM) is not None:
        notes.append(
            "NTM outlines require measured and dominant n=1 or 2 at ≤30 kHz. "
            "They mark time coincidence only; they do not establish a 2/1 island."
        )
        if shot == 201978:
            notes.append("These outlines also include small 3–5 kHz fragments.")
    if drawn.get("lmode_inferred"):
        notes.append(
            "L-mode (inferred) uses pre-transition H-mode-detector absent shading."
        )
    if records.get(mt.SAWTOOTH) is not None:
        notes.append(sawtooth_caption(records[mt.SAWTOOTH]))
        notes.append(
            "The sawtooth row preserves all source intervals, including those "
            "shorter than 10 ms, with no state smoothing. Hatching marks "
            "uncertainty; blank marks unassessed time."
        )
        if shot == 201978:
            notes.append(
                "Here the sawtooth row shows only the four-state notation, "
                "with no present sawtooth events in this window; these physics "
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
            f"Magnetic lines at {lo:.0f}–{hi:.0f} kHz remain visible{timing}, "
            "but stay untagged because the CO2 AE detector is negative. "
            "The AE tags inherit the detector's timing, including its "
            "negative gaps; they do not imply that the magnetic lines disappear."
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
            f"the expert ELM interval (from {start:.0f} ms)."
        )
    if drawn.get("elm_hmode_conflicts_ms"):
        notes.append(
            "Expert ELM intervals overlap H-mode-detector absent time; "
            "sources disagree."
        )
    if drawn.get("ae_physical_review_caveat"):
        notes.append(drawn["ae_physical_review_caveat"])
    notes.append(
        "Open circles delimit expert spans containing many ELMs; downward "
        "triangles mark threshold D-alpha peaks."
    )
    return " ".join(notes)
