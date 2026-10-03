"""Read-only source selection for the TokEye interpreter figure.

CONFINEMENT_RUN_DIR defaults to runs/labeler/confinement/v1 in the checkout
containing LABELER_LABEL_TABLES. Both curated and fallback files are required
when needed; a missing file never silently substitutes the H-mode frame model.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
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
    return lf.Track(replace(spec, title="H-mode"), source, fallback, rows)


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


def sawtooth_track(paths: Paths, shot: int, source: Path, verified_times):
    """Keep physics interval categories; verified crash ticks are separate."""
    file, _, rows, _ = saw_source(source, shot)
    times = np.asarray(verified_times)
    spec = next(s for s in lf.TRACKS if s.key == mt.SAWTOOTH)
    name = "physics sawtooth states; independently ECE-verified crash ticks"
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
    common &= np.abs(ratio - 2) <= 0.1
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0
    result["support_ms"] = float(common.sum() * dt)
    if common.any():
        result.update(
            n1_median_khz=float(np.median(f1[common])),
            n2_median_khz=float(np.median(f2[common])),
        )
    return result


def sawtooth_caption(record: dict) -> str:
    """Summarise displayed physics states without equating a proxy to cutoff."""
    rows = record.get("state_intervals_ms", [])
    if not rows:
        return "Sawtooth unassessed."
    states = {r["state"] for r in rows}
    if len(states) == 1:
        text = f"Sawtooth {rows[0]['state']} throughout"
    else:
        first = rows[0]
        text = f"Sawtooth {first['state']} to {first['end_ms'] / 1000:.2f} s"
        if "uncertain" in states and first["state"] != "uncertain":
            text += "; uncertain intervals"
        if "present" in states and first["state"] != "present":
            text += "; present intervals"
        blanks = [r for r in rows if r["state"] == "unassessed"]
        if blanks and first["state"] != "unassessed":
            start = blanks[0]["start_ms"]
            duration = sum(r["end_ms"] - r["start_ms"] for r in blanks)
            remaining = rows[-1]["end_ms"] - start
            if duration >= 0.5 * remaining:
                text += f"; mostly unassessed from {start / 1000:.2f} s"
            else:
                text += "; intermittently unassessed"
    guard = record.get("density_guard") or {}
    if "unassessed" in states and guard.get("cutoff_proxy"):
        text += ", where a conservative density proxy limits ECE observability"
        if "bt_missing" in guard.get("status", "").lower():
            text += " (no Bt available)"
    return text + "."


def caption(shot: int, records: dict, drawn: dict) -> str:
    """Source-specific paper prose, without internal source identifiers."""
    sources = []
    for key, name in (
        (mt.AE, "AE"),
        (mt.NTM, "NTM"),
        (mt.SAWTOOTH, "sawtooth"),
        ("confinement", "regime"),
        ("edge_localized_mode", "ELMs"),
    ):
        if key == mt.SAWTOOTH:
            continue
        record = records.get(key)
        if record is None:
            description = "unassessed"
        elif record["tier"] == lf.SILVER:
            description = "expert"
        elif record["tier"] == lf.LEGACY:
            description = "imported"
        elif key == mt.AE:
            description = (
                "neural interferometer detector"
                if record["what"].startswith("ae-ours")
                else "interferometer frame detector"
            )
            if record["what"].startswith("ae-ours"):
                description += (
                    f" (p≥{AE_THRESHOLD}; training targets used TokEye's mask)"
                )
            elif record.get("decision_threshold") is not None:
                description += f" (p≥{record['decision_threshold']})"
        elif key == mt.NTM:
            description = "magnetic detector (unverified; shared inputs)"
        elif key == "confinement":
            description = "D-alpha detector"
            if drawn.get("regimes_shown") == []:
                description += " (uncertain here)"
        else:
            description = "detector"
        if key == "confinement" and record is not None:
            name = record["title"]
        sources.append(f"{name}: {description}")
    sentences = [
        (
            f"DIII-D shot {shot}. Raw bands normalised separately; "
            "TokEye's U-Net extracts coherent modes."
        ),
        "; ".join(sources) + ".",
        (
            "Pink ≥80 kHz: AE time/band overlap; n measured ≤30 kHz. "
            "NTM outlines here require dominant n=1 or 2."
        ),
    ]
    harmonic = drawn.get("harmonic_support", {})
    if harmonic.get("support_ms", 0) >= harmonic.get("minimum_support_ms", 50):
        step = harmonic.get("caption_frequency_step_khz", 1.0)
        frequency = round(harmonic["n2_median_khz"] / step) * step
        sentences.append(
            f"The n=2 ridge near {frequency:.0f} kHz is consistent with a second "
            "harmonic of n=1."
        )
    if records.get(mt.SAWTOOTH):
        sentences.append(sawtooth_caption(records[mt.SAWTOOTH]))
    if drawn.get("catalog_sawtooth_frame_model_shown") is False:
        sentences.append("Catalog generated sawtooth frame model not shown.")
    if drawn.get("late_untagged_high_frequency"):
        sentences.append(
            "Late 170–250 kHz lines stay untagged where AE detector is absent."
        )
    if drawn.get("sawtooth_strip_shown"):
        sentences.append("Ticks exclude crashes within 5 ms of D-alpha peaks.")
    sentences.append(
        "Hatching: uncertain; blank: unassessed/unobservable; "
        "circles: expert ELM interval (one span for many ELMs); "
        "triangles: D-alpha peaks from a threshold (not annotated)."
    )
    text = " ".join(sentences)
    if len(text.split()) > 150:
        raise ValueError(f"caption exceeds 150 words: {len(text.split())}")
    return text
