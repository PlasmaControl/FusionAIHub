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

AE_THRESHOLD = 0.7
NTM_THRESHOLD = 0.63
SAWTOOTH_THRESHOLD = 0.6
ELM_VETO_MS = 5.0
ECE_MATCH_MS = 3.0


def elm_category(row: lf.Row) -> int:
    """Expert crowd rows encode ELMing periods, not ordinary uncertainty."""
    if row.crowd == 1 and row.category in (PRESENT, UNCERTAIN):
        return PRESENT
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
    """Paper ae-ours probabilities, if stored and valid; fixed p>=0.7.

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
        "ae-ours (SELDnet-style), stored CO2 activity, p>=0.7",
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
            "ae-ours, CO2 activity predictions, p>=0.7",
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
    """The same physics source as the ticks; no old interval-model phase track."""
    file, _, rows, _ = saw_source(source, shot)
    times = np.asarray(verified_times)
    out = []
    for row in rows:
        # Intervals cannot promote unverified or ELM-coincident point events.
        if row.category == PRESENT:
            row = replace(row, category=UNCERTAIN)
        out.append(row)
    out.extend(lf.Row(float(t - 0.5), float(t + 0.5), PRESENT) for t in times)
    spec = next(s for s in lf.TRACKS if s.key == mt.SAWTOOTH)
    name = (
        "ECE-verified sawtooth crashes"
        if len(times)
        else "sawtooth detector candidates"
    )
    if not len(times):
        spec = replace(spec, title="sawtooth cand.")
    selected = lf.Source(lf.GENERATED, name, lambda p: file)
    return lf.Track(spec, selected, file, tuple(out))


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
    """Coincident n=1/n=2 ridges consistent with a near-15-kHz harmonic.

    This is a plotted-frequency inference, not independent island confirmation.
    Require a 2:1 match within 1.2 kHz and at least 50 ms of sampled support.
    """
    t, f = np.asarray(times), np.asarray(frequencies)
    result = {
        "frequency_tolerance_khz": 1.2,
        "minimum_support_ms": 50.0,
        "n1_band_khz": [6, 10],
        "n2_band_khz": [12, 18],
        "support_ms": 0.0,
    }
    if n_map is None:
        return result
    means, counts = [], []
    for mode in (1, 2):
        measured = mask & (np.asarray(n_map) == mode)
        lo, hi = (6, 10) if mode == 1 else (12, 18)
        measured &= ((f >= lo) & (f <= hi))[:, None]
        count = measured.sum(axis=0)
        counts.append(count)
        means.append((measured * f[:, None]).sum(axis=0) / np.maximum(count, 1))
    f1, f2 = means
    common = (counts[0] > 0) & (counts[1] > 0) & (f2 >= 12) & (f2 <= 18)
    common &= np.abs(f2 - 2 * f1) <= 1.2
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0
    result["support_ms"] = float(common.sum() * dt)
    if common.any():
        result.update(
            n1_median_khz=float(np.median(f1[common])),
            n2_median_khz=float(np.median(f2[common])),
        )
    return result


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
        record = records.get(key)
        if record is None:
            description = "unassessed"
        elif record["tier"] == lf.SILVER:
            description = "expert review"
        elif record["tier"] == lf.LEGACY:
            description = "imported intervals"
        elif key == mt.AE:
            description = (
                "neural interferometer detector"
                if record["what"].startswith("ae-ours")
                else "interferometer frame detector"
            )
        elif key == mt.NTM:
            description = "magnetic detector"
        elif key == mt.SAWTOOTH:
            description = (
                "ECE-verified crashes"
                if drawn["sawtooth_strip_shown"]
                else "physics detector candidates"
            )
        elif key == "confinement":
            description = "D-alpha transition detector"
            if drawn.get("regimes_shown") == []:
                description += " (uncertain here)"
        else:
            description = "detector"
        if key == "confinement" and record is not None:
            name = record["title"]
        sources.append(f"{name}: {description}")
    sentences = [
        f"DIII-D shot {shot}: raw signals, TokEye processing and aligned labels.",
        (
            "The frequency resolution changes at 55 kHz; measured toroidal n is shown "
            "below 30 kHz, with unmeasured mask pixels white."
        ),
        (
            "Pink AE highlights above 60 kHz and orange NTM outlines below 60 kHz "
            "show temporal coincidence, not independent identification."
        ),
        "; ".join(sources) + ".",
        "Hatching means uncertainty; blank means unassessed.",
    ]
    elm = records.get("edge_localized_mode")
    if elm and elm["tier"] == lf.SILVER and drawn.get("elm_crowd_spans_ms"):
        sentences.append("Marked solid ELM bars are expert-reviewed periods.")
    if drawn.get("elm_peaks_in_label", 0):
        sentences.append("Triangles mark D-alpha peaks at the figure threshold.")
    ntm = records.get(mt.NTM)
    if ntm and ntm["tier"] == lf.GENERATED:
        bars = ntm.get("primary_bars") or {}
        failed = bars.get("all") is False or any(v is False for v in bars.values())
        sentences.append(
            "The NTM detector shares the displayed magnetic inputs"
            + (" and failed its primary acceptance check." if failed else ".")
        )
    if drawn.get("n2_island_harmonic"):
        sentences.append(
            "Under an island interpretation, outlined n=2 near 15 kHz is a harmonic "
            "of the same island."
        )
    if drawn["sawtooth_strip_shown"]:
        sentences.append("Ticks exclude crashes within 5 ms of D-alpha peaks.")
    text = " ".join(sentences)
    if len(text.split()) > 150:
        raise ValueError(f"caption exceeds 150 words: {len(text.split())}")
    return text
