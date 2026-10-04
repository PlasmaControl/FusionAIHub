"""Radial relative-drop profiles for the reviewed shots (fix round 4, finding I5).

For each reviewed shot, relaxation events are taken from the rule's own records
(accepted crash points plus the periodic core-relaxation edges) inside the
expert-positive spans. At every event each ECE channel's relative change
(median over 0.3-1.5 ms after minus the same window before, over the before
value) is mapped to a nominal geometric radius rho = (R - R_axis) /
(R_LCFS,out - R_axis); rho < 0 is the high-field side. The profile of a central
sawtooth drops inside the inversion radius and rises just outside it. A drop
that grows outward with a flat core is an edge or off-axis relaxation. The
verified train shot 192148 supplies the sawtooth reference profile.

The verdict rule is written here, before any shot is read, and is not tuned:

* central drop: median relative change within |rho| < 0.15 is <= -0.05;
* outer rise: median change on the low-field side at rho 0.3-0.6 is >= +0.02;
* sawtooth-like: central drop and outer rise;
* edge-driven: no central drop, and the median change falls monotonically
  (within 0.02) outward over rho 0.3-0.9 to <= -0.15 at the outermost channel;
* otherwise indeterminate.

Cutoff is a separate question: the fraction of the expert-positive span that is
observable after the density and ECE-validity guards is reported beside it.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sawtooth_fix3_artifacts import save_plot, style
from sawtooth_physics import ECE_GEOMETRY_ARCHIVE, FS, OUTPUT, REVIEW, WORK, save_json

from labeler.config import Paths
from labeler.events.panels.ece_geometry import align_q
from labeler.sawtooth.geometry import load_radius_geometry

SHOTS = (186636, 189324, 190637, 186532)
REFERENCE_SHOT = 192148
PRE_MS = (-1.5, -0.3)
POST_MS = (0.3, 1.5)
MERGE_S = 0.005
CENTRAL_RHO = 0.15
CENTRAL_DROP = -0.05
OUTER_RISE = 0.02
EDGE_SLOPE_TOLERANCE = 0.02
EDGE_OUTERMOST = -0.15
OKABE = {"blue": "#0072B2", "orange": "#E69F00", "grey": "#666666", "red": "#D55E00"}


def load(work, shot):
    record = json.loads((work / "shots" / f"{shot}.json").read_text())
    signal = np.load(work / "signals" / f"{shot}.npz")
    return record, signal["t"], signal["y"].astype(np.float32), signal["observable"]


def geometry(work, shot, clock):
    radius, _ = load_radius_geometry(
        shot,
        clock,
        48,
        Paths.from_env(),
        archive_root=ECE_GEOMETRY_ARCHIVE,
        metadata_root=work / "geometry",
    )
    return radius


def signed_rho(radius, time_s):
    """Signed nominal rho per channel at one time; NaN where unmapped."""
    R = align_q([time_s], radius.time_s, radius.R_m)[:, 0]
    axis = align_q([time_s], radius.time_s, radius.axis_R_m[None])[0, 0]
    lcfs = align_q([time_s], radius.time_s, radius.lcfs_outer_R_m[None])[0, 0]
    return (R - axis) / (lcfs - axis), R


def merged(times, tolerance=MERGE_S):
    out = []
    for stamp in sorted(times):
        if out and stamp - out[-1][-1] <= tolerance:
            out[-1].append(stamp)
        else:
            out.append([stamp])
    return [float(np.median(group)) for group in out]


def event_times(record, spans, *, present_only=False):
    """Crash points and periodic core edges, merged, inside the given spans."""
    stamps = [
        p["time_s"]
        for p in record["crashes"]
        if not present_only or p["attrs"]["state"] == "present"
    ]
    if not present_only:
        test = record["absence_diagnostics"]["core_relaxation_test"]
        stamps += test.get("periodic_edge_times_s", [])
    inside = [t for t in stamps if any(a <= t <= b for a, b in spans)]
    return merged(inside)


def relative_changes(y, clock, times):
    """(events, channels) relative change across each event, NaN where unknown."""
    out = np.full((len(times), y.shape[0]), np.nan, dtype=np.float32)
    for row, stamp in enumerate(times):
        a = slice(
            *np.searchsorted(clock, [stamp + PRE_MS[0] / 1e3, stamp + PRE_MS[1] / 1e3])
        )
        b = slice(
            *np.searchsorted(
                clock, [stamp + POST_MS[0] / 1e3, stamp + POST_MS[1] / 1e3]
            )
        )
        if a.stop - a.start < 3 or b.stop - b.start < 3:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            before = np.nanmedian(y[:, a], axis=1)
            after = np.nanmedian(y[:, b], axis=1)
        good = np.isfinite(before) & np.isfinite(after) & (before > 0.05)
        out[row, good] = (after[good] - before[good]) / before[good]
    return out


def profile(work, shot, spans, *, present_only=False):
    record, clock, y, observable = load(work, shot)
    radius = geometry(work, shot, clock)
    times = event_times(record, spans, present_only=present_only)
    changes = relative_changes(y[:40], clock, times)
    rho = np.full((len(times), 40), np.nan)
    for row, stamp in enumerate(times):
        rho[row] = signed_rho(radius, stamp)[0][:40]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median_rho = np.nanmedian(rho, axis=0) if len(times) else np.full(40, np.nan)
        median = np.nanmedian(changes, axis=0) if len(times) else np.full(40, np.nan)
        q25 = np.nanpercentile(changes, 25, axis=0) if len(times) else median
        q75 = np.nanpercentile(changes, 75, axis=0) if len(times) else median
    return {
        "record": record,
        "clock": clock,
        "y": y,
        "observable": observable,
        "radius": radius,
        "times": times,
        "changes": changes,
        "rho": median_rho,
        "median": median,
        "q25": q25,
        "q75": q75,
    }


def verdict(rho, median):
    """The pre-registered rule; see the module docstring."""
    ok = np.isfinite(rho) & np.isfinite(median)
    central = ok & (np.abs(rho) < CENTRAL_RHO)
    lfs = ok & (rho >= 0.3) & (rho <= 0.6)
    outer = ok & (rho >= 0.3) & (rho <= 0.9)
    result = {
        "central_median_change": float(np.median(median[central]))
        if central.any()
        else None,
        "outer_rise_max_change_rho_0p3_0p6": float(np.max(median[lfs]))
        if lfs.any()
        else None,
        "outermost_median_change": None,
        "outermost_rho": None,
    }
    if outer.any():
        last = np.flatnonzero(outer)[np.argmax(rho[outer])]
        result["outermost_median_change"] = float(median[last])
        result["outermost_rho"] = float(rho[last])
    central_drop = (
        result["central_median_change"] is not None
        and result["central_median_change"] <= CENTRAL_DROP
    )
    outer_rise = (
        result["outer_rise_max_change_rho_0p3_0p6"] is not None
        and result["outer_rise_max_change_rho_0p3_0p6"] >= OUTER_RISE
    )
    ordered = np.argsort(rho[outer])
    outward = median[outer][ordered]
    monotone = bool(
        outward.size >= 3 and np.all(np.diff(outward) <= EDGE_SLOPE_TOLERANCE)
    )
    edge = (
        not central_drop
        and monotone
        and result["outermost_median_change"] is not None
        and result["outermost_median_change"] <= EDGE_OUTERMOST
    )
    result.update(
        central_drop=bool(central_drop),
        outer_rise=bool(outer_rise),
        monotone_outward_decline=monotone,
        verdict=(
            "sawtooth-like"
            if central_drop and outer_rise
            else "edge-driven"
            if edge
            else "indeterminate"
        ),
    )
    return result


def span_support(prof, spans):
    """Seconds of the expert-positive span, and how much remains observable."""
    clock, observable = prof["clock"], prof["observable"]
    inside = np.zeros(len(clock), dtype=bool)
    for a, b in spans:
        inside |= (clock >= a) & (clock <= b)
    dt = 1.0 / FS
    return {
        "expert_positive_s": float(inside.sum() * dt),
        "observable_s": float((inside & observable).sum() * dt),
        "observable_fraction": float((inside & observable).sum() / inside.sum())
        if inside.any()
        else None,
    }


def cutoff_profile(prof):
    """Median pre-event Te against R for the first and last event: the step."""
    clock, y, radius = prof["clock"], prof["y"], prof["radius"]
    rows = []
    for stamp in prof["times"][:: max(1, len(prof["times"]) // 3)][:3]:
        a = slice(*np.searchsorted(clock, [stamp - 0.0015, stamp - 0.0003]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            te = np.nanmedian(y[:40, a], axis=1)
        rows.append((stamp, align_q([stamp], radius.time_s, radius.R_m)[:40, 0], te))
    return rows


def figure(shot, prof, reference, result, destination):
    plt = style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), constrained_layout=True)
    ax = axes[0]
    ax.axhline(0, color="k", lw=0.5)
    order = np.argsort(reference["rho"])
    ax.fill_between(
        reference["rho"][order],
        reference["q25"][order],
        reference["q75"][order],
        color=OKABE["grey"],
        alpha=0.25,
        lw=0,
    )
    ax.plot(
        reference["rho"][order],
        reference["median"][order],
        color=OKABE["grey"],
        lw=1.0,
        label=f"reference sawtooth, {REFERENCE_SHOT}",
    )
    order = np.argsort(prof["rho"])
    ax.fill_between(
        prof["rho"][order],
        prof["q25"][order],
        prof["q75"][order],
        color=OKABE["blue"],
        alpha=0.25,
        lw=0,
    )
    ax.plot(
        prof["rho"][order],
        prof["median"][order],
        color=OKABE["blue"],
        lw=1.2,
        marker="o",
        ms=2.5,
        label=f"{shot} events (n={len(prof['times'])})",
    )
    ax.set_xlabel(r"nominal $\rho$ (negative: high-field side)")
    ax.set_ylabel("relative ECE change across event")
    ax.set_xlim(-1.0, 1.0)
    ax.legend(frameon=False, loc="lower left")
    ax = axes[1]
    colours = (OKABE["blue"], OKABE["orange"], OKABE["red"])
    for colour, (stamp, R, te) in zip(colours, cutoff_profile(prof), strict=False):
        ax.plot(R, te, color=colour, lw=1.0, marker=".", ms=3, label=f"{stamp:.3f} s")
    ax.axvline(
        float(np.nanmedian(prof["radius"].axis_R_m)),
        color="k",
        lw=0.5,
        ls=":",
    )
    ax.set_xlabel("channel major radius R (m)")
    ax.set_ylabel(r"$T_e$ before event (keV)")
    ax.legend(frameon=False, title="event", title_fontsize=7)
    return save_plot(fig, destination) | {"verdict": result["verdict"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--shots", nargs="+", type=int, default=list(SHOTS))
    args = parser.parse_args()
    review = pd.read_csv(REVIEW)
    reference_record = json.loads(
        (args.work / "shots" / f"{REFERENCE_SHOT}.json").read_text()
    )
    present = [(e["start_s"], e["end_s"]) for e in reference_record["intervals"]]
    reference = profile(args.work, REFERENCE_SHOT, present, present_only=True)
    reference_result = verdict(reference["rho"], reference["median"])
    report = {
        "rule": __doc__.split("The verdict rule", 1)[1]
        .split("Cutoff is", 1)[0]
        .strip(),
        "windows_ms": {"before": PRE_MS, "after": POST_MS},
        "reference": {
            "shot": REFERENCE_SHOT,
            "events": len(reference["times"]),
            **reference_result,
        },
        "shots": {},
    }
    for shot in args.shots:
        rows = review[(review.shot == shot) & (review.category == 1)]
        spans = [(r.t_start / 1000, r.t_end / 1000) for r in rows.itertuples()]
        source = "expert-positive spans"
        if not spans:
            record = json.loads((args.work / "shots" / f"{shot}.json").read_text())
            spans = [(e["start_s"], e["end_s"]) for e in record["intervals"]]
            source = "rule-present trains (no reviewed span)"
        prof = profile(args.work, shot, spans)
        result = verdict(prof["rho"], prof["median"])
        report["shots"][str(shot)] = {
            "event_source": source,
            "spans_s": spans,
            "events": len(prof["times"]),
            "event_times_s": prof["times"],
            "support": span_support(prof, spans),
            "state_seconds": prof["record"]["state_seconds"],
            "guard_accounting_samples": prof["record"]["absence_diagnostics"].get(
                "guard_accounting"
            ),
            "profile": {
                "rho": [None if not np.isfinite(v) else float(v) for v in prof["rho"]],
                "median_change": [
                    None if not np.isfinite(v) else float(v) for v in prof["median"]
                ],
            },
            **result,
            "figure": figure(
                shot,
                prof,
                reference,
                result,
                args.work / "figures" / f"radial_drop_{shot}",
            ),
        }
    save_json(OUTPUT / "radial_drop_profiles.json", report)
    print(
        json.dumps(
            {
                "reference": report["reference"]["verdict"],
                **{k: v["verdict"] for k, v in report["shots"].items()},
            }
        )
    )


if __name__ == "__main__":
    main()
