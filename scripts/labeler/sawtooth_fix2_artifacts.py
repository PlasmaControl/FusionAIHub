"""Audit state changes, queue blind review and draw a column-width example."""

from __future__ import annotations

import argparse
import itertools
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sawtooth_physics import (
    ECE_GEOMETRY_ARCHIVE,
    OUTPUT,
    READER_POLICY,
    REPO,
    REVIEW,
    WORK,
    records_at,
    save_json,
)

from labeler.sawtooth.metrics import spans_at
from labeler.sawtooth.physics import Rule, core_relaxation_phases
from labeler.sawtooth.preprocessing import mask_spans, state_spans

STATES = ("present", "absent", "uncertain", "unassessed")
COLORS = {
    "present": "#009E73",
    "absent": "#FFFFFF",
    "uncertain": "#E69F00",
    "unassessed": "#999999",
}


def short_holes(record, *, candidates=False):
    """Absent portions between candidate-support spans separated by <300 ms."""
    supports = (
        [s for key in ("intervals", "uncertain_intervals") for s in record.get(key, [])]
        if candidates
        else [s for s in record.get("states", []) if s["state"] in STATES[::2]]
    )
    merged = []
    for span in sorted(supports, key=lambda s: s["start_s"]):
        if merged and span["start_s"] <= merged[-1]["end_s"]:
            merged[-1]["end_s"] = max(merged[-1]["end_s"], span["end_s"])
        else:
            merged.append({"start_s": span["start_s"], "end_s": span["end_s"]})
    holes = [
        (a["end_s"], b["start_s"])
        for a, b in itertools.pairwise(merged)
        if 0 < b["start_s"] - a["end_s"] < 0.3
    ]
    absent = [s for s in record.get("states", []) if s["state"] == "absent"]
    seconds = sum(
        max(0, min(end, s["end_s"]) - max(start, s["start_s"]))
        for start, end in holes
        for s in absent
        if s["start_s"] < end and s["end_s"] > start
    )
    return {"holes": len(holes), "absent_seconds": seconds}


def subtract_spans(spans, vetoes):
    remaining = [tuple(s) for s in spans]
    for veto in vetoes:
        lo, hi = veto["start_s"], veto["end_s"]
        pieces = []
        for a, b in remaining:
            if b <= lo or a >= hi:
                pieces.append((a, b))
            else:
                if a < lo:
                    pieces.append((a, lo))
                if b > hi:
                    pieces.append((hi, b))
        remaining = pieces
    return remaining


def refine_records(args):
    """Apply the unbounded phase guard to saved edges without reading corpus."""
    cohort = set(pd.read_csv(REPO / "data/events/catalog/cohort.csv").shot)
    paths = sorted((args.work / "shots").glob("*.json"))
    if args.cohort_only:
        paths = [p for p in paths if int(p.stem) in cohort]
    counts = Counter()
    changed = []
    for path in paths:
        record = json.loads(path.read_text())
        prior_policy = record["reader_policy"]
        record["reader_policy"] = READER_POLICY
        if "error" in record:
            save_json(path, record)
            counts["failed_reads"] += 1
            continue
        counts["processed_shots"] += 1
        if prior_policy == READER_POLICY and "phase_refinement" in record:
            counts["already_refined_shots"] += 1
            refinement = record["phase_refinement"]
            target_changes = refinement["cached_assessment_cells_changed"]
            if target_changes is not None:
                counts["cached_input_arrays_retained"] += 1
                counts["cached_target_cells_changed"] += target_changes
            if refinement["removed_absent_seconds"] > 1e-8 or target_changes:
                changed.append({"shot": record["shot"], **refinement})
            continue
        rule = Rule(**record["rule"])
        diagnostics = record["absence_diagnostics"]
        test = diagnostics["core_relaxation_test"]
        phases = core_relaxation_phases(
            test["ambiguous_edge_times_s"], record["observable_spans"], rule
        )
        test.update(
            phase_spans=phases,
            phase_grouping={
                "edge_source": "ambiguous_edge_times_s",
                "minimum_train": rule.minimum_train,
                "period_ratio": rule.period_ratio,
                "minimum_period_ms": None,
                "maximum_period_ms": None,
                "context_radius_ms": 1.5 * rule.maximum_period_ms,
            },
            positive_train_period_bounds={
                "minimum_period_ms": rule.minimum_period_ms,
                "maximum_period_ms": rule.maximum_period_ms,
            },
        )
        old_seconds = dict(record["state_seconds"])
        old_absence = list(record["absent_evidence_spans"])
        new_absence = subtract_spans(old_absence, phases)
        signal = args.work / "signals" / f"{record['shot']}.npz"
        target_changes = None
        if signal.exists():
            with np.load(signal) as stored:
                arrays = {key: stored[key] for key in stored.files}
            t = arrays["t"]
            old_assessed = arrays["assessed"].copy()
            absence = spans_at(t, new_absence)
            states, assessed = state_spans(
                t,
                arrays["observable"],
                [(s["start_s"], s["end_s"]) for s in record["intervals"]],
                [(s["start_s"], s["end_s"]) for s in record["uncertain_intervals"]],
                absent=absence,
            )
            target_changes = int(np.count_nonzero(old_assessed != assessed))
            assert np.all(~assessed | old_assessed)
            arrays["assessed"] = assessed
            arrays["absent_evidence"] = absence
            if target_changes:
                np.savez_compressed(signal, **arrays)
            record["absent_evidence_spans"] = mask_spans(t, absence)
            record["assessed_spans"] = mask_spans(t, assessed)
            diagnostics["reason_samples"]["tested_absence"] = int(absence.sum())
            diagnostics["reason_samples"]["core_relaxation_phase"] = int(
                spans_at(t, [(p["start_s"], p["end_s"]) for p in phases]).sum()
            )
            counts["cached_input_arrays_retained"] += 1
            counts["cached_target_cells_changed"] += target_changes
        else:
            states = []
            for original in record["states"]:
                if original["state"] != "absent":
                    states.append(dict(original))
                    continue
                a, b = original["start_s"], original["end_s"]
                cuts = sorted(
                    {a, b}
                    | {
                        boundary
                        for phase in phases
                        for boundary in (phase["start_s"], phase["end_s"])
                        if a < boundary < b
                    }
                )
                for lo, hi in itertools.pairwise(cuts):
                    middle = (lo + hi) / 2
                    vetoed = any(p["start_s"] <= middle < p["end_s"] for p in phases)
                    states.append(
                        {
                            "start_s": lo,
                            "end_s": hi,
                            "state": "uncertain" if vetoed else "absent",
                        }
                    )
            record["absent_evidence_spans"] = new_absence
            record["assessed_spans"] = [
                (s["start_s"], s["end_s"])
                for s in states
                if s["state"] in ("present", "absent")
            ]
            if new_absence != old_absence:
                diagnostics.setdefault(
                    "reason_samples_before_phase_refinement",
                    dict(diagnostics["reason_samples"]),
                )
                diagnostics["reason_samples"]["tested_absence"] = None
            diagnostics["reason_samples"]["core_relaxation_phase"] = None
        merged = []
        for span in states:
            if (
                merged
                and merged[-1]["state"] == span["state"]
                and merged[-1]["end_s"] == span["start_s"]
            ):
                merged[-1]["end_s"] = span["end_s"]
            else:
                merged.append(dict(span))
        record["states"] = merged
        record["state_seconds"] = {
            state: sum(s["end_s"] - s["start_s"] for s in merged if s["state"] == state)
            for state in STATES
        }
        assert record["state_seconds"]["present"] == old_seconds["present"]
        assert record["state_seconds"]["unassessed"] == old_seconds["unassessed"]
        assert record["state_seconds"]["absent"] <= old_seconds["absent"] + 1e-8
        removed = old_seconds["absent"] - record["state_seconds"]["absent"]
        refinement = {
            "source": "saved significant per-channel negative core edges",
            "prior_reader_policy": prior_policy,
            "removed_absent_seconds": removed,
            "cached_assessment_cells_changed": target_changes,
            "input_values_unchanged": True,
            "no_corpus_reread": True,
        }
        record["phase_refinement"] = refinement
        save_json(path, record)
        if removed > 1e-8 or target_changes:
            changed.append({"shot": record["shot"], **refinement})
        if counts["processed_shots"] % 2000 == 0:
            print(dict(counts), flush=True)
    save_json(
        args.output
        / (
            "cohort_phase_refinement.json"
            if args.cohort_only
            else "population_phase_refinement.json"
        ),
        {
            "counts": dict(counts),
            "changed_shots": changed,
            "validation_status": "unvalidated conservative absence refinement",
            "source_records": str(args.work / "shots"),
            "reader_policy": READER_POLICY,
            "count_scope": "cumulative saved refinements, including prior cohort pass",
        },
    )
    print({"counts": dict(counts), "changed_shots": len(changed)}, flush=True)


def audit(args):
    cohort = pd.read_csv(REPO / "data/events/catalog/cohort.csv")
    current_shots = [int(p.stem) for p in (args.work / "shots").glob("*.json")]
    prior_shots = [int(p.stem) for p in (args.previous / "shots").glob("*.json")]
    if args.cohort_only:
        current_shots = sorted(set(current_shots) & set(cohort.shot))
        prior_shots = sorted(set(prior_shots) & set(cohort.shot))
    current = {r["shot"]: r for r in records_at(args.work, current_shots)}
    prior = {r["shot"]: r for r in records_at(args.previous, prior_shots)}
    answer = {
        "previous": str(args.previous),
        "current": str(args.work),
        "hole_definition": (
            "strictly positive gap <300ms between consecutive canonical present "
            "or uncertain spans; sum only absent portions inside each gap"
        ),
        "validation_status": "unvalidated research labels; no blind crash truth",
        "scopes": {},
    }
    scopes = [("cohort", set(cohort.shot))]
    if not args.cohort_only:
        scopes.append(("population", set(current) & set(prior)))
    for name, shotset in scopes:
        shots = sorted(
            s
            for s in shotset & set(current) & set(prior)
            if "error" not in current[s] and "error" not in prior[s]
        )
        result = {"shots": shots, "shot_count": len(shots)}
        for label, records in (("before", prior), ("after", current)):
            holes = [short_holes(records[s]) for s in shots]
            candidate_holes = [short_holes(records[s], candidates=True) for s in shots]
            result[label] = {
                "state_seconds": {
                    state: sum(records[s]["state_seconds"][state] for s in shots)
                    for state in STATES
                },
                "short_holes": sum(h["holes"] for h in holes),
                "absent_in_short_holes_seconds": sum(
                    h["absent_seconds"] for h in holes
                ),
                "candidate_support_short_holes": sum(
                    h["holes"] for h in candidate_holes
                ),
                "absent_in_candidate_support_holes_seconds": sum(
                    h["absent_seconds"] for h in candidate_holes
                ),
            }
        answer["scopes"][name] = result
    save_json(args.output / "state_transition_audit.json", answer)
    verified = Counter()
    for shot, record in current.items():
        if "error" in record:
            verified["failed_reads"] += 1
            continue
        verified["processed_shots"] += 1
        spans = record["states"]
        assert all(a["end_s"] == b["start_s"] for a, b in itertools.pairwise(spans))
        diagnostics = record["absence_diagnostics"]
        test = diagnostics["core_relaxation_test"]
        protected = (
            diagnostics["profile_passing_candidate_times_s"]
            + test["ambiguous_edge_times_s"]
        )
        horizon = diagnostics["context_radius_ms"] / 1000
        for span in spans:
            if span["state"] != "absent":
                continue
            assert any(
                a <= span["start_s"] and span["end_s"] <= b
                for a, b in record["absent_evidence_spans"]
            ), shot
            assert not any(
                span["start_s"] - horizon <= point <= span["end_s"] - 0.0001 + horizon
                for point in protected
            ), shot
            assert not any(
                span["start_s"] < phase["end_s"]
                and span["end_s"] - 0.0001 >= phase["start_s"]
                for phase in test["phase_spans"]
            ), shot
            verified["tested_absent_spans"] += 1
        for point in record["crashes"]:
            assert (
                "auxiliary_not_corroborated"
                not in point["attrs"]["uncertainty_reasons"]
            )
            attrs = point["attrs"]
            if attrs.get("neutron_drop_corroboration"):
                assert attrs["nbi_on"] is True
                assert attrs["neutron_drop_absolute"] >= (
                    attrs["neutron_noise_k"] * attrs["neutron_window_noise"]
                )
                verified["noise_qualified_neutron_crashes"] += 1
        from sawtooth_physics import export_rows

        for _, _, crowd, _, attrs in export_rows(record):
            if crowd and attrs["state"] == "present":
                assert all(
                    key in attrs
                    for key in (
                        "train_id",
                        "period_ms",
                        "inversion_channel",
                        "inversion_R_m",
                    )
                )
                verified["metadata_present_spans"] += 1
    save_json(
        args.output / "policy_audit.json",
        {
            "counts": dict(verified),
            "assertions_passed": True,
            "validation_status": "implementation audit, no blind physical validation",
            "rule": current[next(s for s in current if "error" not in current[s])][
                "rule"
            ],
            "source_records": str(args.work / "shots"),
            "boundary_policy": (
                "test sampled bins; a half-open span's end extends one 10kHz "
                "sample beyond its final sample"
            ),
        },
    )
    print({k: v["after"] for k, v in answer["scopes"].items()}, flush=True)


def queue(args):
    cohort = pd.read_csv(REPO / "data/events/catalog/cohort.csv")
    experts = set(pd.read_csv(REVIEW).shot)
    eligible = set(cohort.loc[cohort.split == "val", "shot"]) - experts
    records = {r["shot"]: r for r in records_at(args.work, eligible)}
    regime_source = (
        Path(__import__("os").environ["LABELER_LABEL_TABLES"])
        / "confinement/raw/Jalal_28042024_confinement_regime_shotlist.csv"
    )
    regimes = pd.read_csv(regime_source)
    options = []
    for shot in sorted(eligible):
        rec = records.get(shot, {})
        if "error" in rec or not rec.get("crashes"):
            continue
        times = np.array([r["time_s"] for r in rec["crashes"]])
        regime_rows = regimes.loc[regimes.Shot == shot].to_dict("records")
        if not regime_rows:
            metadata = cohort.set_index("shot").loc[shot]
            beam, ech = metadata.pbeam_max_mw, metadata.pech_max_mw
            fraction = ech / (beam + ech) if beam + ech > 0 else 0
            heating = (
                "beam-dominant"
                if fraction < 0.1
                else "mixed-heating"
                if fraction < 0.5
                else "ECH-dominant"
            )
            periods = [r["attrs"].get("period_ms") for r in rec["crashes"]]
            periods = [p for p in periods if p is not None]
            period = "short-period" if np.median(periods) < 50 else "long-period"
            regime_rows = [
                {
                    "Confinement Start Time (ms)": rec["window_s"][0] * 1000,
                    "Confinement Stop Time (ms)": rec["window_s"][1] * 1000,
                    "regime_proxy": f"{heating}/{period}",
                }
            ]
        for row in regime_rows:
            start = row["Confinement Start Time (ms)"] / 1000
            stop = row["Confinement Stop Time (ms)"] / 1000
            start = max(start, rec["window_s"][0])
            stop = min(stop, rec["window_s"][1])
            inside = times[(times >= start + 0.15) & (times <= stop - 0.15)]
            if stop - start < 0.3 or not len(inside):
                continue
            active = (
                [row["regime_proxy"]]
                if "regime_proxy" in row
                else [r for r in ("L", "H", "QH", "WP") if row[r] == 1]
            )
            if len(active) != 1:
                continue
            center = float(inside[len(inside) // 2])
            lo = max(start, center - 0.3)
            hi = min(stop, lo + 0.6)
            lo = max(start, hi - 0.6)
            options.append(
                {
                    "shot": int(shot),
                    "window": f"{lo * 1000:.1f}:{hi * 1000:.1f} ms",
                    "why": (
                        f"held-out val; {active[0]} regime proxy; H/L unknown; "
                        "predicted crash support; independent timing unvalidated"
                    ),
                    "regime": active[0],
                    "window_s": [lo, hi],
                    "predicted_crashes_in_window": int(
                        ((times >= lo) & (times <= hi)).sum()
                    ),
                }
            )
    # Round-robin regime strata; one window per shot; never use a test shot.
    chosen, used = [], set()
    by_regime = {
        r: [o for o in options if o["regime"] == r]
        for r in sorted({o["regime"] for o in options})
    }
    while len(chosen) < 15:
        added = False
        for choices in by_regime.values():
            candidates = [o for o in choices if o["shot"] not in used]
            if candidates:
                selected = candidates[0]
                chosen.append(selected)
                used.add(selected["shot"])
                added = True
                if len(chosen) == 15:
                    break
        if not added:
            raise ValueError("fewer than 15 nonexpert val shots with regime/crashes")
    destination = REPO / "data/events/sawtooth_oscillation/review/crash_time_queue.csv"
    pd.DataFrame(chosen)[["shot", "window", "why"]].to_csv(destination, index=False)
    save_json(
        args.output / "crash_time_queue.json",
        {
            "queue": str(destination.relative_to(REPO)),
            "regime_source": str(regime_source),
            "regime_author": "Jalal Butt",
            "confinement_regime_coverage": (
                "Jalal Butt table has no fixed-val shots; H/L/QH/WP unknown"
            ),
            "fallback_stratification": (
                "cohort peak ECH/(NBI+ECH): <0.1 beam-dominant, <0.5 mixed, "
                "otherwise ECH-dominant; median predicted period <50ms short "
                "otherwise long. Heating and period proxies, not H/L labels."
            ),
            "selection": "deterministic regime round-robin, one window per val shot",
            "split": "val",
            "excluded_expert_shots": sorted(int(s) for s in experts),
            "strata": dict(Counter(r["regime"] for r in chosen)),
            "rows": chosen,
            "annotation_status": "pending owner; unvalidated",
            "reviewer_display": "shot/window only, hide why and all predictions",
        },
    )
    print({"queue_shots": sorted(used), "strata": Counter(r["regime"] for r in chosen)})


def figure(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    cohort = pd.read_csv(REPO / "data/events/catalog/cohort.csv")
    allowed = set(cohort.loc[cohort.split.isin(["train", "val"]), "shot"])
    experts = set(pd.read_csv(REVIEW).shot)
    if args.shot not in allowed - experts:
        raise ValueError("example must use a nonexpert train/val shot")
    record = records_at(args.work, [args.shot])[0]
    with np.load(args.work / "signals" / f"{args.shot}.npz") as signal:
        t, y = signal["t"], signal["y"].astype(float)
    crashes = record["crashes"]
    if not crashes:
        raise ValueError("example needs a predicted crash train")
    points = [r for r in crashes if r["attrs"].get("state") == "present"] or crashes
    selected = points[len(points) // 2]
    center = selected["time_s"]
    lo, hi = max(t[0], center - 0.15), min(t[-1], center + 0.15)
    core = selected["attrs"]["central_channel"]
    attrs = selected["attrs"]
    outer = int(attrs["rise_start"])
    from labeler.config import Paths
    from labeler.sawtooth.geometry import load_radius_geometry

    radius, geometry = load_radius_geometry(
        args.shot, t, len(y), Paths.from_env(), archive_root=ECE_GEOMETRY_ARCHIVE
    )
    labels = [f"Core ch {core}", f"Outer ch {outer}"]
    if radius is not None:
        near = (t >= lo) & (t <= hi)
        radii = np.nanmedian(radius.R_m[:, near], axis=1)
        labels = [f"Core R={radii[core]:.2f} m", f"Outer R={radii[outer]:.2f} m"]
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(3.25, 2.45), sharex=True)
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.17, top=0.77, hspace=0.12)
    near = (t >= lo) & (t <= hi)
    trace_handles = []
    for axis, channel, color, label in zip(
        axes, (core, outer), ("#0072B2", "#D55E00"), labels, strict=True
    ):
        for span in record["states"]:
            a, b = max(lo, span["start_s"]), min(hi, span["end_s"])
            if b > a:
                axis.axvspan(
                    a * 1000,
                    b * 1000,
                    color=COLORS[span["state"]],
                    alpha=0.2,
                    linewidth=0,
                )
        (line,) = axis.plot(
            t[near] * 1000, y[channel, near], color=color, lw=0.65, label=label
        )
        trace_handles.append(line)
        for point in crashes:
            if lo <= point["time_s"] <= hi:
                axis.axvline(point["time_s"] * 1000, color="0.15", ls=":", lw=0.6)
        axis.set_ylabel("Te (keV)")
        axis.grid(axis="y", color="0.9", lw=0.3)
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].set(xlabel="Time (ms)", xlim=(lo * 1000, hi * 1000))
    axes[-1].xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    fig.legend(
        handles=trace_handles
        + [Line2D([], [], color="0.15", ls=":", lw=0.6, label="Crash candidate")],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        handlelength=1.1,
        columnspacing=0.8,
        labelspacing=0.2,
    )
    fig.legend(
        handles=[
            Patch(facecolor=COLORS[s], edgecolor="0.5", alpha=0.4, label=s.capitalize())
            for s in STATES
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.855),
        ncol=4,
        frameon=False,
        handlelength=0.8,
        columnspacing=0.65,
    )
    destination = args.work / "figures" / "sawtooth_example"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination.with_suffix(".pdf"))
    fig.savefig(destination.with_suffix(".png"), dpi=150)
    plt.close(fig)
    save_json(
        args.output / "paper_example.json",
        {
            "shot": args.shot,
            "split": cohort.set_index("shot").loc[args.shot, "split"],
            "window_s": [float(lo), float(hi)],
            "core_channel": core,
            "outer_channel": outer,
            "trace_labels": labels,
            "radius_geometry": geometry,
            "width_inches": 3.25,
            "minimum_font_points": 7,
            "png_dpi": 150,
            "title": None,
            "pdf": str(destination.with_suffix(".pdf")),
            "png": str(destination.with_suffix(".png")),
            "png_inspected": False,
            "claim": "unvalidated crash candidates and nominal resonance geometry",
        },
    )
    print(str(destination.with_suffix(".png")), flush=True)


def verification(args):
    logs = {}
    for name in ("tests", "lint", "format"):
        path = args.work / f"{name}.log"
        content = path.read_text()
        logs[name] = {"path": str(path), "tail": content.splitlines()[-12:]}
    passed = re.search(r"(\d+) passed", "\n".join(logs["tests"]["tail"]))
    assert passed and "All checks passed!" in "\n".join(logs["lint"]["tail"])
    assert "already formatted" in "\n".join(logs["format"]["tail"])
    save_json(
        args.output / "verification.json",
        {
            "covering_tests_passed": int(passed[1]),
            "logs": logs,
            "checks_passed": True,
            "test_files": [
                f"tests/labeler/test_sawtooth_{name}.py"
                for name in (
                    "physics",
                    "preprocessing",
                    "fix",
                    "benchmark",
                    "classification",
                    "masked_metrics",
                    "shot_geometry",
                    "geometry",
                )
            ],
            "test_command": (
                "bash $LABELER_ROOT/scratch/bin/pt.sh $PWD <covering files> "
                "-q -p no:cacheprovider"
            ),
            "lint_command": "pixi run --frozen --no-install <main manifest> "
            "-e labelmaker ruff check --extend-select E501 <changed .py>",
            "format_command": "pixi run --frozen --no-install <main manifest> "
            "-e labelmaker ruff format --check <new .py>",
        },
    )


def report(args):
    def read(name):
        return json.loads((args.output / name).read_text())

    def source(name, field=""):
        suffix = f": `{field}`" if field else ""
        return f"Source: `{args.output / name}`{suffix}."

    def ci(value, interval):
        return f"{value:.4f} [{interval[0]:.4f}, {interval[1]:.4f}]"

    benchmark = read("benchmark.json")["Tokamak-SI"]
    audit_record = read("state_transition_audit.json")
    policy = read("policy_audit.json")
    queue_record = read("crash_time_queue.json")
    example = read("paper_example.json")
    checks = read("verification.json")
    lines = [
        "## Fix round 2",
        "",
        f"Status: DONE_WITH_CONCERNS. Commits: `{args.commit_range}`.",
        "",
        (
            "All physical accuracy, presence/absence and radial claims remain "
            "unvalidated. The owner is away; no blind crash annotation was performed, "
            "and no model is recommended as latest or stable."
        ),
        "",
        "### Implementation and data",
        "",
        (
            "`physics.py` now requires explicit candidate-free, noise-resolved core "
            "relaxation evidence for absence, preserves ambiguous gaps, chooses the "
            "adjacent gain block, and records positive neutron/Mirnov evidence per "
            "crash. The relaxation check protects the strongest physical per-channel "
            "negative edge so channel averaging cannot erase localized crashes. "
            "Stable negative-edge phases have no period bounds and cannot cross "
            "observability gaps. Saved records were refined from recorded edges "
            "with input values retained; both models were retrained after the "
            "cohort assessment targets changed. "
            "Neutron evidence requires measured window noise and NBI-on support."
        ),
        "",
        (
            "`geometry.py` and the physics driver derive the core per shot, mask "
            "implausible Te, load same-shot archived RF metadata, map nominal second "
            "harmonic R when possible and preserve nullable train/inversion metadata "
            "on present exports. Frequencies come specifically from "
            "`/scratch/gpfs/nc1514/omnimode/data/raw/<shot>.h5:/ecegeom/FREQ`, "
            "ELECTRONS `\\ECE::TOP.SETUP.FREQ`, using the documented GHz convention. "
            "No RF vector is transferred between shots. Other shots use a hottest "
            "physical-channel proxy and null radii."
        ),
        "",
        source("geometry_metadata_audit.json"),
        "",
        (
            "The frozen rule's thresholds and provenance are in `freeze.json`; "
            "development uses the original training-only selection. The fixed cohort "
            "splits and every shot ID are in `data_summary.json` and "
            "`split_manifest.json`. Neither fixed val, test nor anchored expert shots "
            "select training weights or operating thresholds. CSV labels remain "
            "untracked; corpus and production stores were read only."
        ),
        "",
        source("freeze.json"),
        source("data_summary.json"),
        source("split_manifest.json"),
        source("cohort_phase_refinement.json"),
        source("population_phase_refinement.json"),
    ]
    refinement = read("population_phase_refinement.json")
    removed_absent = sum(
        row["removed_absent_seconds"] for row in refinement["changed_shots"]
    )
    lines += [
        "",
        (
            "The saved-edge phase refinement removed "
            f"{removed_absent:.4f} "
            f"absent seconds on {len(refinement['changed_shots'])} population shots. "
            "The cumulative cohort target change is "
            f"{refinement['counts']['cached_target_cells_changed']:,} cells; "
            "cached input arrays were retained."
        ),
        "",
        source("population_phase_refinement.json", "counts; changed_shots"),
    ]
    for scope in ("cohort", "population"):
        record = read(f"{scope}_labels.json")
        comparison = record["q1_major_radius_comparison"]
        successful = len(record["processed_shots"])
        efit_count = record["q_sources"].get("EFIT01", 0)
        lines += [
            "",
            (
                f"{scope.capitalize()}: requested {record['requested_count']} files; "
                f"{len(record['processed_shots'])} successful records and "
                f"{len(record['errors'])} read failures; "
                f"{record['crashes']} diagnostic "
                f"crash candidates. EFIT01 is available on "
                f"{efit_count}/{successful} successful shots. "
                "The frozen rule is shared; evidence coverage differs."
            ),
            "",
            (
                f"Nominal geometry counts: `{record['radius_geometry_counts']}`. "
                f"Paired inversion/q=1 candidates: {comparison['comparable_crashes']}; "
                f"median R difference {comparison.get('median_difference_m')} m, "
                "maximum absolute difference "
                f"{comparison.get('maximum_absolute_difference_m')} m. This is an "
                "unvalidated nominal resonance/EFIT comparison, not calibrated "
                "radial validation."
            ),
            "",
            source(f"{scope}_labels.json"),
        ]
    population = read("population_labels.json")
    previous_population = json.loads(
        (args.output.parent / "fix/population_labels.json").read_text()
    )
    excluded = sorted(
        set(previous_population["processed_shots"]) - set(population["processed_shots"])
    )
    lines += [
        "",
        (
            "Previously processed shots excluded after the physical channel screen: "
            f"`{[(s, population['errors'][str(s)]) for s in excluded]}`. Their "
            "missing coherent core supplies no positive or negative training truth."
        ),
        "",
        source("population_labels.json", "errors"),
    ]
    lines += [
        "",
        "### State seconds and holes",
        "",
        (
            "The before/after comparison uses exactly the same successfully read "
            "shots in each scope. State holes are <300 ms gaps between canonical "
            "present/uncertain support; candidate holes instead separate merged "
            "detected candidate spans. State holes can include explicitly tested "
            "quiet islands flanked by context/noise-limited uncertainty. Both are "
            "reported to avoid conflating these definitions."
        ),
        "",
        (
            "| Scope / run | Present s | Absent s | Uncertain s | Unassessed s | "
            "State holes / absent s | Candidate holes / absent s |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scope, comparison in audit_record["scopes"].items():
        for phase in ("before", "after"):
            row = comparison[phase]
            state_values = " | ".join(f"{row['state_seconds'][s]:.4f}" for s in STATES)
            lines.append(
                f"| {scope} / {phase} | {state_values} | {row['short_holes']} / "
                f"{row['absent_in_short_holes_seconds']:.4f} | "
                f"{row['candidate_support_short_holes']} / "
                f"{row['absent_in_candidate_support_holes_seconds']:.4f} |"
            )
    lines += [
        "",
        source("state_transition_audit.json", "scopes; shot lists"),
        "",
        (
            f"Policy assertions passed: `{policy['counts']}`. These verify "
            "implementation invariants, not physical accuracy."
        ),
        "",
        source("policy_audit.json"),
        "",
        "### Models and held-out metrics",
        "",
        (
            "Both architectures use identical observable inputs during fitting and "
            "inference. Assessment masks affect only losses and scoring. Both use "
            "crash-centred, period-class-balanced sampling; stopping uses uncapped "
            "inner-selection patience. Outer-fold metrics cover the fixed training "
            "cohort, conditioned on conservative algorithm-assessed support. "
            "Each CI uses shot bootstrap; fold records retain the exact grids, "
            "thresholds, class boundaries, stopping histories, shot IDs and runtime."
        ),
        "",
        "| Model | Crash F1 at ±2 ms [95% CI] | Presence F1 [95% CI] |",
        "|---|---:|---:|",
    ]
    for model, record in benchmark.items():
        values = record["crash_tolerance_2ms"]
        lines.append(
            f"| {model} | {ci(values['crash']['f1'], values['ci95']['crash_f1'])} | "
            f"{ci(values['presence']['f1'], values['ci95']['presence_f1'])} |"
        )
        prediction_manifest = read(f"{model}_queue_predictions.json")
        assert prediction_manifest["shots"] == sorted(
            r["shot"] for r in queue_record["rows"]
        )
    hl3 = benchmark["saw-hl3"]
    majority = hl3["three_class_majority_baseline"]
    hl3_accuracy = ci(
        hl3["three_class_window_accuracy"], hl3["three_class_accuracy_ci95"]
    )
    hl3_macro_f1 = ci(hl3["three_class_macro_f1"], hl3["three_class_macro_f1_ci95"])
    queue_windows = [(r["shot"], r["window"]) for r in queue_record["rows"]]
    lines += [
        "",
        (
            f"HL-3 adaptation window accuracy {hl3_accuracy}; "
            f"macro-F1 {hl3_macro_f1}. "
            f"Per-class recalls `{hl3['three_class_per_class_recall']}`; confusion "
            f"`{hl3['three_class_confusion']}`. Fit-chosen majority accuracy "
            f"{ci(majority['accuracy'], majority['accuracy_ci95'])}. The descriptive "
            "observed majority is reported separately in the JSON and results table. "
            "OuYang's published classification accuracies are placed beside these "
            "numbers with their different population/task and count-derived caveat."
        ),
        "",
        source("benchmark.json", "Tokamak-SI; legacy; protocol"),
        "",
        source("saw-hl3_fold_0.json"),
        source("saw-hl3_fold_1.json"),
        source("saw-hl3_fold_2.json"),
        source("saw-ours_fold_0.json"),
        source("saw-ours_fold_1.json"),
        source("saw-ours_fold_2.json"),
        "",
        (
            "Frozen ensembles also exported predictions on every queued val shot, "
            "including observable uncertain input support. The prediction manifests "
            "record each NPZ path, picks, thresholds and fitting-only provenance:"
        ),
        "",
        source("saw-hl3_queue_predictions.json"),
        source("saw-ours_queue_predictions.json"),
        "",
        "### Annotation queue, figure and documentation",
        "",
        (
            "The queue uses only held-out nonexpert val shots, with predicted "
            "crashes; exact windows/reasons and stratification are recorded below. "
            "Jalal Butt's cached H/L table covers no eligible val shots, so heating "
            "and predicted-period strata are explicit proxies and H/L stays unknown. "
            "The README's blind protocol hides predictions, reasons and old expert "
            "spans until the owner locks crash times and observation masks."
        ),
        "",
        source("crash_time_queue.json"),
        "",
        f"Queue shot/windows: `{queue_windows}`.",
        "",
        (
            f"Paper example: shot {example['shot']} ({example['split']}), "
            f"window `{example['window_s']}` s; vector PDF `{example['pdf']}` and "
            f"150-dpi PNG `{example['png']}`. Size {example['width_inches']} inches, "
            f"minimum font {example['minimum_font_points']} pt, no title; PNG "
            f"inspection recorded as `{example['png_inspected']}`."
        ),
        "",
        source("paper_example.json"),
        "",
        (
            "The method/results docs are current-state documents with short "
            "history appendices. The README recommends no latest model. All "
            "expert-shot spans are explicitly anchored to old suggestions, and shot "
            "190637 may include edge-originated relaxation. Immutable old-rule "
            "records were reused; no old-detector code changed. Exploratory span "
            "comparisons and old-rule agreement were rerun with current support."
        ),
        "",
        source("validation.json"),
        source("old_rule_run.json"),
        source("muscatello_reference.json"),
    ]
    validation_record = read("validation.json")
    lines += [
        "",
        (
            "Anchored span checks are exploratory; there are no blind point "
            "crash targets. Observable expert-known bins include algorithmic "
            "abstentions in the definite-present sensitivity denominator."
        ),
        "",
        (
            "| Expert shot | Assessed / observable known bins | "
            "Physics observable F1 | Old-rule observable F1 |"
        ),
        "|---|---:|---:|---:|",
    ]
    for row in validation_record["expert"]["by_shot"]:
        lines.append(
            f"| {row['shot']} | {row['assessed_bins']} / "
            f"{row['observable_known_bins']} | {row['new']['presence']['f1']:.4f} | "
            f"{row['old']['presence']['f1']:.4f} |"
        )
    gallery = read("gallery.json")
    assert gallery["png_inspection"]["complete"]
    lines += [
        "",
        source("validation.json", "expert.by_shot"),
        "",
        (
            "The retained training-shot gallery was also regenerated with "
            "dynamic channel legends, including diagnostic outer-band fallbacks. "
            "Every final PNG was viewed; the source ledger records all figure "
            "paths, crash windows and inspected PNG hashes."
        ),
        "",
        source("gallery.json"),
        "",
        "### Verification and remaining concerns",
        "",
        (
            f"Covering tests: {checks['covering_tests_passed']} passed. Only the "
            "covering files were run; Ruff check and required new-file formatting "
            "checks passed. Commands and output tails:"
        ),
        "",
    ]
    for key, record in checks["logs"].items():
        lines += [
            f"`{key}` — `{record['path']}`",
            "",
            "```text",
            *record["tail"],
            "```",
            "",
        ]
    lines += [
        source("verification.json"),
        "",
        (
            "Temporary storage was swept after long jobs. An initial population "
            "attempt using mean-core relaxation was cancelled after fresh review "
            "found dilution; its partial records were isolated in "
            "`fix2/aborted_mean_core/`. All final runs use the corrected per-channel "
            "policy; discarded records supply no final metrics."
        ),
        "",
        (
            "Remaining concerns: independent blind crash truth is unavailable; "
            "frequency/geometry coverage is sparse and nominal; physical H/L "
            "stratification is unavailable for eligible held-out shots; conservative "
            "uncertainty leaves most observable support without training truth. "
            "Local Muscatello reference files are unavailable and were not fetched. "
            "Next: execute the queued blind protocol, evaluate both frozen models "
            "and rules without retuning, and obtain calibrated ECE geometry before "
            "making physical accuracy or inversion-radius claims."
        ),
        "",
    ]
    existing = args.report.read_text() if args.report.exists() else ""
    existing = existing.split("\n## Fix round 2\n", 1)[0].rstrip()
    args.report.write_text(existing + "\n\n" + "\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "audit",
            "queue",
            "figure",
            "inspected",
            "verification",
            "report",
            "refine-records",
        ],
    )
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--previous", type=Path, default=WORK.parent / "fix")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--shot", type=int, default=196405)
    parser.add_argument("--cohort-only", action="store_true")
    parser.add_argument("--commit-range", default="pending")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/claude-89242e53/r4/reports/saw.md"
        ),
    )
    args = parser.parse_args()
    if args.stage == "inspected":
        path = args.output / "paper_example.json"
        record = json.loads(path.read_text())
        record.update(
            png_inspected=True, inspection="Codex viewed the PNG with view_image"
        )
        save_json(path, record)
    else:
        {
            "audit": audit,
            "queue": queue,
            "figure": figure,
            "verification": verification,
            "report": report,
            "refine-records": refine_records,
        }[args.stage](args)


if __name__ == "__main__":
    main()
