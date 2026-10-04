#!/usr/bin/env python
"""Regenerate the TM methods/results document from the benchmark and label records."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
TM = (
    Path(os.environ.get("LABELER_ROOT", "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker"))
    / "round4/tm"
)
BENCH = REPO / "data/events/neoclassical_tearing_mode/benchmark/tm_benchmark.json"
DOC = REPO / "docs/labeler/tearing_detection.md"
LABELS = (
    REPO / "data/events/neoclassical_tearing_mode/extend_tm_interval/tm_interval.csv"
)
README = REPO / "data/events/neoclassical_tearing_mode/README.md"
SOURCES = "../../data/events/neoclassical_tearing_mode/benchmark/sources"
LOCK_EXAMPLES = ("n = 1 decaying", "n = 1 locking")


def three(x, sign=False):
    """Three decimals; a value that rounds to zero carries no sign."""
    text = f"{x:+.3f}" if sign else f"{x:.3f}"
    return text.lstrip("+-") if float(text) == 0.0 else text


def metric(value):
    if value is None:
        return "—"
    return f"{three(value['value'])} [{three(value['lo'])},{three(value['hi'])}]"


def signed(d):
    """A paired difference as `+0.031 [+0.002, +0.059]`."""
    return f"{three(d['value'], True)} [{three(d['lo'], True)}, {three(d['hi'], True)}]"


METRICS = (
    ("auroc", "AUROC"),
    ("auprc", "AUPRC"),
    ("f1", "F1"),
    ("segf1_0.5", "segmental F1"),
)
OURS, BASELINE, TWIN = "tm-ours", "tm-rms-2line", "tm-onsetcnn-retrained"
#: How a model reads in running text.
SAY = {OURS: "`tm-ours`", BASELINE: "the two-line baseline", TWIN: "the retrained CNN"}


def ahead(d, a, b):
    """The model a paired interval puts ahead: `a` above 0, `b` below, else None."""
    if d["lo"] > 0:
        return a
    if d["hi"] < 0:
        return b
    return None


def verdict(d, a, b):
    """`a ahead`, `b ahead` or `not resolved` for one paired difference."""
    who = ahead(d, a, b)
    return f"{SAY[who]} ahead" if who else "not resolved"


def difference_clauses(block, a, b):
    """`AUROC +0.001 [+0.000, +0.002] (`tm-ours` ahead); AUPRC ...` for one block."""
    diff = block["differences"][f"{a}_minus_{b}"]
    return "; ".join(
        f"{label} {signed(diff[key])} ({verdict(diff[key], a, b)})"
        for key, label in METRICS
    )


def names(items):
    """`AUROC`, `AUROC and AUPRC`, `AUROC, AUPRC and F1`."""
    items = list(items)
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def ranking_across_groups(comparison, a, b):
    """How the order of `a` and `b` changes from the excluded to the negative group.

    A metric is reversed if the two groups' paired intervals put different models
    ahead, held if both put the same one ahead, open if either spans zero.
    """
    first = comparison["primary"]["differences"][f"{a}_minus_{b}"]
    second = comparison["uncertain_negative"]["differences"][f"{a}_minus_{b}"]
    reversed_, held, open_, directions = [], {a: [], b: []}, [], set()
    for key, label in METRICS:
        x, y = ahead(first[key], a, b), ahead(second[key], a, b)
        if x and y and x != y:
            reversed_.append(label)
            directions.add((x, y))
        elif x and y:
            held[x].append(label)
        else:
            open_.append(label)
    clauses = []
    if reversed_:
        text = f"the order reverses between the groups on {names(reversed_)}"
        if len(directions) == 1:
            x, y = next(iter(directions))
            text += (
                f" ({SAY[x]} ahead with uncertain time excluded, {SAY[y]} ahead "
                "with it scored as negative)"
            )
        clauses.append(text)
    for who in (a, b):
        if held[who]:
            clauses.append(f"{SAY[who]} is ahead in both groups on {names(held[who])}")
    if open_:
        clauses.append(
            f"the order is not resolved in at least one group on {names(open_)}"
        )
    text = "; ".join(clauses) if clauses else "no metric separates the models"
    return text[0].upper() + text[1:] + "."


def paired_report(paired):
    """The baseline and CNN comparisons as running text, from the benchmark record.

    Used by the document and, shortened, by the README, so no number is typed twice.
    """
    full, subset = paired["full_set"], paired["cnn_subset"]
    ex, un = full["primary"], full["uncertain_negative"]
    cex, cun = subset["primary"], subset["uncertain_negative"]
    baseline = (
        f"On all {len(ex['shots'])} development shots, which both models score on "
        "identical bins, the paired difference `tm-ours` minus the two-line "
        f"baseline is, with uncertain time excluded ({ex['bins_scored']} bins), "
        f"{difference_clauses(ex, OURS, BASELINE)}; with uncertain time scored as "
        f"negative ({un['bins_scored']} bins, {len(un['shots'])} shots), "
        f"{difference_clauses(un, OURS, BASELINE)}. "
        f"{ranking_across_groups(full, OURS, BASELINE)}"
    )
    twin = (
        f"On the {len(cex['shots'])} shots ({cex['bins_scored']} bins; "
        f"{len(cun['shots'])} shots and {cun['bins_scored']} bins with uncertain time "
        "scored as negative) where the retrained CNN has inputs, `tm-ours` minus the "
        f"retrained CNN is {difference_clauses(cex, OURS, TWIN)}; with uncertain "
        f"time scored as negative, {difference_clauses(cun, OURS, TWIN)}"
    )
    return baseline, twin


def patch_readme(blocks):
    """Replace each `<!-- gen:name -->...<!-- /gen:name -->` span of the README.

    The README is hand-written; the spans that carry benchmark numbers are written
    here from the same record as the document, so no number is typed twice.
    """
    text = README.read_text()
    for name, body in blocks.items():
        pattern = re.compile(
            rf"(<!-- gen:{name} -->).*?(<!-- /gen:{name} -->)", re.DOTALL
        )
        assert pattern.search(text), f"README has no gen:{name} span"
        text = pattern.sub(
            lambda m, body=body: f"{m.group(1)}{body}{m.group(2)}", text, count=1
        )
    README.write_text(text)


def pct(x, digits=1):
    return f"{100 * x:.{digits}f}%"


def ends(counts):
    """`{"decay": 49, ...}` as readable text."""
    return ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))


def table(header, rows, text=1):
    """A pipe table; the first `text` columns are left-aligned, the rest right-aligned."""
    out = ["| " + " | ".join(header) + " |"]
    out.append(
        "|" + "|".join("---" if i < text else "---:" for i in range(len(header))) + "|"
    )
    out.extend("| " + " | ".join(str(c) for c in row) + " |" for row in rows)
    return out


def changelog(diag, previous, *, agreement_before, b):
    """The label-rule history: what each round changed and the numbers it moved.

    The paper-facing document states only the current rule; this appendix keeps the
    before-and-after figures. Round 3 is read from the previous round's record
    (`rule_diagnostics_fix3.json`), round 4 from the current one.
    """
    old_u = previous["uncertain_fraction"]["before"]
    new_u = previous["uncertain_fraction"]["after"]
    fc3 = previous["false_confirmation"]["all"]
    seeds3 = previous["n2_seed_cuts"]
    fraction = diag["uncertain_fraction"]
    fc = diag["false_confirmation"]["rows"]
    lags = (
        "lag_300_ms",
        "lag_1000_ms",
        "lag_2000_ms",
        "lag_from_interval_durations",
    )
    locked = {
        k: v["intervals_by_end"].get("locked", 0) for k, v in diag["locking"].items()
    }
    seeds = diag["n2_seed_cuts"]
    cal = b["harmonic_calibration"]
    onset = {
        ref: b["agreement"][f"{ref}_dev"]["agreement"]["n1"]
        for ref in ("seo", "survival")
    }
    intro = (
        "The methods document states only the current rule. This appendix keeps "
        "what each round changed and the development-cohort numbers it moved. "
        "Round 4 figures come from `benchmark/sources/rule_diagnostics_fix4.json`, "
        "round 3 from `rule_diagnostics_fix3.json`."
    )
    lock_text = (
        "Lock confirmation: a step at each candidate time (median 20 to 120 ms "
        "after, 5 above the median 200 to 20 ms before); confirmed cohort locks"
    )
    realized = fc["lag_from_interval_durations"]["lag_quantiles_ms"]["50"]
    false_text = (
        "False-confirmation rate, previous test against the step test, at lags "
        "of 300 / 1000 / 2000 ms and lags drawn from the interval durations "
        f"(realized median {realized:.0f} ms)"
    )
    veto_text = (
        "Harmonic veto recalibrated on the bins that fit n = 2: level (n = 2 "
        f"seeds removed of {seeds['seeds']})"
    )
    window_text = (
        "Inside-the-onset-window count, Seo / survival (the old flag compared the "
        "reference onset with the interval end; the count is not widened)"
    )
    share_text = (
        "Uncertain share of observable time, catalog window / flat-top (pooled)"
    )
    rows4 = [
        (lock_text, locked["before"], locked["after"]),
        (
            false_text,
            " / ".join(pct(fc[k]["previous_rate"]) for k in lags),
            " / ".join(pct(fc[k]["current_rate"]) for k in lags),
        ),
        (
            veto_text,
            f"{diag['n2_cuts']['harmonic_ratio_previous']} ({seeds['veto_previous']})",
            f"{cal['harmonic_ratio']} ({seeds['veto_current']})",
        ),
        (
            window_text,
            " / ".join(
                f"{agreement_before[r]['inside_previous_definition']} of "
                f"{agreement_before[r]['matched']}"
                for r in ("seo", "survival")
            ),
            " / ".join(
                f"{onset[r]['error_ms']['reference_inside_onset_window']} of "
                f"{onset[r]['matched']}"
                for r in ("seo", "survival")
            ),
        ),
        (
            "Duplicate rows in `tm_interval.csv`",
            diag["duplicate_rows"]["before"],
            diag["duplicate_rows"]["after"],
        ),
        (
            share_text,
            " / ".join(
                pct(fraction["before"][k]["pooled_fraction"])
                for k in ("window", "flat_top")
            ),
            " / ".join(
                pct(fraction["after"][k]["pooled_fraction"])
                for k in ("window", "flat_top")
            ),
        ),
    ]
    weak_text = (
        "Weak tracks released at the weak floor, one weak screen over ramp-up and "
        "flat-top: uncertain share of observable flat-top time (pooled)"
    )
    rise_text = (
        "Lock confirmation by a rise over the pre-onset median: false "
        "confirmations at a 300 ms lag (absolute level against relative rise)"
    )
    cap_text = (
        "n = 2 frequency cap scaled with n (30 to 60 kHz): n = 2 seeds it removes"
    )
    old_veto_text = (
        "Harmonic veto calibrated on the bins that fit n = 1: level (seeds removed)"
    )
    rows3 = [
        (
            weak_text,
            pct(old_u["flat_top"]["pooled_fraction"]),
            pct(new_u["flat_top"]["pooled_fraction"]),
        ),
        (rise_text, pct(fc3["absolute_rate"]), pct(fc3["relative_rate"])),
        (
            "Intervals ending in a confirmed lock",
            previous["locking"]["before"]["intervals_by_end"].get("locked", 0),
            previous["locking"]["after"]["intervals_by_end"].get("locked", 0),
        ),
        (cap_text, seeds3["cap_before"], seeds3["cap_after"]),
        (
            old_veto_text,
            f"0.57 ({seeds3['veto_before']})",
            f"0.72 ({seeds3['veto_after']})",
        ),
    ]
    note = (
        "Round 3's harmonic level was calibrated on the bins that fit n = 1, which "
        "are not the harmonic bins (see the harmonic veto in the methods document); "
        "round 4 replaced it. The round 4 false-confirmation rows re-measure the "
        "previous test and the step test on the same draws (absent stretches of at "
        "least 320 ms, 20 draws per shot and lag, seed 0); round 3's figures used "
        "stretches of at least 700 ms, so the two rounds' numbers for the previous "
        "test differ."
    )
    lines = [
        "# Tearing-mode label rule: changelog",
        "",
        intro,
        "",
        "## Round 4",
        "",
        *table(["Change", "Before", "After"], rows4),
        "",
        "## Round 3",
        "",
        *table(["Change", "Before", "After"], rows3),
        "",
        note,
    ]
    return "\n".join(line.rstrip() for line in lines) + "\n"


def main():
    b = json.loads(BENCH.read_text())
    diag = b["rule_diagnostics"]
    cal = b["harmonic_calibration"]
    agreement = {
        ref: b["agreement"][f"{ref}_dev"]["agreement"]["n1"]
        for ref in ("seo", "survival")
    }
    cohort = pd.read_csv(LABELS)
    population = pd.read_csv(TM / "labels/tm_interval_population.csv")
    overlap = len(set(cohort.shot) & set(population.shot))
    fraction = diag["uncertain_fraction"]
    fc = diag["false_confirmation"]
    n2 = diag["n2_cuts"]
    seeds = diag["n2_seed_cuts"]
    kept = n2["n2_intervals_kept"]
    net = kept["current_veto"] - kept["no_veto"]
    lock_after = diag["locking"]["after"]
    steps = diag["lock_steps"]["cohort"]
    durations = fc["interval_duration_quantiles_ms"]
    realized = fc["rows"]["lag_from_interval_durations"]["lag_quantiles_ms"]
    largest = steps["largest_step"]
    share = fraction["after"]["window"]
    share_text = pct(share["pooled_fraction"])
    counts = {k: v["counts"] for k, v in b["label_counts"].items()}
    cov = b["coverage"]
    paired = b["paired_common_shots"]
    like = b["like_for_like"]
    baseline_text, twin_text = paired_report(paired)
    onset_error = {ref: agreement[ref]["error_ms"] for ref in ("seo", "survival")}
    window_text = {
        ref: f"{e['reference_inside_onset_window']} of {agreement[ref]['matched']}"
        for ref, e in onset_error.items()
    }
    late = {
        ref: e["reference_after_interval_start_by_more_than_tolerance"]
        for ref, e in onset_error.items()
    }
    onset_err = {
        ref: b["agreement"][f"{ref}_dev"]["agreement"]["n1"].get("error_ms")
        for ref in ("seo", "survival")
    }

    def unseeded(which):
        return (
            lock_after["uncertain_by_reason"].get("locked_unseeded", {}).get(which, 0)
        )

    lines = [
        "# Whole-interval tearing-mode labels and magnetic-rule recovery",
        "",
        (
            "The target is a **strong rotating n=1/n=2 magnetic mode**, a tearing-mode "
            "proxy without independent island identification. Because the label is a "
            "threshold rule on the same magnetic RMS the detectors read, `tm-ours` "
            "recovers that rule from Mirnov spectrogram features, and a two-line RMS "
            "baseline with no training (`max(n1 RMS / 12 G, n2 RMS / 6 G)`) is the "
            "reference it has to beat. These results do not establish a better TM "
            "detector. AUROC and AUPRC are the primary comparisons; F1 depends on "
            "calibration."
        ),
        "",
        (
            "All labeling, feature screening, fetching, agreement, fitting, scoring, "
            "coverage and galleries exclude the 50 cohort blind shots before opening "
            "their signals; the blind split carries no tearing-mode labels. The cohort "
            f"table holds the {b['label_counts']['cohort']['n_labelled_shots']} "
            "development shots. The population table covers every non-blind shot "
            f"with fetched RMS ({b['label_counts']['population']['n_labelled_shots']} "
            f"shots) and **includes** the development shots ({overlap} of them), so "
            "cohort and population counts overlap and must not be added. Earlier "
            "blind-split score files were moved unread into "
            "`results/quarantine_blind_test/`."
        ),
        "",
        "## Rule and uncertainty",
        "",
        (
            "Both raw RMS and its 5 ms median must exceed 12 G (n1) or 6 G (n2) "
            "continuously for 50 ms, before joining runs. The n1 level follows "
            "Farre-Kaga et al.; n2 is a local extension. Each qualified seed extends "
            "to max(1 G, 10% of its peak). Release gaps up to 50 ms may join; "
            "acquisition gaps cannot. An EFIT rational surface alone does not "
            "determine m and no ECE island radius is resolved, so m stays unassigned."
        ),
        "",
        (
            "Seed and span frequency evidence uses `coherent_frequency` (a 50 ms "
            "window with p90−p10 width ≤ max(2 kHz, 25% of its median) and ≥80% "
            "coherent support over a span). A seed needs a coherent line at or above "
            "**1.5 kHz** and at or below the cap, whether the line comes from "
            "`N1FREQ`/`N2FREQ` or from the Mirnov fallback. Where `N1FREQ`/`N2FREQ` "
            "is missing the fallback requires an n-resolved phase fit ≥ 0.9, "
            "prominence ≥ 10 dB and coherent amplitude above the frozen development "
            "quiet-time p95; its line search covers 1–30 kHz, so inside a span (not "
            "for a seed) a fallback line between 1 and 1.5 kHz still counts as "
            "support. The frequency cap scales with the toroidal number: 30 kHz for "
            "n = 1, 60 kHz for n = 2 (the Mirnov features stop at 30 kHz, so the "
            "n = 2 cap acts through `N2FREQ` only). These are magnetic proxies, not "
            "proof of an island."
        ),
        "",
        (
            "**Harmonic veto.** An n = 2 seed is dropped where n2/n1 is at or below "
            "the veto level, so that the second harmonic of a rotating n = 1 mode, "
            "whose amplitude is a bounded fraction of the n = 1 amplitude, is not "
            "read as a separate n = 2 mode. The level is the 99th percentile of "
            "n2/n1 over development bins where the n = 2 line sits at twice the "
            "n = 1 frequency **and** the six midplane Mirnov probes' phases at that "
            "frequency fit toroidal n = 2 (best-fit |n| = 2, fit ≥ 0.9): "
            f"{cal['sets']['phase_coherent_best_fit_n2']['n_bins']} bins on "
            f"{cal['sets']['phase_coherent_best_fit_n2']['n_shots']} shots, a "
            f"99th percentile of {cal['harmonic_ratio_unrounded']:.3f}, rounded up "
            f"to **{cal['harmonic_ratio']}**. These are the bins where the line at "
            "2 f1 is the harmonic, because the harmonics of a rotating, "
            "non-sinusoidal n = 1 waveform carry toroidal number 2. **The limit:** "
            "toroidal phase cannot separate such a harmonic from a co-rotating, "
            "frequency-coupled n = 2 mode (a 3/2 mode locked to the 2/1), because "
            "both have n = 2 at 2 f1, so the veto is a heuristic and may also "
            "remove real 3/2 modes. "
            f"It removes {seeds['veto_current']} of the {seeds['seeds']} n = 2 seeds "
            f"(50 ms above 6 G on the RMS alone) on {seeds['shots']} development "
            f"shots; the 60 kHz frequency cap alone removes {seeds['cap']}, and the "
            f"two cuts together {seeds['both_current']}. The finished rule yields "
            f"{kept['current_veto']} n = 2 intervals on the development shots "
            f"({kept['no_veto']} without the veto): the veto removes seeds, not "
            "intervals, and a removed seed can change how its neighbours merge, so "
            f"its net effect on finished n = 2 intervals is {net:+d}."
        ),
        "",
        (
            "**Weak tracks and uncertainty.** A weak track needs a 100 ms coherent "
            "core and is uncertain; it is released at the weak-line amplitude floor "
            "itself, and the same weak screen runs over ramp-up and flat-top. Time "
            "above the frozen weak RMS thresholds (n1 2.0282 G, n2 1.8280 G) that "
            "the screen cannot assess is uncertain rather than absent. Quiet time is "
            "absent, except that in otherwise-absent flat-top time a radial field "
            "that steps up (the test below) and stays at least 5 above the median of "
            "that time for 100 ms is uncertain with reason `locked_unseeded`: a "
            "field event with no mode seen, not a mode. The "
            "label is a strong-mode label, not exhaustive TM truth: weak modes that "
            "fail the screen stay absent."
        ),
        "",
        (
            f"Uncertain time is **{share_text}** of the observable catalog-window "
            "time of the development shots (absent + present + uncertain, pooled "
            f"over {share['shots']} shots; {share['shots_over_50_percent']} shots "
            "are more than half uncertain). This one statistic is the uncertain "
            "share quoted in the benchmark caption."
        ),
        "",
        "### Criterion pass rates",
        "",
        (
            "Each criterion is tested separately, by interval toroidal number, on "
            "**absent** time and on **present** time inside that number's own "
            "intervals. The absent rates are measured after labelling: they are "
            "the share of time the rule called absent that nonetheless passes the "
            "criterion, so they are not independent of the rule. Missing criterion "
            "inputs do not count as failures; the denominator is the time the input "
            "was measured. These are diagnostic associations with the magnetic rule, "
            "not independent validation."
        ),
        "",
    ]
    criterion = b["rule_audit"]["criterion_support_fix4"]
    rows = []
    for scope in ("cohort", "population"):
        for n, states in criterion[scope].items():
            for name in states["absent"]:
                cells = []
                for state in ("absent", "present"):
                    rate = states[state][name]
                    value = rate["pass_rate"]
                    cells.append(
                        "—"
                        if value is None
                        else f"{100 * value:.2f} ({rate['total_ms'] / 1000:.1f} s)"
                    )
                rows.append((scope, f"n = {n}", name, *cells))
    lines.extend(
        table(
            [
                "Set",
                "Interval n",
                "Criterion",
                "Absent pass % (post-labelling)",
                "Present pass % (own intervals)",
            ],
            rows,
            text=3,
        )
    )
    lines.extend(
        [
            "",
            (
                "Source: [criterion_support_fix4.json]"
                f"({SOURCES}/criterion_support_fix4.json)."
            ),
            "",
            "## Abrupt collapse and locking",
            "",
            (
                "An amplitude fall from above seed to below release within ≤ 5 ms is "
                "never `ended=decay`. It is `ended=locked` only when independently "
                "confirmed, otherwise `ended=unknown`, and the following phase is "
                "uncertain until the lock signal falls or the discharge ends. The "
                "independent signal is the n = 1 PTDATA radial field `DUSBRADIAL` "
                "(native ptdata units, treated as gauss by disruption-py; the unit is "
                "not verified here, so no absolute field is claimed). A lock is "
                "confirmed by a **step**. At a candidate time t_c (a frequency drop "
                "at least 50 ms after the seed starts, an abrupt collapse, or an "
                "interval's end) the "
                "median of |DUSBRADIAL| over t_c + 20 ms to t_c + 120 ms must exceed "
                "its median over t_c − 200 ms to t_c − 20 ms by at least 5 (each "
                "window needs 50 ms of measured field, and the earlier window does "
                "not reach before the flat-top start). The baseline is local to the "
                "candidate, so a field that is already high or ramps slowly shows no "
                "step and confirms nothing. The lock is released when the field "
                "stays below its baseline plus 5 for 200 ms. Shots 176030–176912 "
                "carry a corrupted channel and are never confirmed. `N1FREQ`/`N2FREQ` "
                "falling to ≤ 1 kHz alone creates only a candidate. n = 2 has no "
                "independent confirmation."
            ),
            "",
            (
                "A lock is looked for at **every** interval end, not only after a "
                "collapse, and for candidates the rule rejected: a rejected candidate "
                "followed by a confirmed lock tail stays uncertain with the lock "
                "reason. "
                f"Across the {steps['n_confirmed_locks']} confirmed cohort locks "
                f"(n = 1) the step is {steps['step_min']:.1f} to "
                f"{steps['step_max']:.1f} (median {steps['step_median']:.1f}) native "
                "units. The false-confirmation rate of the test was measured by "
                "drawing a pseudo-onset in time the labels call absent, in a stretch "
                "long enough for the 200 ms baseline and the 120 ms step window, and "
                "testing for a lock `lag` later. The step test reads only the field "
                "around that pseudo-lock; the lags are 300, 1000 and 2000 ms and "
                f"lags drawn from the cohort's {fc['n_interval_durations']} interval "
                f"durations (median {durations['50']:.0f} ms, 90th percentile "
                f"{durations['90']:.0f} ms). The last row keeps only the durations "
                "that fit the quiet stretch, so its realized lags are shorter: median "
                f"{realized['50']:.0f} ms, 10th percentile {realized['10']:.0f} ms, "
                f"90th percentile {realized['90']:.0f} ms."
            ),
            "",
        ]
    )
    rows = []
    for key, title in (
        ("lag_300_ms", "300 ms"),
        ("lag_1000_ms", "1000 ms"),
        ("lag_2000_ms", "2000 ms"),
        (
            "lag_from_interval_durations",
            (
                "drawn from the interval durations "
                f"(realized median {realized['50']:.0f} ms)"
            ),
        ),
    ):
        row = fc["rows"][key]
        rows.append(
            (
                title,
                row["draws"],
                row["shots"],
                row["current"],
                pct(row["current_rate"]),
            )
        )
    lines.extend(
        table(
            ["Lag", "Draws", "Shots", "False confirmations", "Rate"],
            rows,
        )
    )
    lines.extend(
        [
            "",
            (
                "Intervals ending in a confirmed lock: "
                f"{lock_after['intervals_by_end'].get('locked', 0)} "
                "on the development shots. Uncertain `locked_unseeded` time: "
                f"{unseeded('rows')} rows, {unseeded('seconds'):.1f} s on "
                f"{unseeded('shots')} shots. The column example's locking shot is "
                f"**{largest['shot']}**, chosen by a fixed rule: the confirmed cohort "
                "n = 1 lock with the largest step "
                f"({largest['step']:.1f} units: {largest['before']:.1f} before, "
                f"{largest['after']:.1f} after); its decaying counterpart is 189514."
            ),
            "",
            (
                f"`DUSBRADIAL` is on file for {diag['lock_records']['dev_shots_with_record']} "
                f"of the {diag['lock_records']['dev_shots']} development shots. The "
                "shots that lacked one were requested on the login node through `fdp run` with one "
                "worker and a stop at the first authentication error (none occurred); "
                "the fetch ended after repeated data-server lookup failures "
                "(`getservbyname` for PTSERVER, not an authentication error) and "
                f"{diag['lock_records']['dev_shots_without_record']} shots were not "
                "retrieved. Those shots "
                "keep an unknown lock status and get no relative baseline or "
                "`locked_unseeded` check; unknown is not evidence of no lock. The "
                "population has the same limitation on shots with no record."
            ),
            "",
        ]
    )
    rows = []
    for name, meta in b["label_counts"].items():
        c = meta["counts"]
        rows.append(
            (
                name,
                meta["n_labelled_shots"],
                c["n_intervals"],
                meta["n_shots_with_a_mode"],
                ends(c["intervals_by_end"]),
                c["n_locked"],
            )
        )
    lines.extend(
        table(
            [
                "Set",
                "Labeled shots",
                "Intervals",
                "Mode shots",
                "Ends",
                "Confirmed locks",
            ],
            rows,
        )
    )
    lines.extend(
        [
            "",
            (
                "Source: benchmark `label_counts`, `locking_coverage` and the adjacent "
                "label metadata."
            ),
            "",
            (
                "The `extend_tm_interval` table requires conversion before promotion to "
                "`review/`: TM catalog state 3 is forbidden, n-specific present/uncertain "
                "rows overlap, onset rows have zero length, and fractional-ms boundaries "
                "must become whole milliseconds. "
                "`test_shot_table_validates_extension_intervals_with_onset_points_and_spans` "
                "tests geometry, not catalog validity."
            ),
            "",
            "## Onsets and historical-onset agreement",
            "",
            (
                "The onset is the interval's start, a point event, and the interval is "
                "the span. The interval start is the qualified seed grown backwards "
                "to max(1 G, 10% of its peak), so it precedes the 50 ms seed "
                "crossing; it is not the time the strong rule first holds. Each onset "
                "carries `onset_window_ms`, from the start of the preceding same-n "
                "weak track to the interval start, the span in which the mode could "
                f"have begun. {counts['cohort']['n_onset_windows']} of "
                f"{counts['cohort']['n_onset_points']} cohort onsets have such a "
                "window (the others have no preceding weak track), and "
                f"{counts['cohort']['n_onset_windows_at_most_5_ms']} of those windows "
                "are 5 ms or shorter and say almost nothing about where the mode "
                f"began: {counts['cohort']['n_onset_rows_flagged_degenerate']} "
                "onset rows are flagged with `onset_window_degenerate: true` (an "
                "interval with no onset point has no row to flag), and no window is "
                "widened."
            ),
            "",
            (
                "A reference onset is counted as matched when it is **contained within "
                "an interval or within ±100 ms of its edges**. That tests whether the "
                "label holds the historical onset, not how accurately it times it; "
                "intervals last seconds. Recall on the development shots is **Seo "
                f"{agreement['seo']['matched']}/{agreement['seo']['reference_onsets']}** "
                f"and **survival {agreement['survival']['matched']}/"
                f"{agreement['survival']['reference_onsets']}**. The strong 50 ms seed "
                "rule omits short and fast-locking modes: of the missed onsets, "
                f"{agreement['seo']['missed_reasons']['short_burst']}/"
                f"{agreement['seo']['missed']} (Seo) and "
                f"{agreement['survival']['missed_reasons']['short_burst']}/"
                f"{agreement['survival']['missed']} (survival) are short bursts, and "
                f"{agreement['seo']['missed_reasons']['coherent_line_not_supported']} "
                f"and {agreement['survival']['missed_reasons']['coherent_line_not_supported']} "
                "lack a supported coherent line. The Seo onsets fall a median "
                f"{agreement['seo']['error_ms']['median']:.0f} ms after the interval "
                "start, inside the interval. A reference onset lies **inside the "
                "onset window** when it is between the window's start and the "
                "interval's start (no widening: the match already allows 100 ms "
                f"past the start): {window_text['seo']} Seo and "
                f"{window_text['survival']} survival matched onsets. {late['seo']} "
                f"Seo and {late['survival']} survival onsets "
                "lie more than 100 ms after the interval start. The "
                "label therefore holds the historical onset, it does not time it. "
                "The survival archive does not follow "
                "a literal continuous-50 ms rule on the 1 kHz `N1RMS`: its onsets "
                "classed short bursts have no 50 ms raw-and-median 12 G crossing "
                "near them. The rule was not selected to maximise archive agreement."
            ),
            "",
        ]
    )
    rows = []
    for ref in ("seo", "survival"):
        a = b["agreement"][f"{ref}_dev"]
        s = a["agreement"]["n1"]
        e = onset_err[ref]
        strict = a["agreement_strict"]["n1"]["matched"]
        rows.append(
            (
                ref,
                s["n_shots_covered"],
                s["reference_onsets"],
                s["matched"],
                strict,
                "—"
                if e is None
                else f"{round(e['median'])} [{round(e['q25'])}, {round(e['q75'])}]",
                "—" if e is None else pct(e["within_ms"]["100"], 0),
                "—"
                if e is None
                else f"{e['reference_inside_onset_window']} of {s['matched']}",
                f"{s['intervals_without_an_onset']} of {s['compared_intervals']}",
            )
        )
    lines.extend(
        table(
            [
                "Reference",
                "Covered shots",
                "Onsets",
                "Matched",
                "Strictly contained",
                "Onset error, ms (median [q25, q75])",
                "Within 100 ms",
                "Inside onset window",
                "Intervals without a reference onset",
            ],
            rows,
        )
    )
    lines.extend(
        [
            "",
            (
                "The onset error is the reference onset minus our interval start, over "
                'matched onsets only (positive: our interval began first). "Inside '
                'onset window" counts matched reference onsets between the window '
                "start and the interval start (not widened). "
                '"Intervals without a reference onset" counts compared intervals '
                "that hold no reference onset of their shot."
            ),
            "",
            (
                f"Sources: benchmark `agreement`, [agreement_seo_cohort_dev.json]({SOURCES}/agreement_seo_cohort_dev.json) and "
                f"[agreement_survival_cohort_dev.json]({SOURCES}/agreement_survival_cohort_dev.json). "
                "The frozen-rule sensitivity is generated by `tm_sensitivity.py` and "
                "bundled as `sensitivity_survival_dev.json`; it does not tune the rule."
            ),
            "",
            "## Development detection benchmark",
            "",
            (
                "The outer held-shot assignment is the seed-0 round-robin. Before any "
                "fit, `tm_cv_plan.py` freezes inner shot roles in "
                "`inner_splits_fix4.json`, "
                "stratified by whether a shot has an interval, with within-stratum swaps "
                "so every model and legacy target has positive support in its "
                "validation shots. No score or held-fold performance chooses roles. "
                "Each fit asserts a positive-bearing early-stopping validation set and "
                "threshold selection raises on zero positives. The published CNN is "
                "shown at its own 0.5 threshold and at the tuned one, in both target "
                "groups. The DSM's published survival "
                "threshold 0.7 corresponds to risk 0.3."
            ),
            "",
            (
                "All learned models are refitted with three seeds on the same recipe "
                "as before; hyperparameters stay fixed from earlier development work, "
                "so selection is not fully nested. The scalar CNN has t+25 ms input "
                "lookahead and is an offline detector. The DSM is a detection head on "
                "its original embedding; published DSM rows remain horizon forecasts, "
                "with the shots in the published training list excluded from their "
                "scores. The published CNN's training overlap with the development "
                "shots is unknown (dagger), so its scores are descriptive."
            ),
            "",
            (
                "Targets use absolute 10 ms bins; legacy rows use their native 25 ms "
                "bins. Each row has its own shot set, listed in the appendix table and "
                "its source JSON. In the primary group categories 2 and 3 and "
                "unavailable inputs are excluded. AUROC and AUPRC use 1,024 score "
                "quantiles and every interval is a 1,000-draw whole-shot bootstrap "
                "(seed 0). Segmental F1 uses IoU 0.5, 50 ms minimum segments and "
                "≤ 50 ms negative-gap closing; unavailable bins are hard barriers."
            ),
            "",
            (
                "**Two co-primary target definitions.** *Uncertain excluded* removes "
                "category 2 and 3 time, so it does not score where an interval starts "
                "or ends: the uncertain time borders the intervals. *Uncertain as "
                "negative* scores category 2 as negative and still excludes category 3, "
                "so it does score the boundaries. Fits, scores and thresholds are the "
                "same in both. Finite uncertain features stay in the temporal input "
                "context; the target mask controls loss and scoring only. The "
                "primary metrics do not score boundary placement in the first group."
            ),
            "",
        ]
    )
    rows = []
    for row in b["rows"]:
        m = row["metrics"]
        label = row["model"] + (f" ({row['variant']})" if row["variant"] else "")
        rows.append(
            (
                row["setting"],
                label + ("†" if not row["held_out"] else ""),
                f"{m['n_shots']} / {m['bins_scored']}",
                *[metric(m.get(k)) for k in ("auroc", "auprc", "f1", "segf1_0.5")],
            )
        )
    lines.extend(
        table(
            [
                "Setting",
                "Model",
                "Shots / bins",
                "AUROC [95% CI]",
                "AUPRC [95% CI]",
                "F1 [95% CI]",
                "Segmental F1 [95% CI]",
            ],
            rows,
            text=2,
        )
    )
    lines.extend(
        [
            "",
            (
                "The Mirnov-derived uncertainty mask is "
                f"**{share_text}** of observable catalog-window time (the statistic "
                "defined above). It shares `tm-ours` inputs, so the primary group "
                "emphasises strong modes against quiet magnetic time and can favour "
                "the magnetic detector; the uncertain-as-negative group also scores "
                "those hard cases. The +RMS row adds circular label inputs. The legacy rows "
                "use their published thresholds as primary and tuned thresholds as "
                "secondary (appendix); AUPRC is not compared across settings because "
                "prevalence differs, and `figure2_tm.json` records it beside each value. "
                "Rows without a variant use the threshold tuned on inner validation. "
                "The DSM raises no alarm at its published level (risk 0.3) on the "
                "legacy bins, so its F1 there is 0 (recall 0, precision undefined)."
            ),
            "",
            (
                "**Reading the table.** The two-line RMS baseline is the rule "
                "restated as a score, with no training. A model that reaches it "
                "has recovered the magnetic rule; one that exceeds it uses "
                f"information the rule does not. {baseline_text} `tm-ours` is "
                "therefore reported as recovering the magnetic rule, not as a better "
                "detector."
            ),
            "",
            "### Paired comparison",
            "",
            (
                "Each comparison uses the development shots and available 10 ms "
                "bins that both of its models score, within each target group; "
                "each model keeps its inner-validation threshold. Paired bootstrap "
                "draws resample the same shots, so each difference row carries a "
                "paired 95% interval; a difference whose interval spans 0 is not "
                "resolved. `tm-ours` and the two-line baseline need only the Mirnov "
                "features, so they are compared on all "
                f"{len(paired['full_set']['primary']['shots'])} "
                "development shots (identical bins) and every statement about the "
                "baseline uses that set. The retrained CNN has inputs on fewer shots, "
                "so only the comparison with the CNN uses the restricted set; "
                "comparing `tm-ours` with the baseline on it would rank them on "
                "the shots the CNN happens to cover. Ranking, not a "
                "threshold-specific F1 gain, is the primary comparison."
            ),
            "",
            f"{twin_text}.",
            "",
        ]
    )
    rows = []
    for comparison, scope, pair in (
        ("full_set", "tm-ours, two-line RMS (all shots)", (OURS, BASELINE)),
        ("cnn_subset", "tm-ours, retrained CNN (CNN shots)", (OURS, TWIN)),
    ):
        for target, mode in (
            ("uncertain excluded", "primary"),
            ("uncertain = negative", "uncertain_negative"),
        ):
            block = paired[comparison][mode]
            count = f"{len(block['shots'])} / {block['bins_scored']}"
            for name, m in block["metrics"].items():
                rows.append(
                    (
                        f"{target}, {scope}",
                        name,
                        count,
                        *[metric(m[k]) for k, _ in METRICS],
                    )
                )
            rows.append(
                (
                    f"{target}, {scope}",
                    f"Difference, {pair[0]} − {pair[1]}",
                    count,
                    *[
                        metric(block["differences"][f"{pair[0]}_minus_{pair[1]}"][k])
                        for k, _ in METRICS
                    ],
                )
            )
    lines.extend(
        table(
            [
                "Target",
                "Model",
                "Shots / bins",
                "AUROC",
                "AUPRC",
                "F1",
                "Segmental F1",
            ],
            rows,
            text=2,
        )
    )
    lines.extend(
        [
            "",
            ("### Published model against its retrained twin"),
            "",
            (
                "For each architecture the published model and the retrained one are "
                "scored on one target, one mask and one set of shots and bins (for the "
                "DSM, outside the published training list). The published CNN is shown "
                "at its own threshold and at the tuned one; the retrained twin at its "
                "tuned threshold."
            ),
            "",
        ]
    )
    rows = []
    for block in like.values():
        for mode, title in (
            ("primary", "uncertain excluded"),
            ("uncertain_negative", "uncertain = negative"),
        ):
            cell = block[mode]
            for label, name in (
                (
                    "published_at_published_threshold",
                    block["published_model"] + " (published threshold)",
                ),
                (
                    "published_at_tuned_threshold",
                    block["published_model"] + " (tuned threshold)",
                ),
                ("retrained", block["retrained_model"]),
            ):
                m = cell["metrics"][label]
                rows.append(
                    (
                        title,
                        name,
                        f"{cell['n_shots']} / {cell['bins_scored']}",
                        f"{cell['prevalence']['value']:.3f}",
                        metric(m["auroc"]),
                        metric(m["auprc"]),
                        metric(m["f1"]),
                    )
                )
    lines.extend(
        table(
            [
                "Target",
                "Model",
                "Shots / bins",
                "Prevalence",
                "AUROC",
                "AUPRC",
                "F1",
            ],
            rows,
            text=2,
        )
    )
    size = b["training_size_confound"]
    lines.extend(
        [
            "",
            (
                "**Training-set size.** The published models were trained on "
                "thousands of shots (the DSM's list has "
                f"{size['published_dsm_training_shots']}); each retrained model sees "
                f"about {size['retrained_models_train_shots_per_fold']['mean']:.0f} "
                "development shots per outer fold. A retrained model that ranks below "
                "its published twin is confounded by that difference, and no "
                "population-scale retrain is part of this benchmark. No "
                "retraining-generalisation claim follows from published-model "
                "comparisons while the original training overlap is unknown."
            ),
            "",
            (
                "Sources: benchmark `paired_common_shots` and `like_for_like`, "
                "[figure2_tm.json](figure2_tm.json)."
            ),
            "",
            "## Coverage and publication artifacts",
            "",
            (
                "**No TM coverage gain is claimed.** Coverage uses the measured "
                "plasma-start to catalog-end domain and the 10 ms grid on each matched "
                "shot set. Observable time includes uncertainty; labeled time excludes "
                "it. Scoring can include quiet ramp-up, so scoring bins and "
                "observable-plasma coverage have different denominators."
            ),
            "",
        ]
    )
    rows = []
    for name in ("ours", "ours_population"):
        c = cov[name]
        rows.append(
            (
                name,
                c["labelled_shots"],
                f"{c['observable_plasma_seconds']:.2f}",
                f"{c['labelled_seconds']:.2f}",
                f"{c['uncertain_seconds']:.2f}",
            )
        )
    lines.extend(
        table(["Set", "Shots", "Observable plasma s", "Labeled s", "Uncertain s"], rows)
    )
    lines.append("")
    rows = []
    for name in ("legacy_seo", "legacy_survival"):
        c = cov[name]
        rows.append(
            (
                name,
                c["labelled_shots"],
                f"{c['labelled_seconds']:.2f}",
                f"{c['interval_seconds_on_same_shots']:.2f}",
                f"{c['common_labelled_seconds']:.2f}",
            )
        )
    lines.extend(
        table(
            [
                "Matched reference",
                "Shots",
                "Legacy labeled s",
                "Interval labeled s",
                "Common labeled s",
            ],
            rows,
        )
    )
    lines.extend(
        [
            "",
            (
                "Sources: benchmark `coverage`, the adjacent label metadata and "
                "[figure2_tm.json](figure2_tm.json). Cohort and population overlap."
            ),
            "",
            (
                "`tm_gallery.py --width 3.25 --columns 1` provides column-sized example "
                "panels with 7.5 pt text at final width; the example shots are one "
                "decaying n = 1 mode (189514) and one locking n = 1 mode "
                f"({largest['shot']}, picked by the fixed rule above: the largest "
                "radial-field step). In the galleries a present interval is drawn "
                "plain and uncertain time is one flat grey without outlines, so "
                "uncertain rows that overlap are drawn once and a boxed or darker "
                "patch never means more uncertainty. In the spectrogram and RMS "
                "panels each interval is shaded in its own colour, and where an "
                "n = 1 and an n = 2 interval overlap the two shadings blend into a "
                'light grey (legend entry "n = 1 and 2 overlap"); that grey is '
                "not uncertain time, which is drawn only in the strip above the "
                "spectrogram. Only locked phases are hatched: "
                "slashes where a step of the radial field confirms a mode's lock, "
                "crosses where the field steps in flat-top time with no mode seen "
                "(`locked_unseeded`), dots where a lock is suspected but "
                "unconfirmed. A grey spectrogram background is time "
                "with no record of that diagnostic. The 7.3-inch MHR and Mirnov "
                "galleries are supplementary audit material and must not be shrunk "
                "into a paper column. Their JSON sidecars record shots, width, font, "
                "diagnostic, label/source hashes and image hashes. "
                "`tm_render_tables.py` renders the final TeX at 6.75-inch text width, "
                "rejects overfull horizontal boxes and writes PDF and 150-dpi PNG "
                "previews with provenance. The appendix explains the dagger and lists "
                "the shots behind each row."
            ),
            "",
            "## Reproduction",
            "",
            (
                "Use the frozen/no-install pixi `labelmaker` environment, the worktree "
                "`PYTHONPATH`, a scratch `TMPDIR` and `LABELER_NO_FETCH=1`. Only the "
                "`DUSBRADIAL` fetch unsets it and runs through `fdp run` on the login "
                "node. Train on `CUDA_VISIBLE_DEVICES=1` with the phase3 CUDA venv."
            ),
            "",
            (
                "Sequence: `tm_magfeatures.py`, `tm_harmonic_calibration.py`, "
                "`tm_label.py` (`--from cohort`, then `--from population`), "
                "`tm_audit_rule.py`, `tm_cv_plan.py`, `tm_rule_diagnostics.py`, "
                "`tm_prior_retrain.py --model cnn/dsm`, `tm_ours.py --features "
                "magnetics/magnetics+rms` and `--baseline`, `tm_prior_published.py` "
                "(legacy and Tokamak-SI), `tm_agreement.py --exclude-test --tag _dev`, "
                "`tm_sensitivity.py`, `tm_gallery.py` (the column examples with "
                "`--lock-example` on the rule-diagnostics record), `tm_benchmark.py "
                "--rescore --gallery-reviewed`, `tm_write_doc.py`, "
                "`tm_render_tables.py`, `tm_verify_artifacts.py`. Source "
                "JSONs and the shot lists are committed under the benchmark's "
                "`sources/`; predictions, weights, signals and figures stay under "
                "`$LABELER_ROOT/round4/tm/`."
            ),
            "",
            (
                "Remaining limitations: the target and the detector inputs share the "
                "magnetic RMS, the n = 2 level and the lock units are local "
                "conventions, the harmonic veto is a heuristic that may remove real "
                "3/2 modes, the onset is the interval start and does not time the "
                "mode independently, weak and fast-locking modes are omitted, the "
                "published "
                "CNN's training overlap is unknown, the training sets of the retrained "
                "models are far smaller than the published ones, lock status is "
                "unknown on development shots without a `DUSBRADIAL` record, and there "
                "is no independent ECE island radius or fully nested hyperparameter "
                "selection."
            ),
            "",
            (
                "Earlier label rules and the numbers each change moved are in the "
                "[changelog](tearing_detection_changelog.md)."
            ),
        ]
    )
    previous = json.loads(
        (BENCH.parent / "sources/rule_diagnostics_fix3.json").read_text()
    )
    DOC.with_name("tearing_detection_changelog.md").write_text(
        changelog(
            diag, previous, agreement_before=diag["onset_window_flag_before"], b=b
        )
    )
    DOC.write_text("\n".join(line.rstrip() for line in lines) + "\n")
    patch_readme(
        {
            "veto": (
                f"it removes {seeds['veto_current']} of {seeds['seeds']} n = 2 seeds "
                "on the development shots; the finished rule has "
                f"{kept['current_veto']} n = 2 intervals, {kept['no_veto']} without "
                "the veto, because a removed seed can change how its neighbours "
                f"merge (net effect on intervals {net:+d})"
            ),
            "paired": baseline_text,
        }
    )
    provenance = {
        "made_by": "scripts/labeler/tm_write_doc.py",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "benchmark_sha256": hashlib.sha256(BENCH.read_bytes()).hexdigest(),
        "document_sha256": hashlib.sha256(DOC.read_bytes()).hexdigest(),
        "document": str(DOC.relative_to(REPO)),
        "readme_sha256": hashlib.sha256(README.read_bytes()).hexdigest(),
    }
    for path in (
        TM / "results/document_fix4.json",
        BENCH.parent / "sources/document_fix4.json",
    ):
        path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(DOC)


if __name__ == "__main__":
    main()
