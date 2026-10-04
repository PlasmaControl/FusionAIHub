#!/usr/bin/env python
"""Regenerate the TM methods/results document from the benchmark and label records."""

from __future__ import annotations

import hashlib
import json
import os
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
SOURCES = "../../data/events/neoclassical_tearing_mode/benchmark/sources"
LOCK_EXAMPLES = ("n = 1 decaying", "n = 1 locking")


def metric(value):
    if value is None:
        return "—"
    return "{value:.3f} [{lo:.3f},{hi:.3f}]".format(**value)


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
    lock_before = diag["locking"]["before"]
    lock_after = diag["locking"]["after"]
    counts = {k: v["counts"] for k, v in b["label_counts"].items()}
    cov = b["coverage"]
    paired = b["paired_common_shots"]
    like = b["like_for_like"]
    two = paired["differences"]["tm-ours_minus_tm-rms-2line"]
    pair_text = (
        f"{two['auroc']['value']:+.3f} AUROC [{two['auroc']['lo']:+.3f}, "
        f"{two['auroc']['hi']:+.3f}] and {two['auprc']['value']:+.3f} AUPRC "
        f"[{two['auprc']['lo']:+.3f}, {two['auprc']['hi']:+.3f}]"
    )
    un = {
        r["model"]: r["metrics"]
        for r in b["rows"]
        if r["setting"] == "Tokamak-SI, uncertain = negative" and not r["variant"]
    }
    un_text = (
        f"the order {'reverses' if un['tm-ours']['auprc']['value'] < un['tm-rms-2line']['auprc']['value'] else 'holds'}: "
        f"AUPRC {un['tm-ours']['auprc']['value']:.3f} for `tm-ours` against "
        f"{un['tm-rms-2line']['auprc']['value']:.3f} for the baseline"
    )
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
            "coherent support over a span). Where `N1FREQ`/`N2FREQ` is missing the "
            "Mirnov fallback requires an n-resolved phase fit ≥ 0.9, prominence ≥ "
            "10 dB, a line frequency between 1 kHz and the cap below, and coherent "
            "amplitude above the frozen development quiet-time p95. The frequency cap "
            "scales with the toroidal number: 30 kHz for n = 1, 60 kHz for n = 2 "
            "(the Mirnov features stop at 30 kHz, so the n = 2 cap acts through "
            "`N2FREQ` only). These are magnetic proxies, not proof of an island."
        ),
        "",
        (
            "**Harmonic veto.** An n = 2 seed is dropped where n2/n1 is at or below "
            "the veto level, so the second harmonic of an n = 1 mode, which is a "
            "bounded fraction of it, is not read as a separate n = 2 mode. The level "
            "is the "
            "99th percentile of n2/n1 over development bins where the n = 2 line sits "
            "at twice the n = 1 frequency **and** the Mirnov best-fit toroidal number "
            f"at that frequency is 1 ({cal['sets']['phase_coherent_best_fit_n1']['n_bins']} "
            f"bins on {cal['sets']['phase_coherent_best_fit_n1']['n_shots']} shots), "
            f"which gives {cal['harmonic_ratio_unrounded']:.3f}, set to "
            f"**{cal['harmonic_ratio']}**. The earlier level, {cal['previous_harmonic_ratio_rounded']}, "
            "used every frequency-matched bin "
            f"({cal['sets']['frequency_matched_all']['n_bins']} bins), a set that "
            "can also hold frequency-coupled 3/2 + 2/1 pairs, which are real n = 2 "
            "modes. The phase-coherent set is a small minority of those bins, because "
            "the harmonic of an n = 1 waveform usually fits n = 2, so the level rests "
            "on a small sample; a higher level vetoes more seeds. "
            f"Of the {seeds['seeds']} n = 2 seeds (50 ms above 6 G on the RMS "
            f"alone) on {seeds['shots']} development shots, the harmonic veto removes "
            f"{seeds['veto_before']} at the earlier level and {seeds['veto_after']} "
            f"at {cal['harmonic_ratio']}; the coherent-frequency cap removes "
            f"{seeds['cap_before']} at 30 kHz and {seeds['cap_after']} at 60 kHz; "
            f"both together remove {seeds['both_before']} before and "
            f"{seeds['both_after']} now. The n = 2 intervals that result: "
            f"{n2['n2_intervals_kept']['before_cap30_veto0.57']} with the earlier "
            f"settings, {n2['n2_intervals_kept']['after_cap60_veto']} with the "
            "current ones."
        ),
        "",
        (
            "**Weak tracks and uncertainty.** A weak track needs a 100 ms coherent "
            "core and is uncertain; it is released at the weak-line amplitude floor "
            "itself, and the same weak screen runs over ramp-up and flat-top. Time "
            "above the frozen weak RMS thresholds (n1 2.0282 G, n2 1.8280 G) that "
            "the screen cannot assess is uncertain rather than absent, and a "
            "sustained exceedance with no seed in otherwise-absent flat-top time is "
            "uncertain with reason `locked_unseeded`. Quiet time is absent. The "
            "label is a strong-mode label, not exhaustive TM truth: weak modes that "
            "fail the screen stay absent."
        ),
        "",
        (
            "The uncertain share of observable development plasma went from "
            f"**{pct(fraction['before']['flat_top']['pooled_fraction'])}** pooled "
            f"(median shot {pct(fraction['before']['flat_top']['median_shot_fraction'])}) "
            f"to **{pct(fraction['after']['flat_top']['pooled_fraction'])}** pooled "
            f"(median shot {pct(fraction['after']['flat_top']['median_shot_fraction'])}) "
            "from flat-top start; counted over the whole catalog window it is "
            f"{pct(fraction['before']['window']['pooled_fraction'])} before and "
            f"{pct(fraction['after']['window']['pooled_fraction'])} now "
            f"(median shot {pct(fraction['before']['window']['median_shot_fraction'])} "
            f"before, {pct(fraction['after']['window']['median_shot_fraction'])} now). "
            f"{fraction['after']['flat_top']['shots_over_50_percent']} of "
            f"{fraction['after']['flat_top']['shots']} shots are more than half uncertain "
            f"(before: {fraction['before']['flat_top']['shots_over_50_percent']})."
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
    criterion = b["rule_audit"]["criterion_support_fix3"]
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
            f"Source: [criterion_support_fix3.json]({SOURCES}/criterion_support_fix3.json).",
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
                "confirmed by a **sustained rise**: |DUSBRADIAL| at least 5 above its "
                "median over the 200 ms before the interval's onset, held for 20 ms "
                "near the interval end or collapse; it is released when the field "
                "stays below that rise for 200 ms. The relative rule replaces an "
                "absolute 5-unit level, which fired on shots whose field sits high "
                "before any mode. Shots 176030–176912 carry a corrupted channel and "
                "are never confirmed. `N1FREQ`/`N2FREQ` falling to ≤ 1 kHz alone "
                "creates only a candidate. n = 2 has no independent confirmation."
            ),
            "",
            (
                "A lock is looked for at **every** interval end, not only after a "
                "collapse, and for candidates the rule rejected: a rejected candidate "
                "followed by a confirmed lock tail stays uncertain with the lock "
                "reason. Quiet-time false confirmations were measured by placing "
                f"{fc['all']['draws']} pseudo-onsets (20 per shot, seed 0) in stretches "
                "labelled absent and testing for a lock 300 ms later: the old absolute "
                f"rule confirmed **{pct(fc['all']['absolute_rate'])}**, the relative "
                f"rule **{pct(fc['all']['relative_rate'])}**; on draws whose "
                "pre-onset field was already ≥ 3 the rates are "
                f"{pct(fc['draws_with_baseline_at_least_3']['absolute_rate'])} and "
                f"{pct(fc['draws_with_baseline_at_least_3']['relative_rate'])} "
                f"({fc['draws_with_baseline_at_least_3']['draws']} draws). "
                f"Intervals ending in a confirmed lock: {lock_before['intervals_by_end'].get('locked', 0)} "
                f"before, {lock_after['intervals_by_end'].get('locked', 0)} now. Uncertain "
                f"`locked_unseeded` time: {unseeded('rows')} rows, "
                f"{unseeded('seconds'):.1f} s on {unseeded('shots')} shots."
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
                "The onset is the interval's start, a point event. The interval is "
                "the span. Because the interval start is where the strong rule first "
                "holds, the true onset lies earlier, in the preceding weak track; each "
                "onset carries `onset_window_ms`, the start of the preceding same-n "
                "weak track (the interval start where none exists), so the window "
                f"{counts['cohort']['n_onset_windows']} of {counts['cohort']['n_onset_points']} "
                "cohort onsets have is the span in which the mode could have begun."
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
                "lack a supported coherent line. The survival archive does not follow "
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
                else pct(e["reference_inside_onset_window_fraction"], 0),
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
                "matched onsets only (positive: our interval began first). "
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
                "fit, `tm_cv_plan.py` freezes inner shot roles in `inner_splits_fix3.json`, "
                "stratified by whether a shot has an interval, with within-stratum swaps "
                "so every model and legacy target has positive support in its "
                "validation shots. No score or held-fold performance chooses roles. "
                "Each fit asserts a positive-bearing early-stopping validation set and "
                "threshold selection raises on zero positives. The published CNN is "
                "also shown at its own 0.5 threshold. The DSM's published survival "
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
                "The Mirnov-derived uncertainty mask excludes about "
                f"**{pct(fraction['after']['flat_top']['pooled_fraction'])}** of "
                "observable development plasma. It shares `tm-ours` inputs, so the primary group "
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
                "information the rule does not. On the shared shots `tm-ours` "
                f"exceeds the baseline by {pair_text} "
                "and, with uncertain time scored as negative, "
                f"{un_text}. `tm-ours` is therefore reported as "
                "recovering the magnetic rule, not as a better detector."
            ),
            "",
            "### Paired comparison",
            "",
            (
                f"Same {len(paired['shots'])} development shots and "
                f"{paired['bins_scored']} identical available 10 ms bins for all three "
                "models; each keeps its inner-validation threshold. Paired bootstrap "
                "draws resample the same shots. Ranking, not a threshold-specific F1 "
                "gain, is the primary comparison."
            ),
            "",
        ]
    )
    rows = []
    for name, m in paired["metrics"].items():
        rows.append(
            (name, *[metric(m[k]) for k in ("auroc", "auprc", "f1", "segf1_0.5")])
        )
    for key, label in (
        ("tm-ours_minus_tm-onsetcnn-retrained", "Difference, tm-ours − retrained CNN"),
        ("tm-ours_minus_tm-rms-2line", "Difference, tm-ours − two-line RMS"),
    ):
        rows.append(
            (
                label,
                *[
                    metric(paired["differences"][key][k])
                    for k in ("auroc", "auprc", "f1", "segf1_0.5")
                ],
            )
        )
    lines.extend(table(["Model", "AUROC", "AUPRC", "F1", "Segmental F1"], rows))
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
                    block["published_model"] + " at thr.",
                ),
                ("published_at_tuned_threshold", block["published_model"] + " tuned"),
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
                "decaying n = 1 mode and one locking n = 1 mode. In the galleries a "
                "present interval is drawn plain, uncertain time is flat grey, and "
                "only the locked phase is hatched (a rise of the radial field, with or "
                "without a preceding mode). A grey spectrogram background is time "
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
                "`tm_audit_rule.py`, `tm_cv_plan.py`, `tm_fix3_diagnostics.py`, "
                "`tm_prior_retrain.py --model cnn/dsm`, `tm_ours.py --features "
                "magnetics/magnetics+rms` and `--baseline`, `tm_prior_published.py` "
                "(legacy and Tokamak-SI), `tm_agreement.py --exclude-test --tag _dev`, "
                "`tm_sensitivity.py`, `tm_gallery.py`, `tm_benchmark.py --rescore "
                "--gallery-reviewed`, `tm_write_doc.py`, `tm_render_tables.py`. Source "
                "JSONs and the shot lists are committed under the benchmark's "
                "`sources/`; predictions, weights, signals and figures stay under "
                "`$LABELER_ROOT/round4/tm/`."
            ),
            "",
            (
                "Remaining limitations: the target and the detector inputs share the "
                "magnetic RMS, the n = 2 level and the lock units are local "
                "conventions, weak and fast-locking modes are omitted, the published "
                "CNN's training overlap is unknown, the training sets of the retrained "
                "models are far smaller than the published ones, lock status is "
                "unknown on development shots without a `DUSBRADIAL` record, and there "
                "is no independent ECE island radius or fully nested hyperparameter "
                "selection."
            ),
        ]
    )
    DOC.write_text("\n".join(line.rstrip() for line in lines) + "\n")
    provenance = {
        "made_by": "scripts/labeler/tm_write_doc.py",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "benchmark_sha256": hashlib.sha256(BENCH.read_bytes()).hexdigest(),
        "document_sha256": hashlib.sha256(DOC.read_bytes()).hexdigest(),
        "document": str(DOC.relative_to(REPO)),
    }
    for path in (
        TM / "results/document_fix3.json",
        BENCH.parent / "sources/document_fix3.json",
    ):
        path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(DOC)


if __name__ == "__main__":
    main()
