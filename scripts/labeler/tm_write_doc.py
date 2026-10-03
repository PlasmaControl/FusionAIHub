#!/usr/bin/env python
"""Regenerate the TM methods/results document from current benchmark JSON sources."""

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


def metric(value):
    if value is None:
        return "—"
    return "{value:.3f} [{lo:.3f},{hi:.3f}]".format(**value)


def main():
    b = json.loads(BENCH.read_text())
    agreement = {
        ref: b["agreement"][f"{ref}_dev"]["agreement"]["n1"]
        for ref in ("seo", "survival")
    }
    weak = pd.read_csv(LABELS).query("shot == 189879 and category == 2")
    weak_end = float(weak.t_end.max()) / 1e3
    lines = [
        "# Whole-interval tearing-mode labels and magnetic-rule recovery",
        "",
        (
            "The target is a **strong rotating n=1/n=2 magnetic mode**, a tearing-mode "
            "proxy without independent island identification. `tm-ours` measures recovery "
            "of this magnetic rule; these results do not establish a better TM detector. "
            "AUROC and AUPRC are the primary comparisons; F1 depends on calibration."
        ),
        "",
        (
            "All new labeling, feature screening, fetching, agreement, fitting, scoring, "
            "coverage and galleries exclude the 50 cohort blind shots before opening "
            "their signals. The current cohort table contains the 450 development shots. "
            "Population results also exclude those IDs. Previous blind result JSONs were "
            "moved without inspection into `results/quarantine_blind_test/`."
        ),
        "",
        "## Rule and uncertainty",
        "",
        (
            "Both raw RMS and its 5 ms median must exceed 12 G (n1) or 6 G (n2) "
            "continuously for 50 ms, before joining runs. The n1 convention follows "
            "Farre-Kaga; n2 is a local extension. Each qualified seed extends to "
            "max(1 G, 10% of its peak). Available release gaps up to 50 ms may join; "
            "acquisition gaps cannot. The frozen development harmonic veto is n2/n1 "
            "≤0.57. EFIT rational surfaces alone do not determine m; no ECE island radius "
            "has been resolved, and m remains unassigned."
        ),
        "",
        (
            "Seed and span frequency evidence now uses `coherent_frequency`, not simply "
            "N1FREQ/N2FREQ lying in 1–30 kHz. It tests a local 50 ms "
            "percentile-stable frequency window with p90−p10 width ≤max(2 kHz, 25% of its median), "
            "and ≥80% actual coherent support over a span. Mirnov fallback requires "
            "n-resolved fit ≥0.9, prominence ≥10 dB, 1–30 kHz, and coherent amplitude "
            "above the frozen development quiet p95. Available >30 kHz records veto "
            "that mode's fallback. These criteria are magnetic proxies, not proof of an island."
        ),
        "",
        (
            "Weak tracks require a 100 ms coherent core and are uncertain, extended "
            "at the weak-line release floor across ≤50 ms evidence gaps. Population "
            "shots run the same Mirnov weak screening wherever existing raw inputs "
            "allow. Unscreened time above the frozen weak RMS thresholds "
            "(n1 2.0282 G, n2 1.8280 G) "
            "is uncertain rather than absent. Missing inputs are disclosed in metadata. "
            "Quiet time can be absent even without a weak-line screen. Weak modes "
            "that fail the screen still remain cohort **absent**: the weak 7 kHz n=2 "
            f"line on 189879 is uncertain until {weak_end:.1f} s; its fading "
            "continuation, below the fit/prominence screen, is absent. A review "
            "of the earlier labels saw the line to about 4.5 s. This is a "
            "strong-mode label, not exhaustive TM truth."
        ),
        "",
        "### Criterion pass rates",
        "",
        (
            "The source records below report the old in-range test, true coherent "
            "frequency and Mirnov fit/prominence criteria separately on absent and "
            "present samples, on the catalog-window RMS grid. Missing criterion inputs "
            "do not count as failures: the source gives their support denominators. "
            "These are diagnostic associations with the magnetic rule, not independent validation."
        ),
        "",
    ]
    criterion = b["rule_audit"]["criterion_support_fix2"]
    lines.extend(
        [
            "| Set / n | Criterion | Absent pass % (input s) | Present pass % (input s) |",
            "|---|---|---:|---:|",
        ]
    )
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
                        else f"{100 * value:.2f} ({rate['total_ms'] / 1000:.2f})"
                    )
                lines.append(f"| {scope} / {n} | {name} | {' | '.join(cells)} |")
    lines.append("")
    lines.extend(
        [
            "Source: [criterion_support_fix2.json](../../data/events/neoclassical_tearing_mode/benchmark/sources/criterion_support_fix2.json).",
            "",
            "## Abrupt collapse and locking",
            "",
            (
                "An amplitude fall from above seed to below release within ≤5 ms is never "
                "`ended=decay`. It is `ended=locked` only when independently confirmed, "
                "otherwise `ended=unknown`. The post-collapse phase is uncertain until "
                "the lock signal falls or the discharge ends; without confirmation/release "
                "evidence it remains uncertain to the catalog discharge end. The fetched "
                "n1 `DUSBRADIAL` PTDATA radial-field detector is in **volts**, not gauss. "
                "Confirmation requires ≥5 V continuously for 20 ms within 100 ms of the "
                "collapse or eligible frequency drop; release requires the field to stay below "
                "5 V for 200 ms, so shorter dips are bridged. "
                "This is a conservative local voltage convention, not a calibrated island-field "
                "measurement. `N1FREQ`/`N2FREQ` dropping to ≤1 kHz for 20 ms after rotation "
                "alone creates only a candidate. n2 has no independent radial confirmation."
            ),
            "",
            "| Set | Labeled shots | Intervals | Mode shots | Ends | Confirmed locks |",
            "|---|---:|---:|---:|---|---:|",
        ]
    )
    for name, meta in b["label_counts"].items():
        c = meta["counts"]
        lines.append(
            f"| {name} | {meta['n_labelled_shots']} | {c['n_intervals']} | "
            f"{meta['n_shots_with_a_mode']} | {c['intervals_by_end']} | {c['n_locked']} |"
        )
    lines.extend(
        [
            "",
            (
                "Source: benchmark `label_counts`, `locking_coverage`, and the adjacent label "
                "metadata. Unknown status is not evidence that no lock occurred. Fetching "
                "runs only on the login node through `fdp run` and stops on the first auth error."
            ),
            "",
            (
                "The `extend_tm_interval` table requires conversion before promotion to "
                "`review/`: TM catalog state 3 is forbidden, n-specific present/uncertain "
                "rows overlap, onset rows have zero length, and fractional-ms boundaries "
                "must become whole milliseconds. "
                "`test_shot_table_validates_extension_intervals_with_onset_points_and_spans` tests "
                "geometry, not catalog validity."
            ),
            "",
            "## Historical-onset agreement and limitations",
            "",
            (
                "Recall of the lab's archived onsets within 100 ms is **Seo "
                f"{agreement['seo']['matched']}/{agreement['seo']['reference_onsets']}** "
                f"and **survival {agreement['survival']['matched']}/"
                f"{agreement['survival']['reference_onsets']}** on the development shots "
                "(13/26 and 18/67 on the earlier 500-shot labels). The strong 50 ms seed "
                "rule omits short and fast-locking modes: of the missed onsets, "
                f"{agreement['seo']['missed_reasons']['short_burst']}/"
                f"{agreement['seo']['missed']} (Seo) and "
                f"{agreement['survival']['missed_reasons']['short_burst']}/"
                f"{agreement['survival']['missed']} (survival) are short bursts, "
                f"{agreement['seo']['missed_reasons']['coherent_line_not_supported']} and "
                f"{agreement['survival']['missed_reasons']['coherent_line_not_supported']} "
                "lack a supported coherent line. Strict "
                "containment and miss reasons are in the linked JSONs. Survival agreement "
                "is near-circular because it shares RMS, 12 G, 50 ms and release "
                "conventions. The rule was not selected to maximize archive agreement."
            ),
            "",
            "| Reference | Covered shots | Onsets | Matched | Strict contained |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for ref in ("seo", "survival"):
        a = b["agreement"][f"{ref}_dev"]
        summary = a["agreement"]["n1"]
        strict = a["agreement_strict"]["n1"]["matched"]
        lines.append(
            f"| {ref} | {summary['n_shots_covered']} | "
            f"{summary['reference_onsets']} | {summary['matched']} | {strict} |"
        )
    lines.extend(
        [
            (
                "Sources: benchmark `agreement`, [agreement_seo_cohort_dev.json](../../data/events/neoclassical_tearing_mode/benchmark/sources/agreement_seo_cohort_dev.json), "
                "[agreement_survival_cohort_dev.json](../../data/events/neoclassical_tearing_mode/benchmark/sources/agreement_survival_cohort_dev.json). "
                "The frozen-rule sensitivity is generated by `tm_sensitivity.py` and "
                "bundled as `sensitivity_survival_dev.json`; it does not tune the rule."
            ),
            "",
            "## Development detection benchmark",
            "",
            (
                "The original seed-0 outer held-shot assignment is unchanged. Before any fit, "
                "`tm_cv_plan.py` freezes inner shot roles in `inner_splits_fix2.json`, using "
                "10% per shot-has-an-interval stratum within each outer-training pool. "
                "Within-stratum swaps enforce observable positive support for every model "
                "and legacy target. No predicted score or held-fold performance chooses "
                "roles. Model availability is applied only after this shared plan. Each fit "
                "asserts positive-bearing early-stopping validation; threshold selection "
                "raises on zero positives and asserts no 0.999 fallback. Published CNN "
                "also appears at its own 0.5 threshold beside the retrained row. DSM's "
                "published survival threshold 0.7 corresponds to risk 0.3."
            ),
            "",
            (
                "All learned models are refitted with three seeds. Prior hyperparameters "
                "remain fixed from earlier development work, so this is not fully nested "
                "hyperparameter selection. The scalar CNN has t+25 ms input lookahead and "
                "is an offline detector. DSM is a detection head on its original embedding; "
                "published DSM rows remain horizon forecasts, with known original training "
                "shots excluded from their reported held-out scores. Published CNN original "
                "training overlap is unknown (dagger), so those scores are descriptive."
            ),
            "",
            (
                "Targets use absolute 10 ms bins. Legacy rows use their 25 ms native bins. "
                "Every row has its own input/reference-dependent shot set, given explicitly "
                "in its source JSON. Categories 2/3 and unavailable inputs/scores are excluded "
                "in primary rows. AUROC/AP use 1,024 score quantiles; all intervals use "
                "1,000 whole-shot bootstrap draws (seed 0). Segmental F1 uses IoU 0.5, "
                "50 ms minimum segments and ≤50 ms negative-gap closing; unavailable bins "
                "remain hard barriers and do not enter intersection or union."
            ),
            "",
            "| Model | Setting | Shots / bins | AUROC [95% CI] | AUPRC [95% CI] | F1 [95% CI] | Segmental F1 [95% CI] |",
            "|---|---|---:|---|---|---|---|",
        ]
    )
    for row in b["rows"]:
        m = row["metrics"]
        lines.append(
            " | ".join(
                [
                    "",
                    row["model"] + ("†" if not row["held_out"] else ""),
                    row["setting"],
                    f"{m['n_shots']} / {m['bins_scored']}",
                    *[metric(m.get(k)) for k in ("auroc", "auprc", "f1", "segf1_0.5")],
                    "",
                ]
            )
        )
    cov = b["coverage"]
    window_cov = b["rule_audit"]["audit_fix2_current"]["cohort"]["screening_coverage"]
    window_uncertain_percent = (
        100 * window_cov["uncertain_seconds"] / window_cov["window_seconds"]
    )
    lines.extend(
        [
            "",
            (
                "The Mirnov-derived mask excludes about "
                f"**{window_uncertain_percent:.0f}% of development "
                "catalog-window time**, equivalent to "
                f"**{100 * cov['ours']['uncertain_fraction']:.1f}%** of current "
                "observable development plasma. "
                "It shares `tm-ours` inputs, so the task emphasizes strong modes versus "
                "quiet magnetic time and can favor the magnetic detector. The "
                "uncertain-as-negative sensitivity row uses identical fits and validation "
                "thresholds, scores category 2 as negative, and still excludes category 3. "
                "Finite uncertain features remain in temporal input context; the target "
                "mask controls loss/scoring only. The +RMS row adds circular label inputs."
            ),
            "",
            "### Paired comparison",
            "",
            (
                f"Same {len(b['paired_common_shots']['shots'])} development shots / "
                f"{b['paired_common_shots']['bins_scored']} identical available 10 ms bins; "
                "each model retains its positive-bearing validation thresholds. Paired "
                "bootstrap draws resample the same shots. Ranking, not a threshold-specific "
                "F1 gain, is the primary comparison."
            ),
            "",
            "| Model | AUROC | AUPRC | F1 | Segmental F1 |",
            "|---|---|---|---|---|",
        ]
    )
    paired = b["paired_common_shots"]
    for name, m in paired["metrics"].items():
        lines.append(
            " | ".join(
                [
                    "",
                    name,
                    *[metric(m[k]) for k in ("auroc", "auprc", "f1", "segf1_0.5")],
                    "",
                ]
            )
        )
    lines.append(
        " | ".join(
            [
                "",
                "Difference, tm-ours−CNN",
                *[
                    metric(paired["difference_ours_minus_cnn"][k])
                    for k in ("auroc", "auprc", "f1", "segf1_0.5")
                ],
                "",
            ]
        )
    )
    ranking = b["cnn_ranking_common_bins"]
    lines.extend(
        [
            "",
            (
                f"Published/retrained CNN ranking on exactly {len(ranking['shots'])} "
                f"common shots and {ranking['bins_scored']} common 10 ms bins:"
            ),
            "",
            "| Model | AUROC [95% CI] | AUPRC [95% CI] |",
            "|---|---:|---:|",
        ]
    )
    for name in ("published", "retrained"):
        scores = ranking["metrics"][name]
        lines.append(
            f"| tm-onsetcnn-{name} | {metric(scores['auroc'])} | "
            f"{metric(scores['auprc'])} |"
        )
    lines.extend(
        [
            "",
            (
                "Source: benchmark `paired_common_shots` and `cnn_ranking_common_bins`. "
                "No retraining-generalization claim follows from published-CNN comparisons "
                "while original training overlap is unknown."
            ),
            "",
            "## Coverage and publication artifacts",
            "",
            (
                "**No TM coverage gain is claimed.** The preceding survival-matched "
                "like-for-like coverage was 435.7 s interval versus 909.9 s legacy; new "
                "conservative uncertainty further changes support. Current coverage uses "
                "the same measured plasma-start/catalog-end domain and 10 ms grid on each "
                "matched shot set. Observable time includes uncertainty; labeled time "
                "excludes it. Scoring can include quiet ramp-up, so scoring bins and "
                "observable-plasma coverage have different denominators."
            ),
            "",
            "| Set | Shots | Observable plasma s | Labeled s | Uncertain s |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in ("ours", "ours_population"):
        c = cov[name]
        lines.append(
            f"| {name} | {c['labelled_shots']} | {c['observable_plasma_seconds']:.2f} | "
            f"{c['labelled_seconds']:.2f} | {c['uncertain_seconds']:.2f} |"
        )
    lines.extend(
        [
            "",
            "| Matched reference | Shots | Legacy labeled s | Interval labeled s | Common labeled s |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in ("legacy_seo", "legacy_survival"):
        c = cov[name]
        lines.append(
            f"| {name} | {c['labelled_shots']} | {c['labelled_seconds']:.2f} | "
            f"{c['interval_seconds_on_same_shots']:.2f} | {c['common_labelled_seconds']:.2f} |"
        )
    lines.extend(
        [
            "",
            (
                "Sources: benchmark `coverage`, adjacent label metadata, and "
                "[figure2_tm.json](figure2_tm.json), which includes ranking/F1 intervals "
                "and exact like-for-like coverage. Cohort/population overlap and must not be added."
            ),
            "",
            (
                "`tm_gallery.py --width 3.25 --columns 1` provides column-sized example "
                "panels with 7.5 pt text at final width. The 7.3-inch complete MHR/Mirnov "
                "galleries remain supplementary audit material and must not be shrunk "
                "into a paper column. Their JSON sidecars record shots, width, font, "
                "diagnostic, label/source hashes and image hashes. All changed PNGs "
                "are visually inspected. `tm_render_tables.py` renders the exact final "
                "TeX at 6.75-inch text width, rejects overfull horizontal boxes and writes "
                "PDF/150-dpi PNG previews with provenance. The appendix explains the dagger."
            ),
            "",
            "## Reproduction",
            "",
            (
                "Use the prescribed frozen/no-install pixi labelmaker environment, "
                "worktree PYTHONPATH, scratch TMPDIR and LABELER_NO_FETCH=1. Only the "
                "authorized DUSBRADIAL fetch unsets LABELER_NO_FETCH and runs through "
                "`fdp run` on the login node. Train on CUDA_VISIBLE_DEVICES=1 with the "
                "prescribed phase3 CUDA venv; never use `--final`."
            ),
            "",
            (
                "Sequence: `tm_magfeatures.py`, `tm_label.py` (development and nonblind "
                "population), `tm_audit_rule.py`, `tm_cv_plan.py`, `tm_prior_retrain.py "
                "--model cnn/dsm`, `tm_ours.py --features magnetics/magnetics+rms` and "
                "`--baseline`, `tm_prior_published.py` (legacy and interval), "
                "`tm_agreement.py --exclude-test --tag _dev`, `tm_sensitivity.py`, "
                "`tm_gallery.py`, `tm_benchmark.py --rescore --gallery-reviewed`, "
                "`tm_write_doc.py`, `tm_render_tables.py`. Source JSONs and exact shot "
                "lists are committed under the benchmark's `sources/`; large predictions, "
                "weights, signals and figures stay under `$LABELER_ROOT/round4/tm/`."
            ),
            "",
            (
                "Remaining research limitations: magnetic target/input sharing, local n2 "
                "and voltage conventions, omissions of weak/fast-locking modes, unknown "
                "published CNN overlap, incomplete weak-screen inputs, and no independent "
                "ECE island radius or fully nested hyperparameter selection."
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
        TM / "results/document_fix2.json",
        BENCH.parent / "sources/document_fix2.json",
    ):
        path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(DOC)


if __name__ == "__main__":
    main()
