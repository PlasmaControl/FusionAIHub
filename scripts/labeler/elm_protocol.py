#!/usr/bin/env python
"""Render current ELM summaries from canonical JSON; preserve category history."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from labeler.config import git_sha, sha256_of
from labeler.elm import swap_tex

REPO = Path(__file__).resolve().parents[2]
OUTPUTS = REPO / "outputs/labeler/elm"


def metric(res, key):
    point = res["point"].get(key)
    if point is None or not math.isfinite(point):
        return "--"
    value = f"{point:.3f}"
    ci = res.get("ci95", {}).get(key)
    if ci is not None and math.isfinite(ci[0]):
        value += f" [{ci[0]:.3f}, {ci[1]:.3f}]"
    return value


def risk_metric(row):
    return metric(
        {"point": {"auroc": row["auroc"]}, "ci95": {"auroc": row["auroc_ci95"]}},
        "auroc",
    )


def patch_models(text, model_lines, caveat):
    """Update entries in all; retain stable, unknown entries and original dates."""
    begin, end = text.index("## Models"), text.index("## Inputs")
    section = text[begin:end]
    stable = next(
        line for line in section.splitlines() if line.startswith("**stable**:")
    )
    old_lines = re.findall(r"^- .*$", section, re.MULTILINE)
    aliases = {
        "elm_clock": "elm-clock",
        "elmo": "elm-elmo",
        "elm-dsm-detect": "elm-dsm (60-input 1×128 refit, detection)",
        "elm-dsm (detection)": "elm-dsm (60-input 1×128 refit, detection)",
    }
    entries = {}
    for line in old_lines:
        name = line[2:].split(" | ")[0]
        name = aliases.get(name, name)
        entries[name] = "- " + name + line[2 + len(line[2:].split(" | ")[0]) :]
    for line in model_lines:
        name = line[2:].split(" | ")[0]
        if name in entries:
            date = re.search(r"\d{4}_\d{2}_\d{2}", entries[name])
            if date:
                line = re.sub(r"\d{4}_\d{2}_\d{2}", date.group(), line, count=1)
        entries[name] = line
    bes_line = entries.pop("elm-ours (BES subset)", None)
    if bes_line is not None:
        ordered = {}
        for name, line in entries.items():
            if name == "elm-elmo":
                ordered["elm-ours (BES subset)"] = bes_line
            ordered[name] = line
        entries = ordered
    rendered = (
        "## Models\n" + stable + "\n\n"
        "**latest**: elm-ours | 2026_10_03\n\n**all**:\n"
        + "\n".join(entries.values())
        + "\n\n"
        + caveat
        + "\n\n"
    )
    return text[:begin] + rendered + text[end:]


def dsm_card(dsm, native):
    """Patch the current evaluation block; keep the adapter's historical details."""
    path = REPO / "src/labeler/models/d3d_elm_time_to_event_dsm/README.md"
    text = path.read_text()
    folds = dsm["detectors"]["elm-dsm-detect"]["folds"]
    context = dsm["model_context"]
    epochs = ", ".join(str(r["best_epoch"]) for r in folds)
    thresholds = ", ".join(f"{r['threshold']:.3f}" for r in folds)
    lines = [
        "## Evaluation",
        "",
        "### Reduced-input reviewed-label detection adaptation",
        "",
        "These are developmental shot-CV occupancy estimates on five fixed folds.",
        "Preliminary outer-fold predictions were available before the reported recipe",
        "was fixed and could have informed inputs, scaling, architecture, selection",
        "or evaluation; the saved records do not establish their influence.",
        "Inner-validation AUPRC selects checkpoints",
        f"at zero-based epochs {epochs}; inner-validation F1 selects",
        f"thresholds {thresholds}. Each fold fits its own normalization and starts",
        "from random weights. No blind-cohort shots enter these fits.",
        "",
        "| Common panel | Shots | 50 ms bins | AUROC [95% shot CI] | F1 [95% shot CI] |",
        "|---|---:|---:|---|---|",
    ]
    for tag, label in (("all119", "All reviewed"), ("bes73", "BES subset only")):
        panel = dsm["sets"][tag]
        row = panel["methods"]["elm-dsm-detect"]
        lines.append(
            f"| {label} | {panel['n_shots']} | {panel['bins']:,} | "
            f"{metric(row, 'auroc')} | {metric(row, 'f1')} |"
        )
    own = dsm["own_target"]
    lines += [
        "",
        (
            "Source: [dsm/evaluation.json](../../../../outputs/labeler/elm/dsm/"
            "evaluation.json), `detectors.elm-dsm-detect` and `sets`."
        ),
        "The detector uses 60 input columns, including measured PCPHD02/03 on",
        "119 shots and DENV2F/3F means on 115 per chord (four mean-filled).",
        "This is elm-dsm (60-input 1×128 refit, detection), a reduced-input",
        "adaptation trained and evaluated on 50 ms rows. The source model",
        "trained on native 1 ms rows with 124 inputs and layers [100, 1000]",
        "for WPQH breakthrough-ELM forecasting; this is not an objective-only",
        "retrain of that model. Fast-density units and filterscope sightlines",
        "are unverified in retained metadata; fixed input scaling, clipping and",
        "magnitude screening do not establish physical calibration.",
        "The companion occupancy U-Net omits FS01 because its retained cache",
        "contains FS02–04 only. Its fast-density inputs divide native values",
        "by `1e14`, clip to `[-3, 12]`, and clip ten times the 0.2 s high-pass",
        "to `[-10, 10]`; chords with median absolute native magnitude above",
        "`1e16` are zeroed by a heuristic failed-digitiser screen.",
        "Offline metadata audits made no new fetches and changed no saved",
        "inputs or weights. Sources: `density_units.json`,",
        "`filterscope_metadata.json` and `src/labeler/elm/inputs.py`.",
        "No independently validated physical-onset detector is delivered,",
        "and run days cross folds in both developmental analyses",
        "(16 review days; 21 of 31 Smith days).",
        "",
        "### Limited-input survival refit (selection evidence)",
        "",
        f"The served checkpoint is after {context['refit_checkpoint_epochs']} epoch",
        (
            f"of a {context['refit_run_epochs']}-epoch run, selected at best_epoch="
            f"{context['refit_best_epoch']}. Its {own['shots']} physical source-validation"
        ),
        "shots selected the checkpoint; they are not independent test evidence.",
        "",
        "| Horizon | AUROC [95% physical-shot CI] |",
        "|---|---|",
    ]
    for row in own["horizons"].values():
        lines.append(f"| {row['horizon_ms']:g} ms | {risk_metric(row)} |")
    exact = native["reviewed_exact_export"]["horizons"]["h50ms"]
    lines += [
        "",
        "Source: `dsm/evaluation.json:own_target.horizons`.",
        "",
        "### Native original-checkpoint audit",
        "",
        "The original 124-input, 1 ms checkpoint has embedding layers [100, 1000].",
        "The limited-input survival refit and reviewed detector instead use one",
        "128-unit layer. Native evaluation uses model9 parameter setting 1 with",
        "ReLU6; the Keras conversion's unbounded ReLU is not substituted.",
        "",
        "The corrected forward presence target asks whether a reviewed present",
        "span intersects (t, t+h]; the separate onset target asks whether a",
        "non-crowd start lies in that interval. Reviewed non-crowd starts",
        "are annotation boundaries without independent physical-onset truth.",
        "",
        f"Exact-export 50 ms AUROC is **{exact['auroc']:.3f}**, CI **null**:",
        (
            f"descriptive only on {exact['n_shots']} shots reused in source fitting and "
            f"{exact['rows']:,} rows"
        ),
        "(" + ", ".join(map(str, exact["shots"])) + ").",
        "196541 entered optimizer fitting; the other three entered checkpoint",
        "selection; all entered source normalization. Five shots have exact exports,",
        "but 192751 has no scored overlap: the exact-export panel and figure",
        "therefore contain four shots. No operating threshold is selected.",
        "",
        "| Native panel | Horizon | Shots | Rows | AUROC [95% physical-shot CI] |",
        "|---|---|---:|---:|---|",
    ]
    for key, label in (
        ("reviewed_reconstructed", "Reconstructed presence"),
        ("own_target", "Source selection target"),
    ):
        for horizon, row in native[key]["horizons"].items():
            lines.append(
                f"| {label} | {horizon[1:-2]} ms | {row['n_shots']} | "
                f"{row['rows']:,} | {risk_metric(row)} |"
            )
    lines += [
        "",
        "Reconstruction is a sensitivity: source smoothing of concatenated",
        "phase rows differs from within-shot NBI smoothing. Source validation",
        "retains the original reversed chronological split and selected weights.",
        "It is not independent evaluation. Coverage, inputs, targets and memberships",
        (
            "are in [native_evaluation.json](../../../../outputs/labeler/elm/dsm/"
            "native_evaluation.json)."
        ),
        "Reproduce scoring without training with `elm_dsm_evaluate.py --rescore`",
        "and `elm_dsm_native.py`; render this block with `elm_protocol.py`.",
        "",
    ]
    begin = text.index("## Evaluation")
    end = text.index("### BES ablation", begin)
    path.write_text(text[:begin] + "\n".join(lines) + "\n" + text[end:])
    return path


def main():
    files = {
        "ours": "ours/evaluation.json",
        "dsm": "dsm/evaluation.json",
        "native": "dsm/native_evaluation.json",
        "swap": "swap/evaluation.json",
        "feature": "ours/feature_only.json",
        "smith": "smith/evaluation.json",
        "strata": "ours/annotation_strata.json",
        "offsets": "review_start_offsets.json",
        "density": "density_units.json",
        "filterscope": "filterscope_metadata.json",
        "history": "training_history.json",
    }
    records = {
        key: json.loads((OUTPUTS / value).read_text()) for key, value in files.items()
    }
    ours, dsm, native, smith = (records[k] for k in ("ours", "dsm", "native", "smith"))
    all_ours = ours["sets"]["all119"]["methods"]["elm-ours"]
    head = smith["methods"]["elm-ours-onset"]["events"]["2"]
    frozen = smith["methods"]["elm-ours"]["occupancy_1ms"]
    audit = smith["onset_window_audit"]
    swap = records["swap"]["interval_audit"]["known_review_majority"]
    exact = native["reviewed_exact_export"]["horizons"]["h50ms"]
    mean_auc = ours["sets"]["all119"]["fold_auroc_sensitivity"]["mean"]
    all_common = dsm["sets"]["all119"]
    bes_common = dsm["sets"]["bes73"]
    paired = []
    for key, label in (("auroc", "AUROC"), ("auprc", "AUPRC"), ("f1", "F1")):
        row = bes_common["paired"][f"elm-ours - elm-elmo: {key}"]
        lo, hi = row["ci95"]
        paired.append(f"{label} {row['value']:+.3f} [{lo:.3f}, {hi:.3f}]")
    paired_text = ", ".join(paired)
    offsets = records["offsets"]
    timing = offsets["offset_ms"]
    timing_lines = [
        (
            f"Among {offsets['n_starts']} non-crowd reviewed starts on "
            f"{offsets['n_shots_with_non_crowd_spans']} BES-subset shots,"
        ),
        (
            f"{offsets['n_matched']} starts on {offsets['n_matched_shots']} shots match "
            "the nearest ELM-O onset within ±50 ms."
        ),
        "For that matched subset only, reviewed start minus ELM-O onset has",
        (
            f"median {timing['median']:.3f} ms and quartiles "
            f"[{timing['p25']:.3f}, {timing['p75']:.3f}] ms;"
        ),
        (
            f"{offsets['n_unmatched']} unmatched starts are omitted. Each start is "
            "matched independently, allowing onset reuse."
        ),
        "This annotation-to-detector offset is not a measured physical-onset error;",
        "neither reference provides independently verified physical-onset truth.",
        (
            "Source: `review_start_offsets.json`, generated by "
            "`elm_review_start_offsets.py`."
        ),
    ]
    lines = [
        "# ELM occupancy and onset evaluation",
        "",
        "`elm-ours` delivers BES-free ELMy-occupancy probability; the requested",
        "finer physical-onset trace is **not delivered to the catalog**.",
        "",
        "Current numbers and what each means (sources below):",
        "",
        (
            f"- {all_ours['point']['auroc']:.3f} AUROC: reviewed occupancy, "
            f"{ours['sets']['all119']['n_shots']} shots / "
            f"{ours['sets']['all119']['bins']:,} interior 50 ms bins."
        ),
        f"- {metric(all_ours, 'f1')} F1: fold thresholds selected on inner validation.",
        f"- {mean_auc:.3f}: mean per-fold AUROC sensitivity to pooling fold scores.",
        (
            f"- {all_common['methods']['elm-ours']['point']['auroc']:.3f} / "
            f"{bes_common['methods']['elm-ours']['point']['auroc']:.3f}: elm-ours "
            "AUROC on all / BES-subset DSM-common bins."
        ),
        (
            f"- {all_common['methods']['elm-dsm-detect']['point']['auroc']:.3f} / "
            f"{bes_common['methods']['elm-dsm-detect']['point']['auroc']:.3f}: "
            "reduced-input DSM detection adaptation on all / BES-common bins."
        ),
        f"- {exact['auroc']:.3f}: native DSM forward-presence AUROC, four shots, no CI.",
        f"- {metric(frozen, 'auroc')}: frozen review-to-Smith occupancy AUROC.",
        f"- {metric(head, 'recall')}: Smith-trained onset recall in selected windows.",
        (
            f"- {audit['max_absolute_error_ms']:.2f} ms / "
            f"{100 * audit['correct_1ms_cell_fraction']:.0f}%: max matched error / correct "
            "1 ms cell."
        ),
        (
            f"- {swap['M']} / {swap['P']}: legacy/review disagreements; eight-shot swap "
            "is inconclusive."
        ),
        "",
        "## Scope and review benchmark",
        "",
        "The 401,714-parameter 1D U-Net reads FS02–04 log D-alpha and DENV2F/3F",
        "density, without BES. The target is reviewed ELMy occupancy; 97% of",
        "positive bins come from crowd spans. Review started from the clock:",
        "within 1 ms, 56% of crowd starts, 44% of ends and 33% of both match it.",
        "FS01 was omitted: the retained input cache contains FS02–04 only.",
        "Reviewed non-crowd starts are annotation boundaries without independent",
        "physical-onset truth.",
        "",
        *timing_lines,
        "",
        "All reported review fits are **developmental shot-CV estimates**.",
        "Preliminary predictions on outer-fold shots were available before the",
        "reported recipe was fixed and could have informed input choice, scaling,",
        "architecture, checkpoint or threshold selection, or the evaluation",
        "protocol; saved records do not establish whether or how much they",
        "influenced these choices. Five shot-grouped",
        "folds cover 69 cohort-train and 50 validation shots; no blind-test shot",
        "enters isolated fits, normalization or threshold selection. The original",
        "occupancy checkpoints and thresholds stay frozen.",
        "No independently validated physical-onset detector is delivered, and",
        "run days cross folds in both developmental analyses",
        "(16 review days; 21 of 31 Smith days).",
        "",
        "`elm-ours` **matches ELM-O without BES; extends coverage to all 119 shots**.",
        f"On the same {bes_common['bins']:,} bins from {bes_common['n_shots']} BES shots,",
        f"paired elm-ours minus ELM-O differences are {paired_text}.",
        "Every interval includes zero; this does not establish superiority.",
        "Each 50 ms bin lies wholly inside a reviewed span and shared signal",
        "coverage, with valid rows in both DSM input variants. elm-ours calls",
        "a bin present when its mean probability reaches the selected threshold;",
        "the DSM adaptation thresholds one aligned row score computed from",
        "50 ms input means. ELM-O and the clock use any detected-span touch.",
        "",
        "| Coverage / method | Shots / bins | AUROC [95% shot CI] | AUPRC | F1 |",
        "|---|---:|---|---|---|",
    ]
    model_lines = []
    rows = (
        ("elm-ours", "elm-ours"),
        ("elm-dsm-detect", "elm-dsm (60-input 1×128 refit, detection)"),
        ("elm-clock", "elm-clock"),
        ("always present", "always-present"),
        ("elm-feature-only", "elm-feature"),
        ("elm-elmo", "elm-elmo"),
    )
    for tag in ("all119", "bes73"):
        panel = dsm["sets"][tag]
        for key, label in rows:
            row = (
                records["feature"]["sets"]["common"][tag]["methods"][key]
                if key == "elm-feature-only"
                else panel["methods"].get(key)
            )
            if row is None:
                continue
            cells = [metric(row, key) for key in ("auroc", "auprc", "f1")]
            lines.append(
                f"| {tag} / {label} | {panel['n_shots']} / "
                f"{panel['bins']:,} | " + " | ".join(cells) + " |"
            )
            if tag == "all119" or key in ("elm-ours", "elm-elmo"):
                model_label = (
                    "elm-ours (BES subset)"
                    if tag == "bes73" and key == "elm-ours"
                    else label
                )
                model_lines.append(
                    f"- {model_label} | 2026_10_03 | AUROC: {cells[0]} | "
                    f"AUPRC: {cells[1]} | F1: {cells[2]} "
                    f"({panel['n_shots']} shots / "
                    f"{panel['bins']:,} common bins)"
                )
    model_lines.append(
        "- elm-ours-onset | 2026_10_03 | Recall ±2/5 ms: "
        + metric(head, "recall")
        + f" | Matched errors ≤{audit['max_absolute_error_ms']:.2f} ms; "
        f"{100 * audit['correct_1ms_cell_fraction']:.0f}% correct 1 ms "
        "cell (211 Smith shots; developmental selected-window shot CV)"
    )
    refit = dsm["sets"]["all119"]["methods"]["elm-dsm"]
    model_lines.append(
        "- d3d_elm_time_to_event_dsm | 2026_09_06 | AUROC: "
        + metric(refit, "auroc")
        + " | AUPRC: "
        + metric(refit, "auprc")
        + " | F1: "
        + metric(refit, "f1")
        + " (elm-dsm refit; supplemental offline risk score reusing source data; "
        f"{all_common['n_shots']} shots / {all_common['bins']:,} common bins)"
    )
    lines += [
        "",
        "The detection DSM is a reduced-input adaptation trained and evaluated",
        "on 50 ms rows: 60 input columns and one 128-unit layer. PCPHD02/03",
        "are measured on all 119 shots; DENV2F and DENV3F means supply the two",
        "density columns on 115 shots per chord, with four per chord mean-filled.",
        "The source DSM trained on native 1 ms rows with 124 inputs and layers",
        "[100, 1000] for WPQH breakthrough-ELM forecasting. The detection row",
        "is not an objective-only retrain of that model; comparative claims",
        "apply to this reduced-input 50 ms adaptation. Historical survival and",
        "detection variants that reuse source weights or statistics remain",
        "supplemental; their upstream normalization includes two blind-cohort",
        "shots. Native presence and non-crowd-start targets use forward (t,t+h].",
        "",
        "### Signal provenance and numerical preprocessing",
        "",
        "The retained caches do not establish fast-density physical ordinate",
        "units or FS01–04 sightlines (divertor versus midplane). FS01 is not",
        "an input to the occupancy U-Net. Paired slow CO2 checks numerical",
        "scales but cannot determine the fast-channel units. Filterscope",
        "levels use `(log10(max(x, 1e12)) - 15) / 1.5`, with contrast to a",
        "0.5 s running median. Fast density is divided by `1e14` native",
        "ordinate units and clipped to `[-3, 12]`; ten times the 0.2 s",
        "high-pass is clipped to `[-10, 10]`. A chord whose median absolute",
        "native magnitude exceeds `1e16` is zeroed by the heuristic failed-",
        "digitiser screen. These are fixed numerical choices, not verified",
        "physical calibration or a validated diagnostic-failure criterion.",
        "Offline metadata audits changed no saved input values, clipping,",
        "screening or weights and made no new fetches. Sources:",
        "`density_units.json`, `filterscope_metadata.json`, and",
        "`src/labeler/elm/inputs.py`.",
        "",
        "Intervals use 1,000 physical-shot resamples, with valid/undefined counts.",
        "Each endpoint needs five denominator-bearing shots; false-alarm rates",
        "on negative-only shots retain their intervals. The saved per-fold AUROCs",
        "provide a sensitivity to pooling differently calibrated fold scores.",
        "",
        "## Smith onset and transfer",
        "",
        "Frozen review-to-Smith transfer is independent: no shared review shots",
        "or run days. The Smith-trained head is developmental shot CV with",
        "within-Smith run-day sharing. Overlap with ELM-O's historical tuning",
        "events is unknown because event membership was not retained.",
        "",
        f"The head recalls {metric(head, 'recall')} of 2,316 hand-labelled windows",
        (
            "on 211 shots. Every matched error is within "
            f"{audit['max_absolute_error_ms']:.2f} ms; "
            f"{100 * audit['correct_1ms_cell_fraction']:.0f}% lie in the correct 1 ms cell."
        ),
        "**Every method's precision/F1 is conditional on selected windows;**",
        "**continuous-discharge precision/F1 is unavailable for every method.**",
        "False positives are counted only inside windows, each holding one",
        "labelled event, while",
        f"{audit['out_of_window_firings']:,} additional firings lie outside. Those",
        "firings lack negative truth. The 10 ms minimum peak separation further",
        "restricts in-window false positives. Experimental traces are available",
        "under `$LABELER_ROOT/round4/elm/smith/cv/pred/`; catalog onsets stay withheld.",
        "ELM-O region-overlap recall is now 0.986 after the 179859 time-axis repair;",
        "region overlap, onset matching and 1 ms occupancy are distinct targets.",
        "",
        "## Reference swap",
        "",
        "The legacy onset table overlaps eight review shots (seven with BES).",
        (
            f"On {swap['bins']} known all-covered 50 ms cells, M={swap['M']} and "
            f"P={swap['P']}; these are definition disagreements, not adjudicated events."
        ),
        "Unknown time stays unknown. Fixed predictions are also compared on 641",
        "strict interior bins. Only AUROC compares references; F1 thresholds were",
        "review-tuned. Covered-gap merges at 100/200/300 ms are sensitivities that",
        "never bridge missing coverage. Subsets excluded from upstream training,",
        "normalization and selection (3 shots; 2 with BES) are too small for",
        "intervals; values are in the JSON. No population",
        "ranking reversal is established.",
        "",
        "## Sources and reproduction",
        "",
        "All paths below are relative to `outputs/labeler/elm/`:",
        "",
        "- `ours/evaluation.json:sets`: original occupancy and per-fold AUROCs.",
        "- `dsm/evaluation.json:{sets,detectors,own_target}`: common bins and refits.",
        "- `dsm/native_evaluation.json`: native forward targets and memberships.",
        "- `ours/feature_only.json`, `ours/annotation_strata.json`: controls and labels.",
        "- `smith/evaluation.json:{methods,onset_window_audit,onset_run_day_audit,protocol}`: onset scope.",
        "- `swap/evaluation.json:{swap,interval_audit}`: fixed-prediction swap.",
        "- `review_start_offsets.json`: reviewed-start/ELM-O matched-subset timing.",
        "- `training_history.json`: preliminary predictions and review run-day sharing.",
        "- `density_units.json`, `filterscope_metadata.json`: signal provenance.",
        "",
        "Use the mandated pixi labelmaker wrapper with LABELER_NO_FETCH=1:",
        "Read the frozen run name from `ours/evaluation.json:run`, then use",
        "`elm_ours_evaluate.py --run <saved-run>`,",
        "`elm_dsm_evaluate.py --run <saved-run> --rescore`,",
        "`elm_smith_evaluate.py evaluate`, `elm_paper_tables.py`,",
        "`elm_example_figure.py --run <saved-run>`, then `elm_protocol.py`.",
        "Figures (PDF / 150 dpi PNG), checkpoints and predictions live under",
        "`$LABELER_ROOT/round4/elm/`; paper tables remain in this worktree.",
        "The original ELM-O benchmark doc is retained with a dated update.",
        "",
    ]
    doc = REPO / "docs/labeler/elm_ours.md"
    doc.write_text("\n".join(lines))
    readme = REPO / "data/events/edge_localized_mode/README.md"
    caveat = (
        "Brackets are 95% shot-bootstrap intervals. Review results are "
        "developmental shot-CV occupancy estimates (97% crowd positives; "
        "clock-seeded review). `elm-ours` matches ELM-O without BES; "
        "extends coverage to all 119 shots. On the common BES subset, paired "
        f"elm-ours minus ELM-O is {paired_text}; every interval includes zero. "
        "Source: [dsm/evaluation.json](../../../outputs/labeler/elm/dsm/"
        "evaluation.json), `sets.bes73.paired`. Catalog physical-onset output "
        "is withheld. Every Smith method's precision/F1 is conditional on "
        "selected windows; continuous-discharge precision/F1 is unavailable. "
        "Inputs, run-day sharing and timing limits: "
        "[elm_ours.md](../../../docs/labeler/elm_ours.md)."
    )
    readme.write_text(patch_models(readme.read_text(), model_lines, caveat))
    card = dsm_card(dsm, native)
    manifest = {
        "git": git_sha(),
        "sources": {
            key: {"path": path, "sha256": sha256_of(OUTPUTS / path)}
            for key, path in files.items()
        },
        "artifacts": {
            str(p.relative_to(REPO)): sha256_of(p) for p in (doc, readme, card)
        },
        "model_names": {key: swap_tex.method_label(key) for key, _ in rows},
    }
    (OUTPUTS / "protocol.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print("protocol and DSM card written; ELM-O historical doc preserved")


if __name__ == "__main__":
    main()
