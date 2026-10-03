#!/usr/bin/env python
"""Render the ELM summaries from the canonical JSON records.

Writes `docs/labeler/elm_ours.md`, patches the `## Models` block of
`data/events/edge_localized_mode/README.md` and the evaluation section of the DSM
adapter card, and records the sources in `outputs/labeler/elm/protocol.json`. Every
figure is read from the evaluation records (through `labeler.elm.facts` or the
record itself); a re-run of an evaluation changes the text with it.
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

from labeler.config import git_sha, sha256_of
from labeler.elm import facts as F
from labeler.elm import swap_tex

REPO = Path(__file__).resolve().parents[2]
OUTPUTS = REPO / "outputs/labeler/elm"
DOC = REPO / "docs/labeler/elm_ours.md"
README = REPO / "data/events/edge_localized_mode/README.md"
CARD = REPO / "src/labeler/models/d3d_elm_time_to_event_dsm/README.md"
DATE = "2026_10_03"
WIDTH = 88
LOWER_BOUND = (
    "These DSM detection rows are lower bounds on DSM detection skill under our "
    "recipe, not the best achievable DSM performance."
)
DSM_LABEL = "elm-dsm (60-input 1×128 refit, detection)"
NATIVE_LABEL = "elm-dsm (124-input [100,1000] detection)"
BOUNDARY = "../../../../outputs/labeler/elm"


def metric(res, key):
    """`0.941 [0.906, 0.967]` from a method result; `--` where undefined."""
    return F.fmt(res["point"].get(key), res.get("ci95", {}).get(key))


def risk_metric(row):
    return F.fmt(row["auroc"], row["auroc_ci95"])


def pct(value: float) -> str:
    return f"{100 * value:.0f}"


def plain(res, key):
    """A point value without its interval; `--` where undefined."""
    return F.fmt(res["point"].get(key))


def wrap(text: str, width: int) -> list[str]:
    """Wrap at spaces only, so paths and hyphenated words stay whole."""
    return textwrap.wrap(text, width, break_long_words=False, break_on_hyphens=False)


class Doc:
    """Markdown lines with wrapped paragraphs and pipe tables."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def heading(self, level: int, text: str) -> None:
        self.lines += ["#" * level + " " + text, ""]

    def para(self, *parts: str) -> None:
        self.lines += wrap(" ".join(parts), WIDTH) + [""]

    def bullets(self, items) -> None:
        for item in items:
            wrapped = wrap(item, WIDTH - 2)
            self.lines += ["- " + wrapped[0]] + ["  " + w for w in wrapped[1:]]
        self.lines.append("")

    def table(self, header, rows, right=()) -> None:
        marks = ["---:" if i in right else "---" for i in range(len(header))]
        self.lines.append("| " + " | ".join(header) + " |")
        self.lines.append("|" + "|".join(marks) + "|")
        self.lines += ["| " + " | ".join(row) + " |" for row in rows]
        self.lines.append("")

    def text(self) -> str:
        return "\n".join(self.lines).rstrip() + "\n"


def load_records() -> dict:
    records = F.load(OUTPUTS)
    records["input_audit"] = json.loads(
        (OUTPUTS / "dsm/detection_input_audit.json").read_text()
    )
    return records


def method_rows(source: dict, scope: str, records: dict, tags=("all119", "bes73")):
    """Table rows `tag / method` for the benchmark, in the paper's order."""
    order = (
        ("elm-ours", "elm-ours"),
        ("elm-elmo", "ELM-O"),
        ("elm-dsm-detect", DSM_LABEL),
        ("elm-clock", "elm-clock"),
        ("always present", "always-present"),
        ("elm-feature-only", "elm-feature"),
    )
    rows = []
    for tag in tags:
        panel = source["sets"][tag]
        for key, label in order:
            if key == "elm-dsm-detect" and scope == "primary":
                continue
            res = (
                records["feature"]["sets"][scope][tag]["methods"][key]
                if key == "elm-feature-only"
                else panel["methods"].get(key)
            )
            if res is None:
                continue
            rows.append((tag, key, label, panel, res))
    return rows


def primary_table(doc: Doc, records: dict) -> None:
    header = [
        "Coverage / method",
        "Shots / bins",
        "AUROC [95% shot CI]",
        "AUPRC",
        "F1",
        "Absent-bin FPR",
        "Absent-span alarm",
    ]
    rows = []
    for tag, _, label, panel, res in method_rows(records["ours"], "primary", records):
        rows.append(
            [
                f"{tag} / {label}",
                f"{panel['n_shots']} / {panel['bins']:,}",
                metric(res, "auroc"),
                *(plain(res, m) for m in ("auprc", "f1")),
                plain(res, "false_alarm_bin_rate"),
                plain(res, "absent_span_alarm_rate"),
            ]
        )
    doc.table(header, rows)


def moved_table(doc: Doc, fx: dict) -> None:
    moved = fx["moved"]
    if not moved:
        return
    rows = []
    for name, res in moved["methods"].items():
        rows.append(
            [
                name,
                metric(res, "auroc"),
                *(plain(res, m) for m in ("auprc", "f1", "false_alarm_bin_rate")),
            ]
        )
    doc.table(
        ["bes73 moved-boundary bins / method", "AUROC [95% CI]", "AUPRC", "F1", "FPR"],
        rows,
    )


def repeats_table(doc: Doc, fx: dict) -> None:
    seeds = fx["dsm_baselines"]
    if not seeds:
        return
    rows = []
    for key, label in (("reduced", DSM_LABEL), ("native", NATIVE_LABEL)):
        row = seeds[key]
        scope = f"{row['n_shots']} / {row['bins']:,}"
        rep, mean = row["reported"]["all119"], row["all119"]
        rows.append(
            [label, scope, "reported (raw selection)"]
            + [f"{rep[m]:.3f}" for m in F.METRIC_KEYS]
        )
        rows.append(
            [label, scope, "post-warm-up repeats, mean (range)"]
            + [
                f"{mean[m]['mean']:.3f} ({mean[m]['min']:.3f}–{mean[m]['max']:.3f})"
                for m in F.METRIC_KEYS
            ]
        )
    doc.table(
        ["Detector", "Shots / bins", "Fit", "AUROC", "AUPRC", "F1"],
        rows,
    )


def detection_inputs(audit: dict) -> str:
    """How the 60 detection columns are sourced, from the input audit."""
    cols = audit["column_sources"]

    def count(column: str, *, rejected: bool, match: str = "") -> int:
        return sum(
            row["n_shots"]
            for name, row in cols[column].items()
            if (("rejected" in name) == rejected) and match in name
        )

    n = audit["n_shots"]
    fetched = count("pcphd02", rejected=False, match="fetched")
    upstream = count("pcphd02", rejected=False, match="upstream")
    return (
        f"PCPHD02/03 means come from a fresh fetch on {fetched} of {n} shots and "
        f"from the upstream WPQH PCPHD02/03 export on {upstream}; DENV2F and DENV3F "
        "means supply the two density columns on "
        f"{count('co2_v2', rejected=False)} and {count('co2_v3', rejected=False)} "
        f"shots, with {count('co2_v2', rejected=True)} and "
        f"{count('co2_v3', rejected=True)} rejected (failed digitiser) and "
        "mean-filled."
    )


def audit_blocks(doc: Doc, records: dict, fx: dict, level: int = 3, full=True) -> None:
    """The native DSM detection comparator, with the DSM repeats.

    `full` writes the method paragraphs (the DSM card); the document keeps one
    summary sentence, the table and the repeats.
    """
    native = records["native_detection"]
    panel_shots = native["sets"]["all119"]["n_shots"]
    doc.heading(level, "Native DSM detection comparator")
    if full:
        doc.para(
            "The timestamp-aware 1 ms audit finds at least 112/124 inputs on "
            f"{native['shots_at_least_90_percent']} reviewed shots; {panel_shots} "
            "have complete 124-input scored bins. Missing column names and shot "
            "counts are in "
            "`dsm/native_detection.json:coverage,missing_column_shot_counts`."
        )
        doc.para(
            "The native [100,1000] ReLU6 architecture was refitted for occupancy on "
            "complete-input shots only, with random weights and optimizer-training-"
            f"only normalization. The fixed {native['recipe']['epochs']}-epoch recipe "
            "uses the original five outer folds and their inner-validation shot "
            "partitions; checkpoint AUPRC and F1 thresholds are selected only on "
            "inner validation. No source weights or statistics and no blind-test "
            "shots are reused."
        )
        doc.para(
            "Inputs are timestamp-aware 1 ms means from stored original corpus H5 "
            "records and retained PCPHD02/03. Standardized inputs are clipped at "
            "±10; NBI uses the source's 100-row centered smoothing within each shot "
            "rather than across concatenated source phases. This reconstructs "
            "native diagnostic inputs, not bit-identical historical exported rows. "
            "Bin scores average the 50 native row probabilities; measured support "
            "and target are identical for every compared method."
        )
    else:
        doc.para(
            "The native [100,1000] ReLU6 architecture, refitted for occupancy "
            f"({native['recipe']['epochs']} epochs, original folds, inner-validation "
            f"selection, no source weights or blind-test shots) on the {panel_shots} "
            "shots with complete 124-input rows, is scored on identical support. "
            "Inputs and method: the DSM card's Evaluation section."
        )
    cell = metric if full else plain
    rows = []
    for tag in ("all119", "bes73"):
        panel = native["sets"][tag]
        for key, label in (
            ("elm-ours", "elm-ours"),
            ("elm-dsm-detect", DSM_LABEL),
            ("elm-dsm-native-detect", NATIVE_LABEL),
            ("elm-elmo", "ELM-O"),
        ):
            res = panel["methods"].get(key)
            if res:
                rows.append(
                    [
                        f"{tag} / {label}",
                        f"{panel['n_shots']} / {panel['bins']:,}",
                        metric(res, "auroc"),
                        *(cell(res, m) for m in ("auprc", "f1")),
                    ]
                )
    doc.table(
        [
            "Matched panel / method",
            "Shots / bins",
            "AUROC [95% shot CI]",
            "AUPRC",
            "F1",
        ],
        rows,
    )
    doc.para(
        "This smaller support panel is a secondary control and does not replace the "
        "primary all119/bes73 benchmark."
    )
    seeds = fx["dsm_baselines"]
    if seeds:
        doc.heading(level, "DSM baselines retrained with post-warm-up selection")
        doc.para(
            "Both detection variants were refitted with "
            f"{swap_tex.count_word(len(seeds['seeds']))} further seeds on the same "
            "outer folds and recipe, choosing the epoch that ends the best "
            "three-epoch mean inner-validation AUPRC window lying wholly at or after "
            f"the warm-up ({seeds['reduced']['warmup_epochs']} of "
            f"{seeds['reduced']['total_epochs']} epochs for the 60-input adaptation, "
            f"{seeds['native']['warmup_epochs']} of {seeds['native']['total_epochs']} "
            "for the native refit); the reported fit takes the raw best epoch. "
            "Ranges are over seeds, not intervals. Selected epochs span "
            f"{seeds['reduced']['epochs'][0]}–{seeds['reduced']['epochs'][1]} and "
            f"{seeds['native']['epochs'][0]}–{seeds['native']['epochs'][1]}; native "
            f"thresholds span {seeds['native']['thresholds'][0]:.4f}–"
            f"{seeds['native']['thresholds'][1]:.3f}, an erratic operating point."
        )
        repeats_table(doc, fx)
        doc.para(LOWER_BOUND, "Source: `dsm/baseline_seeds.json`.")


def rejection_block(doc: Doc, records: dict) -> None:
    rejection = records["rejection"]
    doc.heading(3, "Diagnostic rejection and inference sensitivity")
    doc.para(
        f"Counts use all119 primary coverage ({rejection['n_shots']} shots / "
        f"{rejection['bins']:,} bins); a bin is affected if any valid 0.1 ms input "
        "cell is screened or clipped. Clipping discards no shot or bin, and its "
        "counts exclude already screened chords. Per-shot magnitudes are in "
        "`ours/rejection_sensitivity.json`."
    )
    doc.table(
        ["Chord", "Screen shots / bins", "Clipping shots / bins", "Threshold"],
        [
            [
                name,
                f"{row['screen_shots']} / {row['screen_bins']}",
                f"{row['clipping_shots']} / {row['clipping_bins']}",
                row["threshold"],
            ]
            for name, row in rejection["totals"].items()
        ],
    )
    before, after = (
        rejection["methods"][k] for k in ("screen-enabled", "screen-disabled")
    )
    change = rejection["metric_change_disabled_minus_enabled"]
    doc.para(
        "The failed-digitiser heuristic applies only to DENV2F/3F (median absolute "
        "native 0.1 ms cell mean above 1e16); the primary U-Net zero-fills a "
        "rejected chord's two features. Using rejected chords raw, with frozen "
        f"weights, thresholds and bins and no retraining, changes inputs on "
        f"{len(rejection['affected_shots'])} shots: AUROC, AUPRC and F1 are "
        + ", ".join(plain(after, m) for m in ("auroc", "auprc", "f1"))
        + " (changes "
        + ", ".join(f"{change[m]:+.3f}" for m in ("auroc", "auprc", "f1"))
        + ") and the false-alarm bin rate moves from "
        f"{before['point']['false_alarm_bin_rate']:.3f} to "
        f"{after['point']['false_alarm_bin_rate']:.3f}. This does not establish "
        "physical calibration or prove that screened chords are faulty."
    )


def provenance_note(records: dict) -> str:
    """Where each record's provenance lives, checked against the JSON files."""
    missing = []
    for path in sorted(OUTPUTS.rglob("*.json")):
        body = json.loads(path.read_text())
        if isinstance(body, dict) and "git" not in body:
            missing.append(str(path.relative_to(OUTPUTS)))
    text = (
        "The consolidated `provenance.json` was removed in round seven; each "
        "record carries its own top-level `git` commit (most also a creation "
        "stamp and a script digest), which replaces it."
    )
    if missing:
        text += (
            " Records without a `git` field ("
            + ", ".join(f"`{m}`" for m in missing)
            + ") name their sources and digests instead."
        )
    return text


def model_lines(records: dict, fx: dict) -> list[str]:
    """README `## Models` entries; `bes73` is a scope of elm-ours, not a model."""
    ours, dsm = records["ours"], records["dsm"]
    sets = ours["sets"]
    common = dsm["sets"]["all119"]

    def triple(res):
        return " | ".join(
            f"{name}: {metric(res, key)}"
            for name, key in (("AUROC", "auroc"), ("AUPRC", "auprc"), ("F1", "f1"))
        )

    def scope(tag):
        return f"{sets[tag]['n_shots']} shots / {sets[tag]['bins']:,} bins"

    own = sets["all119"]["methods"]["elm-ours"]
    bes = sets["bes73"]["methods"]["elm-ours"]
    lines = [
        (
            f"- elm-ours | {DATE} | {triple(own)} (primary all119; {scope('all119')}; "
            f"on the bes73 scope, {scope('bes73')}: AUROC {metric(bes, 'auroc')}, "
            f"AUPRC {metric(bes, 'auprc')}, F1 {metric(bes, 'f1')})"
        )
    ]
    refit = common["methods"]["elm-dsm"]
    lines.append(
        f"- d3d_elm_time_to_event_dsm | 2026_09_06 | {triple(refit)} (elm-dsm refit; "
        "supplemental offline risk score reusing source data; "
        f"{common['n_shots']} shots / {common['bins']:,} common bins)"
    )
    reduced = common["methods"]["elm-dsm-detect"]
    bound = ""
    seeds = fx["dsm_baselines"]
    if seeds:
        row = seeds["reduced"]["auroc"]
        bound = (
            f"; {len(seeds['seeds'])} post-warm-up repeats give mean AUROC "
            f"{row['mean']:.3f} ({row['min']:.3f}–{row['max']:.3f})"
        )
    lines.append(
        f"- {DSM_LABEL} | {DATE} | {triple(reduced)} (secondary control; lower "
        f"bound on DSM detection skill under our recipe{bound}; "
        f"{common['n_shots']} shots / {common['bins']:,} common bins)"
    )
    for key, name, role in (
        ("elm-clock", "elm-clock", ""),
        ("elm-elmo", "elm-elmo", ""),
        ("always present", "always-present", "baseline; "),
    ):
        tag = "bes73" if key == "elm-elmo" else "all119"
        lines.append(
            f"- {name} | {DATE} | {triple(sets[tag]['methods'][key])} "
            f"({role}primary {tag}; {scope(tag)})"
        )
    control = records["feature"]["sets"]["primary"]["all119"]["methods"]
    lines.append(
        f"- elm-feature | {DATE} | {triple(control['elm-feature-only'])} "
        f"(control; primary all119; {scope('all119')})"
    )
    return lines


def caveat(records: dict, fx: dict) -> str:
    quiet = fx["non_crowd_only"]
    parts = [
        (
            "Brackets are 95% shot-bootstrap intervals. Review results are "
            "developmental shot-CV estimates of ELMy-period occupancy, not onsets "
            f"({pct(fx['crowd_share'])}% of positive bins lie in crowd spans); the "
            "review was seeded by the D-alpha clock, which favours D-alpha-input "
            "models over ELM-O. A frozen-model score on the blind split awaits "
            "review of those shots."
        ),
        (
            f"Primary benchmark: all119 ({fx['bins']['all119']:,} bins) and bes73 "
            f"({fx['bins']['bes73']:,} bins). On bes73, paired elm-ours minus ELM-O is "
            f"{F.delta_text(fx['paired_primary'])}; every interval includes zero, so "
            "the two are statistically indistinguishable (equivalence untested), while "
            "elm-ours extends BES-free coverage to all 119 shots. On the secondary "
            "common bins the same difference is "
            f"{F.delta_text(fx['paired_common_elmo'])}."
        ),
        (
            f"On non-crowd-only shots elm-ours precision is "
            f"{F.fmt(*quiet['precision'])}; its absent-span alarm rate is "
            f"{F.fmt(*fx['alarm']['all119/elm-ours']['span_alarm'])} against "
            f"{F.fmt(*fx['alarm']['all119/elm-clock']['span_alarm'])} for the clock."
        ),
        (
            "The elm-dsm detection rows are lower bounds on DSM detection skill under "
            "our recipe."
        ),
    ]
    tiled = fx["tiled"]
    if tiled:
        parts.append(
            "Serving runs a shot whole through the five fold models with the mean "
            f"fold threshold ({tiled['serving_threshold']:.3f}, unevaluated); "
            f"{tiled['tile_ms']:,} ms tiled inference and run-day-grouped folds "
            "are sensitivities in the protocol document."
        )
    parts.append(
        "Catalog physical-onset output is withheld. Every Smith method's "
        "precision/F1 is conditional on selected windows; continuous-discharge "
        "precision/F1 is unavailable. The experimental Smith onset head is omitted "
        "from catalog model claims. Inputs, run-day sharing and timing limits: "
        "[elm_ours.md](../../../docs/labeler/elm_ours.md)."
    )
    return " ".join(parts)


def patch_models(text: str, new_lines: list[str], note: str) -> str:
    """Update the entries in `all`; keep `stable`, unknown entries and their dates."""
    begin, end = text.index("## Models"), text.index("## Inputs")
    section = text[begin:end]
    stable = next(
        line for line in section.splitlines() if line.startswith("**stable**:")
    )
    aliases = {
        "elm_clock": "elm-clock",
        "elmo": "elm-elmo",
        "elm-dsm-detect": DSM_LABEL,
        "elm-dsm (detection)": DSM_LABEL,
    }
    entries = {}
    for line in re.findall(r"^- .*$", section, re.MULTILINE):
        name = line[2:].split(" | ")[0]
        name = aliases.get(name, name)
        entries[name] = "- " + name + line[2 + len(line[2:].split(" | ")[0]) :]
    for line in new_lines:
        name = line[2:].split(" | ")[0]
        if name in entries:
            date = re.search(r"\d{4}_\d{2}_\d{2}", entries[name])
            if date:
                line = re.sub(r"\d{4}_\d{2}_\d{2}", date.group(), line, count=1)
        entries[name] = line
    for scope_only in ("elm-ours-onset", "elm-ours (BES subset)"):
        entries.pop(scope_only, None)
    rendered = (
        "## Models\n"
        + stable
        + f"\n\n**latest**: elm-ours | {DATE}\n\n**all**:\n"
        + "\n".join(entries.values())
        + "\n\n"
        + "\n".join(wrap(note, WIDTH))
        + "\n\n"
    )
    return text[:begin] + rendered + text[end:]


def dsm_card(records: dict, fx: dict) -> Path:
    """Rewrite the card's evaluation section, with the native comparator inside it."""
    dsm, native = records["dsm"], records["native"]
    text = CARD.read_text()
    trailing = text.find(
        "\n### Native DSM detection comparator\n", text.index("\n## Contact")
    )
    if trailing != -1:  # the pre-round-eight layout appended it after `## Contact`
        text = text[:trailing].rstrip() + "\n"
    folds = dsm["detectors"]["elm-dsm-detect"]["folds"]
    context = dsm["model_context"]
    own = dsm["own_target"]
    epochs = ", ".join(str(r["best_epoch"]) for r in folds)
    thresholds = ", ".join(f"{r['threshold']:.3f}" for r in folds)
    doc = Doc()
    doc.heading(2, "Evaluation")
    doc.heading(3, "Reduced-input reviewed-label detection adaptation")
    doc.para(
        "These are developmental shot-CV occupancy estimates on five fixed folds. "
        "Preliminary outer-fold predictions were available before the reported "
        "recipe was fixed and could have informed inputs, scaling, architecture, "
        "selection or evaluation; the saved records do not establish their "
        "influence. Inner-validation AUPRC selects checkpoints at zero-based "
        f"epochs {epochs}; inner-validation F1 selects thresholds {thresholds}. "
        "Each fold fits its own normalization and starts from random weights. No "
        "blind-cohort shots enter these fits."
    )
    rows = []
    for tag, label in (("all119", "All reviewed"), ("bes73", "BES subset only")):
        panel = dsm["sets"][tag]
        res = panel["methods"]["elm-dsm-detect"]
        rows.append(
            [
                label,
                str(panel["n_shots"]),
                f"{panel['bins']:,}",
                metric(res, "auroc"),
                metric(res, "auprc"),
                metric(res, "f1"),
            ]
        )
    doc.table(
        [
            "Common panel",
            "Shots",
            "50 ms bins",
            "AUROC [95% shot CI]",
            "AUPRC",
            "F1",
        ],
        rows,
        right=(1, 2),
    )
    seeds = fx["dsm_baselines"]
    repeat = ""
    if seeds:
        row = seeds["reduced"]["auroc"]
        repeat = (
            f" With post-warm-up checkpoint selection and {len(seeds['seeds'])} "
            f"further seeds the all119 AUROC has mean {row['mean']:.3f} (range "
            f"{row['min']:.3f}–{row['max']:.3f}); the native comparator below is "
            "refitted the same way."
        )
    doc.para(
        f"Source: [dsm/evaluation.json]({BOUNDARY}/dsm/evaluation.json), "
        "`detectors.elm-dsm-detect` and `sets`. " + LOWER_BOUND + repeat
    )
    doc.para(
        "The detector uses 60 input columns. "
        f"{detection_inputs(records['input_audit'])} "
        "This is elm-dsm (60-input 1×128 refit, detection), a reduced-input "
        "adaptation trained and evaluated on 50 ms rows. The source model trained "
        "on native 1 ms rows with 124 inputs and layers [100, 1000] for WPQH "
        "breakthrough-ELM forecasting; this is not an objective-only retrain of "
        "that model. Fast-density units and filterscope sightlines are unverified "
        "in retained metadata; fixed input scaling, clipping and magnitude "
        "screening do not establish physical calibration."
    )
    doc.para(
        "The companion occupancy U-Net omits FS01 because its retained cache "
        "contains FS02–04 only. Its fast-density inputs divide native values by "
        "`1e14`, clip to `[-3, 12]`, and clip ten times the 0.2 s high-pass to "
        "`[-10, 10]`; chords with median absolute native magnitude above `1e16` "
        "are zeroed by a heuristic failed-digitiser screen. Offline metadata audits "
        "made no new fetches and changed no saved inputs or weights. Sources: "
        "`density_units.json`, `filterscope_metadata.json` and "
        "`src/labeler/elm/inputs.py`. No independently validated physical-onset "
        "detector is delivered, and run days cross folds in both developmental "
        f"analyses ({fx['review_run_days']['crossing']} of "
        f"{fx['review_run_days']['days']} review days; "
        f"{fx['smith_run_days']['crossing']} of {fx['smith_run_days']['days']} "
        "Smith days)."
    )
    doc.heading(3, "Limited-input survival refit (selection evidence)")
    doc.para(
        f"The served checkpoint is after {context['refit_checkpoint_epochs']} epoch "
        f"of a {context['refit_run_epochs']}-epoch run, selected at best_epoch="
        f"{context['refit_best_epoch']}. Its {own['shots']} physical "
        "source-validation shots selected the checkpoint; they are not independent "
        "test evidence."
    )
    doc.table(
        ["Horizon", "AUROC [95% physical-shot CI]"],
        [[f"{r['horizon_ms']:g} ms", risk_metric(r)] for r in own["horizons"].values()],
    )
    exact = native["reviewed_exact_export"]["horizons"]["h50ms"]
    doc.para("Source: `dsm/evaluation.json:own_target.horizons`.")
    doc.heading(3, "Native original-checkpoint audit")
    doc.para(
        "The original 124-input, 1 ms checkpoint has embedding layers [100, 1000]. "
        "The limited-input survival refit and reviewed detector instead use one "
        "128-unit layer. Native evaluation uses model9 parameter setting 1 with "
        "ReLU6; the Keras conversion's unbounded ReLU is not substituted."
    )
    doc.para(
        "The corrected forward presence target asks whether a reviewed present span "
        "intersects (t, t+h]; the separate onset target asks whether a non-crowd "
        "start lies in that interval. Reviewed non-crowd starts are annotation "
        "boundaries without independent physical-onset truth."
    )
    doc.para(
        f"Exact-export 50 ms AUROC is {exact['auroc']:.3f}, CI null: descriptive only "
        f"on {exact['n_shots']} shots reused in source fitting and {exact['rows']:,} "
        "rows (" + ", ".join(map(str, exact["shots"])) + "). 196541 entered optimizer "
        "fitting; the other three entered checkpoint selection; all entered source "
        "normalization. Five shots have exact exports, but 192751 has no scored "
        "overlap, so the exact-export JSON contains four shots. No operating "
        "threshold is selected."
    )
    rows = []
    for key, label in (
        ("reviewed_reconstructed", "Reconstructed presence"),
        ("own_target", "Source selection target"),
    ):
        for horizon, row in native[key]["horizons"].items():
            rows.append(
                [
                    label,
                    f"{horizon[1:-2]} ms",
                    str(row["n_shots"]),
                    f"{row['rows']:,}",
                    risk_metric(row),
                ]
            )
    doc.table(
        ["Native panel", "Horizon", "Shots", "Rows", "AUROC [95% physical-shot CI]"],
        rows,
        right=(2, 3),
    )
    doc.para(
        "Reconstruction is a sensitivity: source smoothing of concatenated phase "
        "rows differs from within-shot NBI smoothing. Source validation retains the "
        "original reversed chronological split and selected weights; it is not "
        "independent evaluation. Coverage, inputs, targets and memberships are in "
        f"[native_evaluation.json]({BOUNDARY}/dsm/native_evaluation.json). Reproduce "
        "scoring without training with `elm_dsm_evaluate.py --rescore` and "
        "`elm_dsm_native.py`; render this block with `elm_protocol.py`."
    )
    audit_blocks(doc, records, fx)
    begin = text.index("## Evaluation")
    end = text.index("### BES ablation", begin)
    CARD.write_text(text[:begin] + doc.text() + "\n" + text[end:])
    return CARD


def timing_text(fx: dict) -> str:
    off = fx["offsets"]
    return (
        f"Of {off['starts']} non-crowd reviewed starts on {off['shots']} BES-subset "
        f"shots, {off['matched']} (on {off['matched_shots']} shots) match the "
        "nearest ELM-O onset within ±50 ms; the other "
        f"{off['unmatched']} are omitted and each start is matched independently. "
        "Reviewed start minus ELM-O onset has median "
        f"{off['median']} ms and quartiles [{off['p25']}, {off['p75']}] ms on the "
        "matched subset. This annotation-to-detector offset is not a measured "
        "physical-onset error: neither reference has independently verified "
        "onset truth (`review_start_offsets.json`, "
        "`elm_review_start_offsets.py`)."
    )


def glossary(doc: Doc, fx: dict) -> None:
    common = fx["common_bins"]
    doc.heading(2, "Glossary")
    doc.bullets(
        [
            (
                f"**all119**: the {fx['shots']['all119']} reviewed shots, scored on "
                f"{fx['bins']['all119']:,} interior 50 ms bins that lie wholly inside "
                "one reviewed absent, non-crowd or crowd span and inside signal "
                "coverage."
            ),
            (
                f"**bes73**: the {fx['shots']['bes73']} of them with BES (ELM-O's "
                f"coverage), {fx['bins']['bes73']:,} bins. It is a scope, not a model."
            ),
            (
                f"**common bins**: primary bins that also have valid DSM rows "
                f"({common['all119']:,} / {common['bes73']:,}); the secondary control."
            ),
            (
                "**crowd / non-crowd / absent**: reviewed ELMing-period spans, other "
                "present spans, and spans without ELMs."
            ),
            (
                "**known all-covered**: legacy-swap 50 ms cells with at least 25 ms of "
                "reviewed present occupancy inside every method's coverage; unknown "
                "time stays unknown."
            ),
            "**exact exports**: the source DSM's saved native rows.",
            (
                "**source stats / source weights**: DSM variants that reuse upstream "
                "normalization or weights (marked ‡); that normalization includes two "
                "blind-cohort shots."
            ),
            "**†**: recall of at least 0.99.",
        ]
    )


def target_section(doc: Doc, fx: dict) -> None:
    share = fx["clock_share"]
    quiet = fx["non_crowd_only"]
    alarm = fx["alarm"]
    doc.heading(2, "Target and labels")
    doc.para(
        f"The target is ELMy-period occupancy: {pct(fx['crowd_share'])}% of "
        "positive bins lie in crowd spans, and the label set has no onset trace. "
        "Scored bins lie wholly inside reviewed spans, so boundary-straddling bins "
        "are dropped and the task is easier than whole-shot detection. Review "
        f"started from the D-alpha clock: {pct(share['start'])}% of crowd starts, "
        f"{pct(share['end'])}% of ends and {pct(share['both'])}% of both "
        f"({share['crowd_spans']} crowd spans) lie within 1 ms of its boundaries, "
        "which favours D-alpha-input models over ELM-O (BES and interferometers)."
    )
    doc.para(timing_text(fx))
    doc.para(
        f"On non-crowd-only shots ({quiet['shots']} shots, {quiet['bins']:,} bins) "
        f"elm-ours precision is {F.fmt(*quiet['precision'])}. The absent-span "
        "alarm rate (any detection touching a covered absent span) is "
        f"{F.fmt(*alarm['all119/elm-ours']['span_alarm'])} for elm-ours and "
        f"{F.fmt(*alarm['all119/elm-clock']['span_alarm'])} for the clock on all119, "
        f"{F.fmt(*alarm['bes73/elm-elmo']['span_alarm'])} for ELM-O on bes73."
    )


def benchmark_section(doc: Doc, records: dict, fx: dict) -> None:
    doc.heading(2, "Benchmark")
    doc.para(
        "All review results are **developmental shot-CV estimates**: predictions on "
        "outer-fold shots existed before the recipe was fixed and could have "
        "informed input choice, scaling, architecture, selection or the protocol; "
        "the saved records do not establish whether they did. Five shot-grouped "
        "folds cover the 119 reviewed shots; no blind-test shot enters fits, "
        "normalization or threshold selection, and a frozen-model score on the "
        "blind split awaits review of those shots. Each fold selects its "
        "checkpoint by inner-validation AUPRC and its threshold by inner-validation "
        f"F1. Intervals are {fx['replicates']:,}-draw shot bootstraps needing five "
        "denominator-bearing shots per endpoint."
    )
    folds = fx["ours_folds"]
    early = min(folds, key=lambda r: r["epoch"])
    thr = [r["threshold"] for r in folds]
    seeds = fx["seed_range"]["all119"]
    doc.para(
        "elm-ours calls a bin present when its mean probability reaches the fold "
        "threshold; ELM-O and the clock use any touching detected span. Fold "
        f"{early['fold']} (zero-based) selected epoch {early['epoch']} of "
        f"{fx['epochs']}; fold thresholds span {min(thr):.3f}–{max(thr):.3f}. Four "
        f"training seeds give all119 AUROC {seeds['auroc']['min']:.3f}–"
        f"{seeds['auroc']['max']:.3f}, AUPRC {seeds['auprc']['min']:.3f}–"
        f"{seeds['auprc']['max']:.3f} and F1 {seeds['f1']['min']:.3f}–"
        f"{seeds['f1']['max']:.3f}. On bes73, paired elm-ours minus ELM-O is "
        f"{F.delta_text(fx['paired_primary'])}: every interval includes zero "
        "(equivalence untested), while elm-ours extends BES-free coverage to all "
        "119 shots."
    )
    primary_table(doc, records)
    moved = fx["moved"]
    if moved:
        doc.heading(3, "Spans whose reviewed boundaries moved from the clock")
        paired = moved["paired"]["elm-ours - elm-elmo: auroc"]
        doc.para(
            f"{moved['present_spans']} of {moved['all_present_spans']} bes73 present "
            "spans have no boundary within 1 ms of the clock's (an absent span is "
            f"the gap between clock spans): {moved['bins']:,} bins on "
            f"{moved['shots']} shots, {moved['positive_bins']:,} positive, with "
            f"{moved['absent_spans']} absent spans. The intervals are wide and the "
            "stratum inconclusive; paired elm-ours minus ELM-O AUROC is "
            f"{F.fmt(*F.paired(paired), signed=True)}. Source: "
            "`ours/moved_boundary_stratum.json`."
        )
        moved_table(doc, fx)
    dsm = records["dsm"]["sets"]["all119"]["methods"]["elm-dsm-detect"]
    pairs = fx["paired_common_dsm"]
    doc.para(
        "**Secondary DSM common-bin control** "
        f"({fx['common_bins']['all119']:,} / {fx['common_bins']['bes73']:,} bins; "
        "tables in `dsm/evaluation.json:sets` and the paper). The reduced-input "
        f"elm-dsm detector scores all119 AUROC {metric(dsm, 'auroc')}, AUPRC "
        f"{plain(dsm, 'auprc')}, F1 {plain(dsm, 'f1')}; its reported fit is fragile "
        "(selected epochs "
        + ", ".join(str(r["epoch"]) for r in fx["dsm_folds"])
        + "). Paired elm-ours minus elm-dsm, all119: "
        f"{F.delta_text(pairs['all119'])}; bes73: {F.delta_text(pairs['bes73'])}. "
        "Paired elm-ours minus ELM-O on bes73: "
        f"{F.delta_text(fx['paired_common_elmo'])}. " + LOWER_BOUND
    )


def serving_section(doc: Doc, fx: dict) -> None:
    tiled, run_day = fx["tiled"], fx["run_day"]
    doc.heading(2, "Serving protocol and sensitivities")
    if tiled:
        thresholds = ", ".join(f"{t:.3f}" for t in tiled["fold_thresholds"])
        doc.para(
            "**Serving.** A new shot runs whole (zero-padded to the network's "
            "length multiple) through each of the five fold models; the score is "
            "their mean probability and the threshold is the mean of the five fold "
            f"thresholds ({thresholds}; mean {tiled['serving_threshold']:.3f}). The "
            "reported numbers use the one fold model that never saw each shot, so "
            "the ensemble and this threshold are unevaluated: every reviewed shot "
            "is in some fold's training set and blind-test shots may not be used."
        )
        doc.para(
            "**Tiling.** Group normalisation takes its statistics over the whole "
            f"input and training used {tiled['tile_ms']:,} ms crops. Scoring the "
            f"same frozen models and thresholds on independent {tiled['tile_ms']:,} ms "
            f"tiles gives all119 AUROC {tiled['tiled_auroc']:.3f} against "
            f"{tiled['whole_auroc']:.3f} whole (whole minus tiled "
            f"{F.fmt(*tiled['whole_minus_tiled'], signed=True)}). Nothing is "
            "refitted. Source: `ours/tiled_inference.json`."
        )
    if run_day:
        mix = ", ".join(str(r["shots_with_non_crowd_spans"]) for r in run_day["folds"])
        parts = []
        for tag in ("all119", "bes73"):
            res = run_day[tag]
            head = res["headline"]
            parts.append(
                f"{tag} AUROC {F.fmt(*res['auroc'])} (headline {head['auroc']:.3f}), "
                f"AUPRC {res['auprc'][0]:.3f} ({head['auprc']:.3f}), "
                f"F1 {res['f1'][0]:.3f} ({head['f1']:.3f})"
            )
        doc.para(
            "**Run-day folds.** "
            f"{fx['review_run_days']['crossing']} of {fx['review_run_days']['days']} "
            "run days cross the headline's shot-grouped folds. Repeating the CV with "
            f"every run day whole inside one fold ({run_day['days']} days, same "
            "recipe, folds not tuned on performance) gives "
            + "; ".join(parts)
            + ". The greedy day-balanced dealing balances shot counts, not "
            "annotation kinds (shots with non-crowd spans per fold: "
            + mix
            + "), so this is a sensitivity beside the headline. Source: "
            "`ours/run_day_cv.json`."
        )


def smith_swap_sections(doc: Doc, records: dict, fx: dict) -> None:
    head = fx["smith_head"]
    overlap = records["swap"]["swap"]["overlap"]
    doc.heading(2, "Smith onset and transfer")
    doc.para(
        "Frozen occupancy transfer is omitted for target mismatch: 50 ms occupancy "
        "cannot resolve approximately 8.5 ms Smith windows. The frozen "
        "reviewed-span start output is also omitted. The experimental "
        "elm-ours-onset head is developmental shot CV with within-Smith run-day "
        f"sharing ({fx['smith_run_days']['crossing']} of "
        f"{fx['smith_run_days']['days']} run days cross folds); overlap with ELM-O's "
        "historical tuning events is unknown."
    )
    doc.para(
        f"At ±2 ms the head recalls {head['recall']:.3f} of {head['windows']:,} "
        f"hand-labelled windows on {head['shots']} shots; every matched error is "
        f"within {head['max_error_ms']:.2f} ms and {pct(head['correct_cell'])}% lie "
        "in the correct 1 ms cell. **Every method's precision/F1 is conditional on "
        "selected windows; continuous-discharge precision/F1 is unavailable.** "
        f"{head['out_of_window']:,} further firings lie outside the windows and "
        "lack negative truth; the 10 ms minimum peak separation fixes the head's "
        "in-window precision, so none is reported. ELM-O region-overlap recall is "
        f"{head['elmo_region_recall']:.3f} after the 179859 time-axis repair. "
        "Traces: `$LABELER_ROOT/round4/elm/smith/cv/pred/`; catalog onsets stay "
        "withheld."
    )
    doc.heading(2, "Reference swap")
    doc.para(
        swap_paragraph(records["swap"], records["ours"])
        + " Fixed predictions are also compared on "
        f"{overlap['reviewed']['bins']} strict interior bins; only AUROC compares "
        "references, and covered-gap merges at 100/200/300 ms never bridge missing "
        "coverage."
    )


def swap_paragraph(record: dict, ours: dict) -> str:
    """The swap result in prose; every figure is read from the swap record."""
    overlap, bes = record["swap"]["overlap"], record["swap"]["overlap_bes"]
    held = record["swap"]["overlap_dsm_heldout"]
    held_bes = record["swap"]["overlap_bes_dsm_heldout"]
    audit = record["interval_audit"]["known_review_majority"]
    change = overlap["comparison"]["auroc_change_paired"][swap_tex.NAME["dsm"]]
    inside = change["ci95"][0] <= 0 <= change["ci95"][1]
    elmo = swap_tex.NAME["elmo"]
    panel = bes["reviewed"]["methods"][elmo]["point"]["auroc"]
    whole = ours["sets"]["bes73"]["methods"][elmo]["point"]["auroc"]
    count = swap_tex.count_word
    return (
        f"The legacy onset table overlaps {count(overlap['n_shots'])} review shots "
        f"({count(bes['n_shots'])} with BES). On {audit['bins']} known all-covered "
        f"50 ms cells, M={audit['M']} and P={audit['P']}; these are definition "
        "disagreements, not adjudicated events. "
        + swap_tex.ranking_sentence(overlap, bes)
        + "The supplemental refit's paired legacy-minus-review AUROC change is "
        f"{swap_tex.signed(change['value'], change['ci95'])}, in-sample on "
        f"{len(overlap['dsm_refit_training_shots'])}/{overlap['n_shots']} shots and "
        f"{'within' if inside else 'outside'} its interval; no AE Finding-2 analogue "
        f"is established. The {count(held['n_shots'])} shots ({held_bes['n_shots']} "
        "with BES) outside original DSM fitting are too small for intervals; their "
        "values are in the JSON. All overlap shots use the upstream WPQH PCPHD02/03 "
        "export in the reduced-input detection adaptation. ELM-O scores "
        f"{panel:.3f} on these {count(bes['n_shots'])} WPQH shots against "
        f"{whole:.3f} on the BES subset, likely domain shift."
    )


def provenance_section(doc: Doc, fx: dict) -> None:
    doc.heading(2, "Signal provenance and numerical preprocessing")
    doc.para(
        "The retained caches do not establish fast-density physical ordinate units "
        "or FS01–04 sightlines (divertor versus midplane); FS01 is not an input to "
        "the occupancy U-Net, and paired slow CO2 checks numerical scales only. "
        "Filterscope levels use `(log10(max(x, 1e12)) - 15) / 1.5` with contrast to "
        "a 0.5 s running median. Fast density is divided by `1e14`, clipped to "
        "`[-3, 12]`, and ten times its 0.2 s high-pass is clipped to `[-10, 10]`; a "
        "chord whose median absolute native magnitude exceeds `1e16` is zeroed. "
        "These are fixed numerical choices, not verified calibration or a validated "
        "failure criterion. Offline audits changed no saved inputs, screening or "
        "weights and made no new fetches. Sources: `density_units.json`, "
        "`filterscope_metadata.json`, `src/labeler/elm/inputs.py`."
    )


def sources_section(doc: Doc, records: dict) -> None:
    doc.heading(2, "Sources and reproduction")
    doc.para("Records under `outputs/labeler/elm/`:")
    doc.bullets(
        [
            (
                "`ours/evaluation.json`, `ours/seed_repeats.json`, "
                "`ours/feature_only.json`, `ours/annotation_strata.json`: occupancy "
                "results, seeds, control and labels."
            ),
            (
                "`ours/moved_boundary_stratum.json`, `ours/tiled_inference.json`, "
                "`ours/run_day_cv.json`, `ours/rejection_sensitivity.json`: stratum "
                "and sensitivities."
            ),
            (
                "`dsm/evaluation.json`, `dsm/native_evaluation.json`, "
                "`dsm/native_detection.json`, `dsm/baseline_seeds.json`: common bins, "
                "refits, native targets and comparator, repeats."
            ),
            (
                "`smith/evaluation.json`, `swap/evaluation.json`, "
                "`review_start_offsets.json`: onset scope, swap and timing."
            ),
            (
                "`training_history.json`, `density_units.json`, "
                "`filterscope_metadata.json`: folds and signal provenance."
            ),
        ]
    )
    doc.para(provenance_note(records))
    doc.para(
        "Run through the pixi labelmaker wrapper with `LABELER_NO_FETCH=1`: read "
        "the frozen run from `ours/evaluation.json:run`, then `elm_ours_evaluate.py "
        "--run <run>`, `elm_dsm_evaluate.py --run <run> --rescore`, "
        "`elm_feature_evaluate.py`, `elm_moved_boundary_stratum.py`, "
        "`elm_tiled_inference.py`, `elm_run_day_summary.py` (after "
        "`elm_ours_evaluate.py --run cv2_runday`), `elm_dsm_baseline_seeds.py`, "
        "`elm_smith_evaluate.py evaluate`, `elm_reference_swap.py`, "
        "`elm_example_figure.py --run <run>`, `elm_paper_tables.py` and "
        "`elm_protocol.py`. Figures, checkpoints and predictions live under "
        "`$LABELER_ROOT/round4/elm/`; the original ELM-O benchmark doc is retained "
        "with a dated update."
    )


def render_doc(records: dict, fx: dict) -> str:
    ours = records["ours"]
    own = ours["sets"]["all119"]["methods"]["elm-ours"]
    bes = ours["sets"]["bes73"]["methods"]["elm-ours"]
    doc = Doc()
    doc.heading(1, "ELM occupancy and onset evaluation")
    doc.para(
        "`elm-ours` delivers BES-free ELMy-occupancy probability; the requested "
        "finer physical-onset trace is **not delivered to the catalog**. On all119 "
        f"({fx['bins']['all119']:,} interior 50 ms bins) its AUROC is "
        f"{metric(own, 'auroc')}, AUPRC {metric(own, 'auprc')} and F1 "
        f"{metric(own, 'f1')}; on bes73 ({fx['bins']['bes73']:,} bins) AUROC "
        f"{metric(bes, 'auroc')} and F1 {metric(bes, 'f1')}. DSM common-bin results "
        "are secondary controls; native detection is scored on a smaller "
        "identical-support panel. Smith onset recall describes selected windows "
        "only, and the legacy swap measures reference-definition disagreement; "
        "neither confirms continuous-discharge physical-onset performance."
    )
    glossary(doc, fx)
    target_section(doc, fx)
    benchmark_section(doc, records, fx)
    serving_section(doc, fx)
    doc.heading(2, "DSM baselines")
    doc.para(
        "The detection DSM is a reduced-input adaptation trained and evaluated on "
        "50 ms rows: 60 input columns and one 128-unit layer. "
        + detection_inputs(records["input_audit"])
        + " The historical source DSM trained on native 1 ms rows with 124 inputs "
        "and layers [100, 1000] for WPQH breakthrough-ELM forecasting, so the "
        "detection row is not an objective-only retrain of that model and "
        "comparative claims apply to this adaptation; the native detection refit "
        "is compared below. Historical survival and detection variants that reuse "
        "source weights or statistics remain supplemental. " + LOWER_BOUND
    )
    audit_blocks(doc, records, fx, full=False)
    rejection_block(doc, records)
    provenance_section(doc, fx)
    smith_swap_sections(doc, records, fx)
    sources_section(doc, records)
    return doc.text()


def main() -> None:
    records = load_records()
    fx = F.facts(records)
    DOC.write_text(render_doc(records, fx))
    README.write_text(
        patch_models(README.read_text(), model_lines(records, fx), caveat(records, fx))
    )
    card = dsm_card(records, fx)
    manifest = {
        "git": git_sha(full=True),
        "script_sha256": sha256_of(Path(__file__)),
        "sources": {
            key: {"path": str(path.relative_to(OUTPUTS)), "sha256": sha256_of(path)}
            for key, path in F.paths(OUTPUTS).items()
        },
        "artifacts": {
            str(p.relative_to(REPO)): sha256_of(p) for p in (DOC, README, card)
        },
        "model_names": {
            key: swap_tex.method_label(key)
            for key in (
                "elm-ours",
                "elm-dsm-detect",
                "elm-clock",
                "always present",
                "elm-feature-only",
                "elm-elmo",
            )
        },
    }
    (OUTPUTS / "protocol.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print("protocol and DSM card written; ELM-O historical doc preserved")


if __name__ == "__main__":
    main()
