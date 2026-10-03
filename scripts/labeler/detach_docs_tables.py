#!/usr/bin/env python
"""Print the result tables of `docs/labeler/detachment.md` from the result JSONs.

    python scripts/labeler/detach_docs_tables.py [--out tables.md]

Nothing is computed here: every number is read from a JSON a script wrote
(`data/events/detachment/extend_detach_vote/records/*.json`, and
`docs/labeler/results/detachment_*.json`), so the page can be refreshed after a rerun
without copying numbers by hand. Missing files are skipped.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RECORDS = REPO / "data" / "events" / "detachment" / "extend_detach_vote" / "records"
RESULTS = REPO / "docs" / "labeler" / "results"
NAMES = {"afrac": "Afrac", "prad": "Prad,div", "tangtv": "TangTV"}
STATES = ("attached", "detached", "marfe")


def load(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def ci(entry: dict, digits: int = 2) -> str:
    """`value [lo, hi]` of a `{value, ci95}` entry."""
    if entry is None:
        return "-"
    value, (lo, hi) = entry["value"], entry["ci95"]
    if math.isnan(value):
        return "-"
    return f"{value:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def coverage_tables() -> str:
    cov = load(RECORDS / "coverage.json")
    if cov is None:
        return ""
    rows = []
    for key, name in NAMES.items():
        c = cov[key]
        v = c["votes"]
        rows.append(
            [
                name,
                c["valid_bins"],
                c["shots_with_valid_bins"],
                v["attached"],
                v["detached"],
                v["marfe"],
            ]
        )
    out = table(
        ["indicator", "valid bins", "shots", "attached", "detached", "marfe"], rows
    )
    reasons = []
    for key, name in NAMES.items():
        why = cov[key]["invalid_reason_bins"]
        text = ", ".join(
            f"{k} {v}" for k, v in sorted(why.items(), key=lambda x: -x[1])
        )
        reasons.append([name, text])
    out += "\nInvalid bins by reason:\n\n" + table(["indicator", "reasons"], reasons)
    return (
        f"{cov['n_bins']} bins of {cov['n_shots_with_bins']} shots with bins; "
        f"{cov['n_eligible_shots']} eligible shots "
        f"({', '.join(f'{k} {v}' for k, v in cov['eligible_by_split'].items())}).\n\n"
        + out
    )


def agreement_table() -> str:
    agree = load(RECORDS / "agreement.json")
    if agree is None:
        return ""
    rows = []
    for pair, e in agree["all_eligible_bins"].items():
        a, b = pair.split("__")
        rows.append(
            [
                f"{NAMES[a]} / {NAMES[b]}",
                e["both_valid_bins"],
                e["both_vote_bins"],
                "-" if e["agreement"] is None else f"{e['agreement']:.2f}",
                "-" if e["kappa"] is None else f"{e['kappa']:.2f}",
            ]
        )
    return table(["pair", "both valid", "both vote", "agreement", "kappa"], rows)


def model_tables() -> str:
    rec = load(RECORDS / "label_model.json")
    if rec is None:
        return ""
    fit = rec["fit"]
    rows = []
    for key, name in NAMES.items():
        a = rec["labelling_functions"][key]
        plain = fit["all_bins_fit_for_comparison"][key]
        rows.append(
            [
                name,
                f"{a['implied_accuracy']:.2f}",
                f"{a['acc_weight']:.2f}",
                f"{plain['implied_accuracy']:.2f}",
            ]
        )
    out = (
        f"Fit: {fit['n_shots']} shots, {fit['n_bins']} bins "
        f"(test split excluded); accuracies learned on {fit['anchor_bins']} bins of "
        f"{fit['anchor_shots']} shots where all three indicators are valid.\n\n"
        + table(
            [
                "indicator",
                "implied accuracy (anchored fit, used)",
                "weight",
                "accuracy if fitted on all bins",
            ],
            rows,
        )
    )
    cand = rec["structure_selection"]["candidates"]
    rows = [[k, f"{v['mean_heldout_loglik_per_bin']:.3f}"] for k, v in cand.items()]
    out += (
        f"\nStructure (held-out log-likelihood per bin, shot-grouped "
        f"{rec['structure_selection']['cv_folds']}-fold CV); chosen: "
        f"**{rec['chosen_structure']}**.\n\n"
        + table(["structure", "held-out log-lik / bin"], rows)
    )
    counts = rec["state_counts_bins"]
    rows = [
        [k, *[counts[k][s] for s in (*STATES, "uncertain")]]
        for k in ("label_model", "rule")
    ]
    out += "\nStates of the assessed bins:\n\n" + table(
        ["labeler", *STATES, "uncertain"], rows
    )
    m = rec["rule_vs_label_model"]
    rows = []
    for row_state, cols in m["rows_rule_columns_label_model"].items():
        rows.append([row_state, *[cols[s] for s in (*STATES, "uncertain")]])
    out += (
        f"\nRule (rows) against label model (columns), {m['bins']} bins: agreement "
        f"{m['agreement']:.2f}, kappa {m['kappa']:.2f}.\n\n"
        + table(["rule \\ label model", *STATES, "uncertain"], rows)
    )
    return out


def benchmark_tables() -> str:
    bench = load(RESULTS / "detachment_benchmark.json")
    if bench is None:
        return ""
    out = ""
    for ref, title in (
        ("loo", "against the label with that indicator's vote withheld"),
        ("combined", "against the combined label (circular, an upper bound)"),
    ):
        for subset in ("all", "test_shots"):
            rows = []
            for key, name in NAMES.items():
                e = bench["indicators"][key].get(ref, {}).get(subset)
                if e is None:
                    continue
                rows.append(
                    [
                        name,
                        f"{e['n_bins']['value']:.0f}",
                        ci(e["agreement"]),
                        ci(e["kappa"]),
                        ci(e["f1_attached"]),
                        ci(e["f1_detached"]),
                        ci(e["f1_marfe"]) if key == "tangtv" else "n/a",
                        ci(e["auroc_attached_vs_not"]),
                    ]
                )
            if rows:
                out += f"\n{title}, {subset.replace('_', ' ')}:\n\n" + table(
                    [
                        "indicator",
                        "bins",
                        "agreement",
                        "kappa",
                        "F1 attached",
                        "F1 detached",
                        "F1 marfe",
                        "AUROC att / not",
                    ],
                    rows,
                )
    rows = []
    for source in ("inversion", "surrogate"):
        for ref in ("loo", "combined"):
            e = bench["indicators"]["tangtv"].get(ref, {}).get(f"source_{source}")
            if e:
                rows.append(
                    [
                        source,
                        ref,
                        f"{e['n_bins']['value']:.0f}",
                        ci(e["agreement"]),
                        ci(e["kappa"]),
                    ]
                )
    if rows:
        out += "\nTangTV by the source of its front height:\n\n" + table(
            ["source", "reference", "bins", "agreement", "kappa"], rows
        )
    te = bench.get("divertor_te_check")
    if te:
        rows = [
            [
                state,
                v["n_bins"],
                "{:.1f} / {:.1f} / {:.1f}".format(*v["te_quartiles_ev"]),
                f"{v['share_below_threshold']:.2f}",
            ]
            for state, v in te["by_state"].items()
        ]
        out += (
            f"\nDivertor Thomson Te (eV; highest real-time point per bin), "
            f"threshold {te['threshold_ev']:.0f} eV:\n\n"
            + table(
                ["label state", "bins with Te", "Te quartiles", "share < 5 eV"], rows
            )
        )
        a = te["auroc_detached_vs_attached"]
        out += (
            f"\nAUROC of -Te for detached against attached bins: {ci(a)} "
            f"({a['n_bins']} bins, {a['n_shots']} shots).\n"
        )
        rows = [
            [NAMES[k], ci(v), v["n_bins"], v["n_shots"]]
            for k, v in te["indicator_auroc_cold_plate"].items()
        ]
        out += (
            "\nEach indicator's value against a cold plate (Te < 5 eV), no label:\n\n"
        )
        out += table(["indicator", "AUROC", "bins", "shots"], rows)
    return out


def sensitivity_table() -> str:
    sens = load(RESULTS / "detachment_bin_sensitivity.json")
    if sens is None:
        return ""
    rows = []
    for width in ("20ms", "50ms", "100ms"):
        e = sens[width]
        vs = e.get("vs_50ms")
        rows.append(
            [
                width,
                e["n_bins"],
                f"{e['assessed_share']:.2f}",
                f"{e['uncertain_share_of_assessed']:.2f}",
                f"{e['flicker_per_s']:.2f}",
                "-" if vs is None else f"{vs['agreement']:.3f}",
                "-" if vs is None else f"{vs['kappa']:.3f}",
            ]
        )
    return table(
        [
            "bin",
            "bins",
            "assessed",
            "uncertain / assessed",
            "flicker / s",
            "agreement with 50 ms",
            "kappa with 50 ms",
        ],
        rows,
    )


def learned_table() -> str:
    rows = []
    for name, key in (("detach-ours", "ours"), ("detach-victor", "victor")):
        r = load(RESULTS / f"detachment_{key}.json")
        if r is None:
            continue
        n = r.get("n_windows", r.get("n_frames"))
        cv, maj = r["cv_shots"], r["cv_majority"]
        rows.append(
            [
                name,
                "CV, by shot",
                f"{n} / {r['n_shots']}",
                ci(cv["accuracy"]),
                ci(cv["kappa"]),
                ci(cv["macro_f1"]),
                f"{maj['accuracy']:.2f}",
            ]
        )
        if "test_shots" in r:
            t, tm = r["test_shots"], r["test_majority"]
            rows.append(
                [
                    name,
                    "test split",
                    f"- / {r['n_test_shots']}",
                    ci(t["accuracy"]),
                    ci(t["kappa"]),
                    ci(t["macro_f1"]),
                    f"{tm['accuracy']:.2f}",
                ]
            )
        if "cv_no_tangtv_vote" in r:
            f = r["cv_no_tangtv_vote"]
            rows.append(
                [
                    name,
                    "CV, TangTV not voting",
                    f"{f['n_frames']} / {f['n_shots']}",
                    ci(f["accuracy"]),
                    ci(f["kappa"]),
                    ci(f["macro_f1"]),
                    f"{f['majority']['accuracy']:.2f}",
                ]
            )
    return table(
        [
            "model",
            "scored on",
            "windows or frames / shots",
            "accuracy",
            "kappa",
            "macro F1",
            "majority accuracy",
        ],
        rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    parts = {
        "COVERAGE": coverage_tables(),
        "INDICATOR AGREEMENT": agreement_table(),
        "LABEL MODEL": model_tables(),
        "BENCHMARK": benchmark_tables(),
        "BIN WIDTH": sensitivity_table(),
        "LEARNED BASELINES": learned_table(),
    }
    text = "".join(f"<!-- {k} -->\n{v}\n" for k, v in parts.items() if v)
    if args.out:
        Path(args.out).write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
