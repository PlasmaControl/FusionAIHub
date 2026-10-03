#!/usr/bin/env python
"""Write paper ELM benchmark and swap tables from the committed evaluation JSONs.

Run after the three evaluations. Outputs default to $LABELER_ROOT/round4/elm/.
Each panel names its shot/bin set; no method is compared across different bins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from labeler.config import Paths, git_sha
from labeler.elm import compare, swap_tex

REPO = Path(__file__).resolve().parents[2]
OUTPUTS = REPO / "outputs" / "labeler" / "elm"


def benchmark_table(ours, dsm) -> str:
    names = dict(swap_tex.ROWS)
    names["always present"] = "Always-present rule"
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & AUROC & AUPRC & F1 \\",
    ]
    descriptions = []
    for source, record in (("Primary reviewed bins", ours), ("Common DSM bins", dsm)):
        for tag in ("all119", "bes73"):
            res = record["sets"][tag]
            subset = (
                "BES subset, ELM-O chunks" if tag == "bes73" else "All reviewed shots"
            )
            description = (
                f"{source}, {subset}: {res['n_shots']} shots/{res['bins']:,} bins"
            )
            descriptions.append(description)
            lines += [r"\midrule", r"\multicolumn{4}{l}{" + description + r"} \\"]
            for key in ("ours", "elmo", "clock", "dsm", "detect", "init", "always"):
                method = compare.NAME[key] if key != "always" else "always present"
                if method not in res["methods"]:
                    continue
                result = res["methods"][method]
                cells = [
                    swap_tex.metric_cell(result, m) for m in ("auroc", "auprc", "f1")
                ]
                lines.append(names[method] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    caption = (
        "ELM interval-occupancy benchmark. " + "; ".join(descriptions) + ". "
        "All cells pool identical 50 ms bins within a panel: bins are wholly inside "
        "one reviewed absent, non-crowd present or crowd span and analysed time. "
        "Primary panels use fetched-signal coverage or ELM-O chunks; common panels "
        "also require DSM offline-risk and detection rows. elm-ours and DSM detection "
        "heads use shot-grouped out-of-fold predictions; the initialized variant "
        "retains a pretrained embedding with reviewed and cohort-test shot overlap. "
        "Operating thresholds "
        "come only from each fold's inner-validation shots. ELM-O uses eta 0.997 "
        "for hard calls; its rank metrics come from the saved nested eta sweep "
        "on these same bins. Clock and always-present rows are rules; the clock "
        "seeded the review and is not independent. Brackets are 95\\% percentile "
        "intervals from 1,000 shared shot-bootstrap resamples. elm-dsm refit uses "
        "60 of the original 124 inputs and the legacy onset table (Hiro Farre "
        "Josep Kaga annotations, compiled by labels\\_format.py/source\\_formatters) "
        "source with reviewed and cohort-test shot overlap, unavailable diagnostics "
        "mean-filled and "
        "inputs clipped at "
        "$|z|=10$. Every DSM variant has no D-alpha input (pcphd02/03 mean-filled), "
        "50 ms-mean serving of a 1 ms-trained model, and CO2 missing on 75/119 shots. "
        "The 1 ms training refers to the source survival refit; detection heads "
        "are refitted on reviewed 50 ms-mean rows. "
        "The refit is an offline risk score with 25 ms centered-NBI lookahead "
        "(not a causal forecast). Every DSM variant uses upstream normalization "
        "constants computed before the upstream split, including blind-cohort "
        "source shots 190646 and 190532 (feature-statistics exposure). "
        "The original full-input model is not evaluated here. A dagger marks "
        "recall $\\geq0.99$; numeric F1 and intervals are retained. "
    )
    flagged = []
    for source, record in (("primary", ours), ("common", dsm)):
        for tag, subset in record["sets"].items():
            for method, result in subset["methods"].items():
                if result["point"]["recall"] >= 0.99:
                    p, r = result["point"]["precision"], result["point"]["recall"]
                    flagged.append(
                        f"{source} {tag} {names[method]}: precision {p:.3f}, "
                        f"recall {r:.3f}"
                    )
    caption += "Dagger cells: " + "; ".join(flagged) + "."
    return swap_tex.wrap_table("\n".join(lines), caption, "tab:elm-benchmark")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args(argv)
    out = args.out_dir or Paths.from_env().root / "round4" / "elm"
    out.mkdir(parents=True, exist_ok=True)
    files = {key: OUTPUTS / key / "evaluation.json" for key in ("ours", "dsm", "swap")}
    records = {key: json.loads(path.read_text()) for key, path in files.items()}
    (out / "table_elm_benchmark.tex").write_text(
        benchmark_table(records["ours"], records["dsm"])
    )
    (OUTPUTS / "table_elm_benchmark.tex").write_bytes(
        (out / "table_elm_benchmark.tex").read_bytes()
    )
    swap_tex.write(records["swap"], out)
    swap_output = OUTPUTS / "swap"
    swap_output.mkdir(parents=True, exist_ok=True)
    for path in out.glob("table_elm_*.tex"):
        if path.name != "table_elm_benchmark.tex":
            (swap_output / path.name).write_bytes(path.read_bytes())
    manifest = {
        "git": git_sha(),
        "sources": {
            key: {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for key, path in files.items()
        },
        "tables": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(out.glob("table_elm_*.tex"))
        },
        "benchmark_json_paths": [
            f"{key}/evaluation.json:sets.{tag}.methods"
            for key in ("ours", "dsm")
            for tag in ("all119", "bes73")
        ],
        "swap_json_paths": [
            "swap/evaluation.json:swap.overlap",
            "swap/evaluation.json:swap.overlap_bes",
            "swap/evaluation.json:interval_audit",
        ],
    }
    (out / "tables.json").write_text(json.dumps(manifest, indent=1))
    (OUTPUTS / "tables.json").write_text(json.dumps(manifest, indent=1))
    print("tables", out, sorted(manifest["tables"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
