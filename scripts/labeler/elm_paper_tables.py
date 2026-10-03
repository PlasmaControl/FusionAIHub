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
            description = f"{source}: {res['n_shots']} shots/{res['bins']:,} bins"
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
        "also require DSM forecast and detection rows. elm-ours and DSM detection "
        "heads use shot-grouped out-of-fold predictions; the initialized variant "
        "retains a pretrained embedding with reviewed-shot overlap. Operating thresholds "
        "come only from each fold's inner-validation shots. ELM-O uses eta 0.997 "
        "for hard calls; its rank metrics come from the saved nested eta sweep "
        "on these same bins. Clock and always-present rows are rules; the clock "
        "seeded the review and is not independent. Brackets are 95\\% percentile "
        "intervals from 1,000 shared shot-bootstrap resamples. DSM refit, limited "
        "inputs (60 of the original 124), uses Hiro's survival/onset training "
        "source with reviewed-shot overlap, unavailable diagnostics mean-filled and "
        "inputs clipped at "
        "$|z|=10$; its original full-input model is not evaluated here. F1 is "
        "marked degenerate when recall $\\geq0.99$."
    )
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
    swap_tex.write(records["swap"], out)
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
