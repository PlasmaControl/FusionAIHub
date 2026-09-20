#!/bin/bash
# Run after the three `shot_design_simulate.sh` jobs demo_frontier.sh submitted have
# finished (see each outputs/recommender_frontier_demo/<slug>/simulate_job). Copies
# each design's simulation/{report.md,panels} next to its other demo files and writes
# outputs/recommender_frontier_demo/index.md linking all three.
set -euo pipefail
source "$(dirname "$0")/../slurm_frontier/_shot_design_common.sh"
OUT="$ROOT/outputs/recommender_frontier_demo"

for slug in tearing_eccd elm_rmp ae_nbi; do
  d="$OUT/$slug"
  ident=$(cat "$d/design_id")
  sim="$ROOT/outputs/$ident/simulation"
  if [ ! -f "$sim/report.md" ]; then
    echo "warning: $slug ($ident): $sim/report.md not there yet -- skipping" >&2
    continue
  fi
  cp "$sim/report.md" "$d/report.md"
  rm -rf "$d/panels"
  cp -r "$sim/panels" "$d/panels"
done

"$PY" - "$OUT" tearing_eccd elm_rmp ae_nbi <<'PY'
import json
import sys
from pathlib import Path

out_dir, slugs = Path(sys.argv[1]), sys.argv[2:]
lines = ["# Recommender Frontier demo", ""]

for slug in slugs:
    d = out_dir / slug
    report = d / "report.md"
    if not report.exists():
        lines += [f"## {slug}", "", "Simulation has not finished yet.", ""]
        continue

    ident = (d / "design_id").read_text().strip()
    metadata = {}
    try:
        import h5py

        with h5py.File(d / "design.h5", "r") as f:
            metadata = json.loads(f["metadata"][()])
    except OSError:
        pass  # design.h5 not copied yet; the rest of the section still renders

    prompt = (d / "prompt.md").read_text().strip()
    explanation = (metadata.get("explanation") or "").strip()
    excerpt = explanation if len(explanation) <= 280 else explanation[:277] + "..."
    references = [metadata.get("reference_shot"), *metadata.get("comparison_shots", [])]
    references = [r for r in references if r is not None]

    table_lines, in_table = [], False
    for line in report.read_text().splitlines():
        if line.startswith("| modality"):
            in_table = True
        if in_table:
            table_lines.append(line)
            if line.strip() == "":
                break

    lines += [
        f"## {slug}",
        "",
        f"**Design id:** `{ident}`  ",
        f"**Model:** {metadata.get('model', '?')}  ",
        f"**Reference shots:** {', '.join(str(r) for r in references) or '?'}",
        "",
        "**Prompt:**",
        "",
        f"> {prompt}",
        "",
        f"**Explanation:** {excerpt or '(none)'}",
        "",
        *table_lines,
    ]
    panels = sorted((d / "panels").glob("*.png")) if (d / "panels").exists() else []
    for p in panels:
        lines.append(f"![{p.stem} ({slug})]({slug}/panels/{p.name})")
    lines.append("")

(out_dir / "index.md").write_text("\n".join(lines) + "\n")
print(f"wrote {out_dir / 'index.md'}")
PY
