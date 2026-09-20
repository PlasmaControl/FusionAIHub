#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/../slurm_frontier/_shot_design_common.sh"
OUT="$ROOT/outputs/recommender_frontier_demo"; mkdir -p "$OUT"
for slug in tearing_eccd elm_rmp ae_nbi; do
  d="$OUT/$slug"; mkdir -p "$d"
  "$PY" -m shot_design evalsets prompt frontier_demo_prompts "$slug" > "$d/prompt.md"
  ident=$("$PY" -m shot_design assistant --prompt "$(cat "$d/prompt.md")" --provider agy --trace "$d/trace.jsonl" | tail -1)
  echo "$ident" > "$d/design_id"
  "$PY" -m shot_design design show "$ident" --references > "$d/references.md"
  "$PY" -m shot_design design show "$ident" --actuation-csv > "$d/actuation.csv"
  cp "$ROOT/outputs/$ident.h5" "$d/design.h5"
  jid=$(sbatch --parsable "$REPO/scripts/slurm_frontier/shot_design_simulate.sh" "$ident")
  echo "$jid" > "$d/simulate_job"
done
echo "simulations queued; run scripts/shot_design/demo_frontier_collect.sh when they finish"
