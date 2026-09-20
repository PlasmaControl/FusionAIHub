### Task G1: `scripts/shot_design/demo_frontier.sh` and the three prompts

**Files:**
- Create: `configs/shot_design/evalsets/frontier_demo_prompts.yaml` (the three prompts verbatim from the user, slugs `tearing_eccd`, `elm_rmp`, `ae_nbi`), `scripts/shot_design/demo_frontier.sh`
- Modify: `src/shot_design/cli.py` — `assistant` subcommand if absent (`--prompt`, `--provider`, `--trace out.jsonl`), wrapping `design.assistant.run_design` with a `progress` callback that appends `{"stage", "status", "detail", "t"}` lines and an `LLMClient` wrapper that appends every request/reply (`{"kind": "llm", "messages", "reply"}`) to the same trace.

- [ ] **Step 1:** Test `tests/shot_design/test_assistant_cli.py`: with `run_design` monkeypatched, `python -m shot_design assistant --prompt "x" --trace t.jsonl` writes a trace with at least the 5 stage lines and prints the design id.
- [ ] **Step 2–4:** fail → implement → pass; commit `git commit -m "shot_design: assistant CLI with JSONL trace"`.
- [ ] **Step 5: Demo script**

```bash
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
```
Add `design show` flags if they do not exist (read `cli.py:1839-1912` and `design/program.py:634-660`). `demo_frontier_collect.sh` copies `outputs/<ident>/simulation/{report.md,panels}` into each `$d/` and writes `$OUT/index.md` (prompt, model, reference shots, explanation excerpt, skill table, panel images).

- [ ] **Step 6: Run it** (after F2). Read every `report.md`; the deliverable is `$OUT/index.md`. Copy `index.md` + panels into `docs/examples/` (Task H2).

---

