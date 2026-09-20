### Task B5: Rename `ideate` to `shot-design` everywhere live

User directive 2026-09-19 16:11: "rename all the ideate to shot_design". Pixi rejects underscores in environment names (measured: `Failed to parse environment name 'shot_design', please use only lowercase letters, numbers and dashes`), so environment/feature names use the dash form; the Python package stays `shot_design`.

**Mapping (exact):**
- pixi features and environments: `ideate` → `shot-design`, `ideate-cpu` → `shot-design-cpu`, `ideate-frontier` → `shot-design-frontier` (in `pyproject.toml` `[tool.pixi.feature.*]`, `[tool.pixi.environments]`, task/comment text; regenerate `pixi.lock` so its environment keys follow; the installed dir becomes `.pixi/envs/shot-design-frontier`).
- `.mcp.json` `-e ideate-cpu` → `-e shot-design-cpu`; `src/shot_design/mcp/server.py` resource `ideate://manifest` → `shot-design://manifest`; `src/shot_design/ui/app.py` `COOKIE = "ideate_token"` → `"shot_design_token"`.
- Schema tags: `src/shot_design/shotdb/legacy_raw.py` `SCHEMA = "ideate-raw-v1"` → `"shot-design-raw-v1"`, `src/shot_design/design/provenance.py` `SCHEMA = "ideate-frame-codes-provenance-v1"` → `"shot-design-frame-codes-provenance-v1"`; wherever a reader compares the tag, also accept the old string (`LEGACY_SCHEMAS = {"ideate-raw-v1"}`) so Stellar files still load. Test: a file stamped with the old tag validates.
- Every `pixi run -e ideate*`, `.pixi/envs/ideate*` and prose "ideate" in `scripts/`, `src/`, `tests/`, `docs/`, `AGENTS.md`, `data/events/README.md`, `configs/shot_design/*.yaml` comments, and the CURRENT plan/spec (`.claude/superpowers/{plans,specs}/2026-09-19-*`).
- Slurm scripts under `scripts/shot_design/*.sbatch` and `scripts/slurm_frontier/_shot_design_common.sh`: env dir names as above.

**Do NOT rename:** the Stellar filesystem path string `/scratch/gpfs/EKOLEMEN/nc1514/ideate` (a real directory on another cluster; renaming the string cannot rename the directory), and the archival `.claude/superpowers/plans-recommender/`, `.claude/superpowers/specs/2026-09-0*`/`2026-09-1[0-8]*`, `.claude/superpowers-runtime/` files (history), and `tests/labeler/data/jobstats/*.txt` fixtures (verbatim Slurm output).

- [ ] **Step 1:** `grep -rIn "ideate" --exclude-dir=.pixi --exclude-dir=.git --exclude-dir=.superpowers --exclude-dir=node_modules . | grep -v "^./.claude/superpowers-runtime\|^./.claude/superpowers/plans-recommender\|^./.claude/superpowers/specs/2026-09-0\|^./.claude/superpowers/specs/2026-09-1[0-8]\|/scratch/gpfs/EKOLEMEN/nc1514/ideate\|tests/labeler/data/jobstats" > /tmp/ideate_before.txt; wc -l /tmp/ideate_before.txt`
- [ ] **Step 2:** Apply the mapping with `sed -i` per file class (dash form for env names, `shot_design_token` for the cookie, `shot-design://` for the MCP resource, `shot-design-` for schema tags, plain "shot design"/"shot_design" for prose as reads naturally). Add the legacy-tag acceptance + its test.
- [ ] **Step 3:** Re-run the Step 1 grep: expected 0 lines. `pixi install -e shot-design-frontier` (this replaces B2's install if B2 has not finished); `pixi run -e shot-design-frontier pytest tests/shot_design tests/labeler -q`.
- [ ] **Step 4:** `git mv` nothing (no file is named ideate outside archives); commit `git commit -m "repo: rename ideate environments and tags to shot-design"`.

## Phase C: IGNITE v4 migration

