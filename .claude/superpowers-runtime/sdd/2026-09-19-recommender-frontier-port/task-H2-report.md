# Task H2 report — docs reorganisation into Docusaurus sections

Worktree: `/lustre/orion/fus187/scratch/nchen/FusionAIHub-wt-h`, branch `nathan_dev-h`.
Commit: `fa50938` — "docs: reorganise into Docusaurus sections; Frontier, LLM
providers, simulation and database pages" (37 files changed, 1579
insertions, 584 deletions). Working tree clean after commit.

## Final `docs/` tree

```
docs/intro.md                                    (slug: /, sidebar_position 1)

docs/getting-started/_category_.json
docs/getting-started/install.md

docs/clusters/_category_.json
docs/clusters/stellar.md                          (from CLUSTERS.md)
docs/clusters/frontier.md                         (new)
docs/clusters/adding-a-cluster.md                 (new)

docs/data/_category_.json
docs/data/corpus.md                               (new)

docs/models/_category_.json
docs/models/foundation-model.md                   (from E2E_ARCHITECTURE.md)
docs/models/ignite.md                             (from IGNITE_DESIGN.md, +v4 section)
docs/models/ignite-codec-retrain-spec.md          (from IGNITE_CODEC_RETRAIN_SPEC.md)
docs/models/ignite-rollout-quality-plan.md        (from IGNITE_ROLLOUT_QUALITY_PLAN.md)
docs/models/spectrogram-tokenizer-plan.md         (from spectrogram_tokenizer_plan.md)
docs/models/stage2-genvid-integration-plan.md     (from stage2_genvid_integration_plan.md)
docs/models/stage2-with-video-plan.md             (from stage2_with_video_plan.md)
docs/models/video-tokenizer-plan.md               (from video_tokenizer_plan.md)

docs/shot-design/_category_.json
docs/shot-design/overview.md                      (from SHOT_DESIGN.md, remainder)
docs/shot-design/cli.md                           (split from SHOT_DESIGN.md lines 20-47)
docs/shot-design/mcp.md                           (split from SHOT_DESIGN.md lines 395-488, genericized)
docs/shot-design/llm-providers.md                 (split from SHOT_DESIGN.md lines 516-649 + new Frontier/agy section)
docs/shot-design/programs.md                      (from SHOT_DESIGN_PROGRAMS.md)
docs/shot-design/simulation.md                    (new)
docs/shot-design/database-build.md                (new)

docs/labeler/_category_.json
docs/labeler/overview.md                          (from LABELER.md)

docs/examples/_category_.json
docs/examples/index.md                            (new, short placeholder per ruling)

docs/reference/_category_.json
docs/reference/research-plan.md                   (from ResearchPlan.MD)
docs/reference/environment-variables.md           (new)
docs/reference/slurm-scripts.md                   (new)
```

Every directory under `docs/` has a `_category_.json`. Every `.md` file has
front matter (`title` + `sidebar_position`; `docs/intro.md` also has
`slug: /`). No `sidebar_position` collisions remain within any directory
(verified programmatically after the fact).

## New pages and their fact sources

- `docs/intro.md` — landing page; repository layout facts from `CLAUDE.md`
  and direct directory listing (`src/tokamak_foundation_model/`,
  `src/shot_design/`, `src/labeler/`, `configs/shot_design/`, `scripts/`,
  `data/`).
- `docs/clusters/frontier.md` — sourced from real files read in the
  worktree: `configs/shot_design/paths.frontier.yaml`, `pyproject.toml`
  (pixi environments/features), `scripts/slurm_frontier/_shot_design_common.sh`,
  `_frontier_settings.sh`, `_frontier_common.sh`, and the plan's Phase B
  text (`.claude/superpowers/plans/2026-09-19-recommender-frontier-port.md`).
- `docs/clusters/adding-a-cluster.md` — generalised from the concrete
  Stellar/Frontier split (three env vars + `SHOT_DESIGN_PATHS` pattern,
  confirmed against `configs/shot_design/paths.yaml` key set).
- `docs/data/corpus.md` — CLAUDE.md's modality table plus
  `TokamakH5Dataset`/`SIGNAL_CONFIGS`/`MOVIE_CONFIGS` docstrings in
  `data_loader.py`.
- `docs/reference/environment-variables.md`, `slurm-scripts.md` — read
  every `scripts/slurm_frontier/*.sh` wrapper directly
  (`shot_design_census.sh`, `_build.sh`, `_encode.sh`, `_genc.sh`,
  `_simulate.sh` [confirmed stub], `train_dynamics.sh`, `eval_dynamics.sh`,
  `_gpu_sampler.sh`, `setup_frontier_env.sh`) rather than the plan text.
- `docs/shot-design/simulation.md` — read `src/shot_design/simulate/cli.py`
  in full (`run()`, `_write_status`, `_windowed_real_actuators`,
  `DEFAULT_DECODE`) and the `simulate` subparser in `src/shot_design/cli.py`.
  Documents the `status.json` atomic-write contract and explicitly flags
  that the Actuator-editor UI route (`/api/design/{ident}/simulate`) is
  concurrent Phase D4 work not landed in this checkout at time of writing
  (confirmed absent: `src/shot_design/ui/simulate_routes.py` does not exist
  here).
- `docs/shot-design/database-build.md` — sourced from Phase F1/F2 plan text
  plus the real `scripts/slurm_frontier/shot_design_{census,build,encode}.sh`
  content. States up front that shot counts are targets, not measured
  results, per Controller Ruling 1 (no invented numbers). Flags
  `blurb_frontier.sh` / `--workers` as concurrent Phase E3 work not yet
  present in this checkout.
- `docs/models/ignite.md` §12 "v4 generation" — modality table (15
  modalities incl. `mirnov`), pinned-bundle YAML, pin/check commands, G-ENC
  gate description written as "no run recorded yet — read
  `$SHOT_DESIGN_DATA_ROOT/runs/genc_v4.json`" rather than fabricating a
  pass/fail; dynamics checkpoint caveat (`mskfull/dynamics_best.pt` step
  3200, early/qualitative).
- `docs/shot-design/llm-providers.md` — kept the original Ollama/Gemma
  content verbatim, appended a new Frontier/`agy` section from Phase E1/E2
  plan text; confirmed via `configs/shot_design/llm.yaml` that this
  checkout is still `provider: ollama` (agy not yet landed) and said so.

## Build output (final, clean)

```
[webpackbar] ✔ Server: Compiled successfully in 1.16s
[webpackbar] ✔ Client: Compiled successfully in 1.71s
[SUCCESS] Generated static files in "build".
```

Zero MDX compilation errors, zero broken links, zero broken anchors.

### Errors fixed this session (beyond the 3 already fixed before this
session's start)

1. `docs/models/ignite-codec-retrain-spec.md` — a second, sibling
   `EXTRA_ARGS="<from table>"` / `<id>` pair in the same indented block
   triggered `Expected a closing tag for <from>`. Fixed by converting the
   whole indented block to a fenced ` ```bash ` code block (fenced code is
   never MDX/JSX-parsed, unlike 4-space-indented code, which this build
   was still tokenizing as prose in places).
2. `docs/models/spectrogram-tokenizer-plan.md` line 127 — a second,
   previously-unseen `<2.1%)` occurrence (risk-register table, distinct
   from the line ~68 occurrence fixed earlier) — backtick-escaped.
3. `docs/reference/research-plan.md` — a bare `<10s`/`<30s`/`<60s`/`<15
   min`/`<50 ms` family across the §5.10 test-execution table and two
   latency-target lines (lines 221-231, 271, 287) — all backtick-escaped.
   This file had never been part of a Docusaurus build before, so these
   were previously-undetected pre-existing issues in `ResearchPlan.MD`,
   not something introduced by this task.
4. `docs/labeler/overview.md` line 409 — `[data/events/README.md](../../data/events/README.md)`
   pointed outside the `docs/` content root (to the repo-root
   `data/events/README.md`), which Docusaurus's docs plugin cannot resolve
   as a page and flags as a broken markdown link even though the target
   file genuinely exists on disk. Fixed by turning it into inline code
   (`` `data/events/README.md` (repository root, outside this docs tree)
   ``) rather than a hyperlink — preserves the reference without claiming
   it is a page in this site. No `onBrokenMarkdownLinks` weakening used.
5. Three broken-anchor warnings: `docs/clusters/frontier.md`,
   `docs/shot-design/cli.md`, `docs/shot-design/simulation.md` all linked
   to `../models/ignite.md#v4-generation`, but Docusaurus's GitHub-style
   heading slugger turns `## 12. v4 generation` into `12-v4-generation`
   (keeps the leading number, drops the period). Fixed all three links to
   `#12-v4-generation`.

## `npm run serve` check

Started `npm run serve -- --port 3111 --no-open` in the background,
`curl -s localhost:3111/FusionAIHub/` (site `baseUrl` is `/FusionAIHub/`)
returned HTTP 200 and 19 occurrences of "FusionAIHub" in the rendered
HTML. Server process confirmed stopped afterward (no docusaurus/serve
process left running).

## Test run

```
cd /lustre/orion/fus187/scratch/nchen/FusionAIHub-wt-h
PYTHONPATH=$PWD/src /lustre/orion/fus187/scratch/nchen/FusionAIHub/.pixi/envs/shot-design-frontier/bin/python \
  -m pytest tests/shot_design/test_phenomena.py -q -p no:cacheprovider
```
Result: `85 passed in 6.13s`. (The `distutils-precedence.pth`
`_distutils_hack.add_shim` stderr traceback that appears on `python3`
invocations is unrelated interpreter noise per the brief and did not
appear here since this used the pixi env's `python` directly.)

`tests/shot_design/test_phenomena.py`'s docs-path tuple points at
`docs/shot-design/overview.md`; the `RANKING_SENTENCE` text it asserts on
is preserved verbatim in that file's Phenomena section.

## Files changed (staged and committed)

`git add -A docs README.md tests/shot_design/test_phenomena.py` — 37 files:
13 renames (`git mv`-tracked, preserving history) from the original flat
`docs/*.md` names, 19 new files (`_category_.json` × 8 + 11 new/split
`.md` pages), 1 deletion (`docs/CLUSTERS.md`, fully split into three
`clusters/` files with no `git mv` since it fans out 1→3), plus `README.md`
and `tests/shot_design/test_phenomena.py` modifications. No files outside
this scope were staged (verified via `git status --short` before commit).

## Self-review findings

- `git grep -in "claude|superpowers|subagent" -- docs README.md` and the
  same with `--untracked`: both exit 1 (no matches) — clean, including all
  the Claude Code / superpowers references found and removed earlier this
  task (4 extra `.claude/superpowers/specs/...` citations in
  `labeler/overview.md`, one `.claude/notes/...` citation in
  `spectrogram-tokenizer-plan.md`, and the "Attaching it in Claude Code"
  MCP subsection genericized).
- Every `docs/` subdirectory has `_category_.json`; every `.md` file has
  front matter; no `sidebar_position` collisions; only `docs/intro.md`
  remains directly under `docs/` (by design, via `slug: /`).
- Page lengths: all new/split pages fall in a reasonable range (11-872
  lines). `docs/examples/index.md` is the deliberate 11-line placeholder
  per Controller Ruling exemption. `docs/shot-design/cli.md` is short (40
  lines) because it is a verbatim extraction of SHOT_DESIGN.md's original
  CLI section (lines 20-47), not a newly-authored page — no padding was
  added, consistent with the no-invented-content constraint.
- Relative-link depth was checked repo-wide
  (`grep -rnoE ']\(\.\./\.\./[^)]+\)' docs/`): the only remaining `../../`
  link was the `data/events/README.md` one fixed above (now not a
  hyperlink); all other cross-directory links are single `../<dir>/file.md`
  and resolve correctly at their new depth.

## Concerns for later tasks

- `docs/shot-design/simulation.md` and `docs/shot-design/database-build.md`
  document Phase D4 (Actuator-editor UI simulate route), Phase E1/E2
  (`agy` LLM provider), and Phase E3 (`blurb_frontier.sh` parallel
  backfill) purely from the plan's contracts, since none of that code
  exists in this worktree yet. Background agents for D4, E1/E2 (review),
  and E3 were still running elsewhere in this session's task tree at the
  time this doc work was done — when those land, the caveat sentences in
  these two pages ("had not landed in this checkout at the time this page
  was written" / "may not be present in every checkout yet") should be
  revisited and likely removed once the corresponding code merges.
- `docs/models/ignite.md` §12's G-ENC section deliberately has no
  pass/fail result recorded (Controller Ruling 1) — once a real G-ENC run
  exists, someone should replace the "read `runs/genc_v4.json` yourself"
  pointer with the actual measured fractions.
- The MDX build pattern observed (each `npm run build` surfaces the next
  unescaped `<...>` sequence only after the previous one in the same file
  is fixed, rather than all at once) means it is possible — though I
  checked every remaining `<[a-zA-Z0-9]` occurrence across all touched
  files after the last clean build — that a stray `<...>` pattern from
  files nobody touched this task (e.g. inside `docs/models/*-plan.md`
  files that were only front-matter-stamped, not content-edited) could
  still exist outside what a bare `grep` catches (e.g. multi-line JSX-like
  constructs). The final `npm run build` came back fully clean with no
  errors or warnings of any kind, which is the authoritative signal.

## Fix round 1

Controller note (2026-09-20 12:30): the fixer agent committed d19eda4 on `nathan_dev-h` (6 files: docs/clusters/frontier.md, docs/reference/slurm-scripts.md, docs/shot-design/database-build.md, docs/shot-design/llm-providers.md, docs/shot-design/simulation.md, src/shot_design/retrieval/phenomena.py; +156/-79) and reported in chat that `cd website && npm run build` succeeded and `tests/shot_design/test_phenomena.py` passed (85 passed in 5.99 s), but the session was interrupted before it appended its per-page before→after list here. Treat all claims as unverified: the re-review verifies from the diff and re-runs both gates.
