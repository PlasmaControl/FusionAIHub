# Recommender on Frontier: branch consolidation, IGNITE v4, simulation loop, docs site

Written 2026-09-19. Owner: nchen. Status: approved in chat, plan to follow.

## Goal

Run the Shot Designer recommender end to end on Frontier with the full DIII-D corpus:
a request in natural language -> reference-shot retrieval -> an LLM-proposed actuation
-> an IGNITE rollout of the proposed shot -> figures. Consolidate the `recommender` branch
into a new `nathan_dev`, migrate the recommender from the pinned v2 IGNITE bundle to the v4
codec/dynamics generation, and replace the flat `docs/` tree with a Docusaurus site.

Success is the three demo prompts (tearing-mode control with ECCD/ECH, ELM control with
RMP/coils, Alfven-eigenmode control with NBI) each producing: prompt, reasoning trace,
retrieved references, proposed actuation, saved HDF5, simulated diagnostics and panels.

## Decisions taken (with the user, 2026-09-19)

| Question | Decision |
|---|---|
| Target branch | New `nathan_dev` created from `origin/recommender` (a fast-forward of `nathan_fm`); `dev-peter`'s one extra commit merged in; the uncommitted `nathan_fm` work committed on top; `origin/recommender` deleted; `nathan_fm` untouched |
| IGNITE generation | v4 end to end: `ignite_codecs_v4` (15 modalities, vocab 1000, 1209 tok/frame, t0 = 1.0 s) and the `ignite_prod_v4` dynamics line, accepting that its rollouts are early and weak |
| LLM | One provider: `agy` (Antigravity CLI) running Gemini Flash (`gemini-3.8-flash-high` for design, `gemini-3.8-flash-low` for the offline shot blurbs). Gemma/Ollama is dropped (user, 2026-09-19 13:26); the Ollama v0.34.2 install already under `proj-shared/nchen/ollama` stays on disk but is not a provider |
| Text corpus | No `sql/` copy (user, 2026-09-19 13:26). The Frontier `shotsummary/processed/per_shot_txt` bundles (22,950 shots) are the whole text source; `mpid` is therefore `None` for every shot (the <= 5 per-MP cap is skipped) and `labels/claims.py` sees no logbook text. Blurbs are regenerated on Frontier with Gemini Flash. The database targets about 5000 shots, not 500 |
| Docs | Claude/superpowers planning material lives under `.claude/`, never `docs/`; `docs/` becomes a Docusaurus site |

## Facts the design rests on (measured 2026-09-19)

- `origin/recommender` = `nathan_fm` + 692 commits, 0 behind. `dev-peter` has exactly one
  commit not in it (`f71acd4`, IGNITE actuator counterfactuals); a dry merge conflicts in
  `src/tokamak_foundation_model/ignite/{eval_dynamics,maskgit}.py` only.
- `nathan_dev` does not exist on the remote; `dev-nathan` is a stale Feb 2026 branch and is
  not touched.
- `nathan_fm` working tree: 3 modified files (`_frontier_settings.sh`,
  `benchmark_plugin_perf.sh`, `ignite/train_dynamics.py`) plus ~35 untracked evaluation
  scripts and `docs/eval_improvement_recommendations.md`.
- v4 codecs: `/lustre/orion/fus187/proj-shared/models/ignite_codecs_v4/{m}/codec_best.pt`,
  15 modalities = the recommender's 14 + `mirnov` (spectro, 192 tok). Every vocab is 1000.
  Frame-code cache `ignite_prod_v4/frame_codes/` has 8752 shots, keys
  `codes/actuators/n_frames/vocabs`, actuators `(F, 88) float16`, `t0_start = 1.0`,
  1209 tokens/frame (manifest `_codec_manifest.json`).
- v4 dynamics: all runs are d1024/depth16/k0 20/n_predict 80, `actuator_dim` 88,
  15 modalities read from the checkpoint. Best by val masked-CE is
  `ignite_prod_v4/runs/mskfull/dynamics_best.pt` (step 3200, masked_ce 2.17, gen 2.72).
  Every arm overfits after 1-3k steps; Peter's runs are still writing (mtime today).
  The v2 bundle (prod_nfullrs2@13400, gen 1.42, 14 modalities) is the last validated one.
- The recommender's IGNITE code does not know `mirnov`; `FROZEN_MODALITIES` in
  `dynamics_config.py` and the pinned `production_vocabs` in `ignite_modalities.yaml`
  describe v2. The bundle's `ignite_infer.load_dynamics` reads `modalities` and the
  actuator dim from the checkpoint and so loads v4 already.
- `mirnov` exists as a group in the Frontier `<shot>_processed.h5` files (33 groups).
- Frontier text corpus (`proj-shared/foundation_model_text`, 24 GB) has
  `shotsummary/processed/per_shot_txt` (22,950 shots) and `shotsummary/raw/<run>/`, but not
  `sql/logs.jsonl` and `sql/index.json`, which `shotdb/logs.py` reads.
- No Ollama binary and no LLM API key on Frontier. `agy` (1.2.7) supports
  `-p --output-format json --json-schema <schema>` for non-interactive structured output.
  Ollama v0.34.2 ships `ollama-linux-amd64-rocm.tar.zst`.
- Node 22.23 / npm 10.9 are available on the login node.
- Working Pythons with ROCm torch 2.10: `.pixi/envs/frontier/bin/python` (3.11) and
  `/lustre/orion/fus187/scratch/nchen/tokeye/.venv/bin/python` (3.13).
- Encoding a shot for embeddings is I/O-bound: ~22 s/shot with 8 workers on Stellar
  (2 s GPU, 16 s HDF5 + STFT). 5000 shots on one GCD is ~30 h; an 8-task array on one node
  is ~4 h.

## 1. Git consolidation

1. `git branch nathan_dev origin/recommender`; switch to it carrying the `nathan_fm` working
   changes (stash, checkout, pop; resolve conflicts in the three modified files in favour of
   the newer recommender code where they overlap, keeping the `nathan_fm` intent).
2. Commit the carried work as one commit: "nathan_fm: Frontier settings, plugin benchmark,
   train_dynamics edits, Stage-1 evaluation scripts".
3. `git merge origin/dev-peter`; resolve the two conflicts keeping both Peter's
   counterfactual/dropout/Gumbel additions and the recommender-side changes; run the
   IGNITE unit tests that exist (`tests/` under `tokamak_foundation_model`) before committing.
4. Move Claude material: `git mv docs/superpowers .claude/superpowers`,
   `git mv .superpowers .claude/superpowers-runtime` (or delete if it is only cache; decide
   by content), and move status/finding notes that are working notes rather than
   documentation (`docs/phase_c_step1_status.md`, `docs/spectro_video_status.md`,
   `docs/spectrogram_step0_findings.md`, `docs/eval_stage1_panels_patch.md`,
   `docs/eval_improvement_recommendations.md`, `docs/*_plan.md` that describe Claude
   sessions) to `.claude/notes/`. Fix links. Real design docs (`IGNITE_DESIGN.md`,
   `E2E_ARCHITECTURE.md`, `SHOT_DESIGN*.md`, `LABELER.md`, `CLUSTERS.md`,
   `IGNITE_CODEC_RETRAIN_SPEC.md`, `IGNITE_ROLLOUT_QUALITY_PLAN.md`, tokenizer plans) stay
   and become site pages.
5. Push `nathan_dev`; `git push origin --delete recommender` only after the push is verified
   and the branch tip equals the local one.

## 2. Frontier roots and environment

Data roots (create, group `fus187`, setgid):

```
/lustre/orion/fus187/proj-shared/nchen/shot_design   SHOT_DESIGN_DATA_ROOT  (db/, raw/, ignite_inputs/, outputs/, runs/, llm/, llm_cache/, text_cache/, actuations/, sessions/)
/lustre/orion/fus187/proj-shared/nchen/labeler       LABELER_ROOT           (events/, labels/, models/ as provided)
/lustre/orion/fus187/proj-shared/nchen/ollama        {bin,models,home}  (installed 2026-09-19, unused: not a provider)
/lustre/orion/fus187/proj-shared/foundation_model    SHOT_DESIGN_CORPUS     (exists)
/lustre/orion/fus187/proj-shared/foundation_model_text/sql   (absent by decision; paths keep pointing here so a future copy needs no code change)
```

`configs/shot_design/paths.yaml` keeps Stellar defaults; a new
`configs/shot_design/paths.frontier.yaml` carries the Frontier values and is selected by the
env's activation (`SHOT_DESIGN_PATHS`). `pyproject.toml` gets a feature `ideate-frontier`
(ROCm torch from the `frontier` feature's index, ideate's PyPI deps, activation env with the
three roots and `SHOT_DESIGN_PATHS`) and an environment
`ideate-frontier = ["ideate", "ideate-frontier"]`. If `ideate` cannot resolve against ROCm
torch in pixi, the fallback is a plain `uv` venv described in the docs; the activation
contract (three roots + paths file) is the same either way.

Slurm: `scripts/slurm_frontier/shot_design_{encode,build,census,serve_llm,simulate}.sh`,
each sourcing `_frontier_settings.sh`, `-A fus187 -p batch`, logs under
`$SHOT_DESIGN_DATA_ROOT/runs/slurm/`. Job stats gate: `labeler.jobstats` grows a Frontier
backend reading `sacct` plus a `rocm-smi` sample written by the job itself.

Hugging Face: `sentence-transformers/all-MiniLM-L6-v2` is pre-populated from the login node
into `$HF_HOME` (default), offline afterwards.

## 3. IGNITE v4 migration

`configs/shot_design/ignite_modalities.yaml` `model:` block becomes generation-aware:

```yaml
model:
  generation: v4
  codec_tmpl: /lustre/orion/fus187/proj-shared/models/ignite_codecs_v4/{m}/codec_best.pt
  codec_manifest: <models_dir>/ignite_v4/MANIFEST.json      # written once by `shot_design model --pin`
  dynamics_file: <models_dir>/ignite_v4/dynamics_mskfull_step3200.pt
  frame_codes_cache: /lustre/orion/fus187/proj-shared/models/ignite_prod_v4/frame_codes
  t0_start_s: 1.0
  frame_tokens: 1209
  production_vocabs: {all 15: 1000}
```

- `shot_design model --pin` copies the chosen dynamics checkpoint and resolves the codec
  symlinks into `<models_dir>/ignite_v4/`, writes `MANIFEST.json` (name, family, tokens,
  vocab, sha256, source path, date) and refuses to run if any sha changes later. Peter's live
  runs cannot silently change the demo.
- `tokamak_foundation_model.ignite.dynamics_config`: `FROZEN_MODALITIES` stays as the v2
  default; add `modalities_from_manifest(path)` and make `train_dynamics._single_shot_dataset`
  accept any spectro name that exists as an HDF5 group (mirnov uses the same STFT path as mhr,
  verified against the v4 cache on 3 shots before use).
- `shotdb/ignite.py`: `load_codecs` reads the manifest (local template or HF bundle);
  `frame_codes` first looks in `frame_codes_cache` and only encodes when absent;
  `encode_shot` (continuous latents for `ignite_knn`) is unchanged apart from the modality
  list. Vocab validation compares cache `vocabs` to the checkpoint's modality table.
- `design/seed.py` docstring and G-ENC gate updated to 15 modalities; the `g_enc.py` check
  compares fresh encodes against the v4 cache instead of the bundle's samples.
- `eval_dynamics.load_model` reads `actuator_dim` and `modalities` from the checkpoint
  (the known 70-channel bug) so the repo's own loader works for v2 and v4.

Validation: on 3 held-out cache shots, fresh v4 frame codes match the cache exactly for the
non-spectro modalities and at >= 99 % of tokens for spectro/video (the same bar the v2 G-ENC
used); `validate_shot` rejects a v2 cache against the v4 checkpoint.

## 4. Simulation stage

New module `src/shot_design/simulate/` and CLI `shot_design simulate <design-ident>`:

1. Load the saved design (HDF5 + program JSON): reference shot, prediction window
   (default 1-5 s), proposed actuator waveforms in physical units.
2. Seed: reference shot's v4 frame codes (cache or fresh encode); actuators from
   `design/actuators.py` (88-channel builder) for both the reference (real) and the proposal,
   z-scored with the reference shot's own per-channel stats, so the proposal is expressed on
   the same scale the cache uses.
3. Rollout: k0 = 20 frames (1 s), n_predict = 80 (4 s), paired seeds; arms `real` and
   `proposed`; optional persistence baseline per modality.
4. Decode: filterscopes (ELM), mhr and mirnov band power (TM/AE), ts_core_density and
   ts_core_temp; token-space `divergence_vs_real` and decoded band-restricted deltas.
5. Write `outputs/<design>/simulation.h5` (tokens, decoded arrays, actuators, metadata with
   checkpoint sha), `panels/*.png`, `report.md` (skill vs persistence, static-fraction, and
   an explicit "v4 dynamics is early; treat as qualitative" line when skill < 0).

Frontier execution: `scripts/slurm_frontier/shot_design_simulate.sh <ident>` on one GCD,
`-q debug` for the demo (under 2 h). UI: `POST /api/design/{ident}/simulate` submits the
sbatch and records `simulate/status.json`; `GET .../simulate` polls it; panels are served
when done. No in-process GPU inference.

## 5. LLM provider

`configs/shot_design/llm.yaml` gains `provider: agy | ollama | off` and per-provider blocks; `agy`
is the only provider used on Frontier.

- `agy`: `shot_design/llm/agy.py` runs `agy --model <model> --output-format json --json-schema
  <schema> --disable-slash-commands -p=<prompt>` from the login node with the messages rendered
  into one prompt. (`-p` must be attached with `=`; `agy -p --model ...` takes `--model` as the
  prompt.) Tool calling is emulated: the tool list is rendered into the prompt and the schema is
  `{content: str, tool_calls: [{name, arguments}]}`; the assistant loop treats the result exactly
  as an OpenAI reply. Cache keyed on the full request as today. Timeout and retries configurable;
  the reasoning text the model emits is kept in the trace.
  Measured 2026-09-19: one `gemini-3.8-flash-low` structured call = 5.7 s wall, 24k input tokens
  of agy system overhead, `structured_output` may come back one level over-encoded
  (`{"content": "{\"content\": \"ready\"}"}`), so the parser unwraps a `content` that itself
  parses as an object with the schema's keys.
- Models: `quality: gemini-3.8-flash-high` (design assistant, describe.polish),
  `fast: gemini-3.8-flash-low` (blurbs). `agy models` lists the ids.
- The Gemma-specific code goes: `design/assistant.py`'s `"gemma" not in model_name` guard becomes
  a `provider == "off"` check and its stage labels/prompts say "the model", not "Gemma".
- Blurb backfill: `shot_design blurb` gains `--workers N` (thread pool over the per-shot model
  calls; the parquet rewrite stays single-process) so ~5000 blurbs at ~6 s each finish in about an
  hour at 8 workers. Runs on the login node; no GPU.
- `ollama`: the existing OpenAI-compatible path is kept as a provider name for Stellar; nothing
  on Frontier serves it.

## 6. Database at ~5000 shots

1. `shot_design corpus scan` over all `<shot>_processed.h5` on Frontier (parallel, header
   only) -> census parquet.
2. `shot_design corpus select --n 5000 --name recommender_frontier_v1` with the same rule as
   `recommender_v1` (abs Ip, >= 200 shot chars, physics-first themes, two-pass flat top),
   theme quotas scaled by 10 and the AE quota filled first since it was unmet at 500. The
   list is committed under `configs/shot_design/shot_lists/`.
3. `shot_design logs import` from `foundation_model_text/sql` once the user's copy lands;
   until then the build runs with `--no-logs` and is rebuilt after (build is atomic).
4. `shot_design labels join` from the in-git `data/events/*/format` tables plus whatever the
   user places in `LABELER_ROOT`.
5. `shot_design build` (CPU-heavy, one node) then `shot_design encode` embeddings as an
   8-task array on one node (~4 h), followed by the coverage report and the `describe 190736`
   parity check against the Stellar record.

## 7. Demo

`scripts/shot_design/demo_frontier.sh` runs the three prompts through
`shot_design assistant --prompt-file ... --provider agy`, then `shot_design simulate` for
each design, and writes `outputs/recommender_frontier_demo/<slug>/` with `prompt.md`,
`trace.jsonl` (every LLM round and tool call), `references.md`, `actuation.csv` and the
HDF5, `simulation/` panels and `report.md`, plus `outputs/recommender_frontier_demo/index.md`
linking all three. The same page is added to the docs site under "Examples".

## 8. Documentation site

`website/` is a Docusaurus 3 classic TypeScript site in docs-only mode
(`routeBasePath: '/'`, `blog: false`, `path: '../docs'`), built with Node 22 on the login
node and deployed to GitHub Pages by `.github/workflows/docs.yml` on pushes to `nathan_dev`
(switch to `main` when merged). `docs/` is reorganised into sidebar sections with
`_category_.json` files:

```
docs/
  intro.md                      (from README: what FAITH is, install, quick start)
  getting-started/              install (pixi envs per platform), data layout, running a demo
  clusters/                     stellar.md, frontier.md (from CLUSTERS.md, updated with the actuals from this work), adding-a-cluster.md
  data/                         corpus, modalities, text corpus, preprocessing stats
  models/                       foundation model, IGNITE (design, codecs, dynamics, rollout quality), spectrogram/video tokenizers
  shot-design/                  overview, CLI, database build, retrieval, LLM providers, simulation, UI, MCP, evaluation
  labeler/                      LABELER.md
  examples/                     the three demo prompts with figures
  reference/                    configs, environment variables, SLURM scripts
```

Existing pages are moved, not rewritten, except for front matter and link fixes; new pages
are the Frontier quick start, LLM providers, simulation, and the examples. `docs/` contains
no Claude or superpowers material.

## Testing

- `tests/shot_design` green under the Frontier env (`pixi run -e ideate-frontier pytest`).
- New: manifest parse/pin (sha mismatch refuses), 15-modality frame layout equals 1209,
  `validate_shot` v2-vs-v4 rejection, `agy` provider (subprocess mocked: schema shape, tool
  call emulation, timeout), simulate module on a 4-frame synthetic cache (shapes, z-score
  round trip, report fields), UI simulate route (submits, polls, serves panels; sbatch
  mocked).
- `npm run build` for the site passes with `onBrokenLinks: 'throw'`.

## Out of scope

Retraining or selecting among Peter's dynamics arms beyond pinning one; MDSplus fetching on
Frontier (not reachable); merging `nathan_dev` into `main`; the stale `dev-nathan` branch.

## Execution

Plan in `.claude/superpowers/plans/2026-09-19-recommender-frontier-port.md`. Sections 1-5
and 8 start immediately; 6 no longer waits on anything external; 7 waits on 6. Claude subagents own
sections 1, 3, 4, 5 (judgement-heavy); Antigravity owns the bulk ports in 2 and 8 (Slurm
copies, docs moves and front matter, test scaffolds), verified by the
conductor.
