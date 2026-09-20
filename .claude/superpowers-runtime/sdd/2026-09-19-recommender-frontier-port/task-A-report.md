# Phase A report — git consolidation (Tasks A1, A2, A3)

Branch `nathan_dev` = `origin/recommender` (a978f0e) + carried `nathan_fm` work + `origin/dev-peter` (f71acd4) + the `.claude` reorganization. Pushed; `origin/recommender` deleted.

**Status: DONE_WITH_CONCERNS** — all steps complete and verified; three deviations from the brief text, recorded below. The `.gitignore` one was mandatory.

---

## Commits created

| SHA | Subject |
|---|---|
| `e4fe84c` | nathan_fm: carry Frontier settings, plugin benchmark, train_dynamics edits and Stage-1 evaluation scripts onto nathan_dev |
| `294097f` | merge dev-peter: IGNITE actuator counterfactuals, dropout, Gumbel reveal, honest video panel |
| `e968501` | repo: move superpowers specs/plans and working notes under .claude, keep docs/ for documentation |
| `25667aa` | plan: serve_llm.sbatch is Stellar-only and stays, not a Frontier deletion |

`HEAD` = `origin/nathan_dev` = `25667aa54f7e83afed17d06c8de7395e5f4b5dc8`. Working tree clean (`git status --short --untracked-files=all` empty apart from the git-ignored `.superpowers/` workspace).

Merge parents of `294097f`: `e4fe84c` (nathan_dev) + `f71acd4` (origin/dev-peter).

Commits are authored `Nathaniel Chen <nchen@login04.frontier.olcf.ornl.gov>` (git's auto-detected identity on this login node; same shape as Peter's `f71acd4`). No co-author trailer on any commit — a repo hook actively blocks that line, so omitting it is correct.

---

## Task A1 — create `nathan_dev`, carry the `nathan_fm` working tree

### Starting state (recorded)

```
$ git rev-parse HEAD
702aaae4c258b96ca4b08ecd46b2dade12def1dc
$ git rev-parse --abbrev-ref HEAD
nathan_fm
```

3 modified tracked files, 47 untracked files (expanded with `--untracked-files=all`; plain `git status --short` collapses two of them into directories, which is why the brief says "35"). Saved to `/tmp/claude-20641/nathan_fm_status.txt` and `/tmp/claude-20641/nathan_fm_tracked.patch` (121 lines).

Ancestry confirmed before switching:

```
$ git merge-base --is-ancestor HEAD origin/recommender  -> true
$ git rev-list --count HEAD..origin/recommender          -> 692
$ git rev-list --count origin/recommender..HEAD          -> 0
$ git log --oneline origin/recommender..origin/dev-peter -> f71acd4 IGNITE: expose actuator counterfactuals ...
```

### Conflicts

**None.** `git stash pop` applied cleanly. Checked beforehand why: `git diff --stat HEAD origin/recommender` was empty for all three modified files, i.e. they were byte-identical on `702aaae` and `a978f0e`, so the stash could not conflict. Verified after the pop that the restored diff is byte-identical to the pre-stash patch:

```
$ diff /tmp/claude-20641/nathan_fm_tracked.patch /tmp/claude-20641/nathan_dev_tracked.patch
PATCHES IDENTICAL
```

No recommender-side equivalent of any `nathan_fm` hunk existed, so ruling 4's "prefer the recommender code" case never arose.

### Full list of untracked files added (ruling 3)

The brief's step-4 paths covered every untracked file — nothing fell outside them, so nothing extra had to be added. 47 files:

- `docs/eval_improvement_recommendations.md`
- `scripts/data_fetching_omega/n1rms_fetch/` — `README.md`, `config_n1rms.yaml`, `shots_n1rms.txt`, `shots_n1rms_finetune.txt`, `shots_n1rms_finetune_meta.csv` (5)
- `scripts/evaluation/` — `check_extract_vs_ckpt.py`, `extract_latents.py`, `ignite_bp_cases.py`, `ignite_build_bp_cache.py`, `ignite_case_panels.py`, `ignite_encode_shots.py`, `ignite_rerender.py`, `ignite_simple_plot.py`, `ingest_n1rms.py`, `smoke_data_probe.py`, `study_a_latent_analysis.py`, `study_a_probes.py`, `study_a_recon_fidelity.py`, `study_a_rollout_divergence.py`, `study_b_actuator_scan.py`, `study_b_figures.py`, `study_b_layerwise.py`, `study_b_physics_sweeps.py`, `study_c_elm_lora.py`, `study_c_make_labels.py`, `study_c_probes.py` (21)
- `scripts/evaluation/slurm/` — `_eval_env.sh`, `elm_lora.sbatch`, `extract_full.sbatch`, `extract_smoke.sbatch`, `ignite_ae_rerender.sbatch`, `ignite_bp_cases.sbatch`, `ignite_cases.sbatch`, `ignite_elm_tm.sbatch`, `ignite_rerender.sbatch`, `studies_gpu.sbatch`, `studies_makeup.sbatch` (11)
- `scripts/evaluation/tfm_eval/` — `data.py`, `inverse_preprocess.py`, `labels.py`, `latents.py`, `metrics.py`, `physics.py`, `plotting.py`, `probing.py` (8; these join `__init__.py` and `ckpt.py`, already tracked on recommender)
- `scripts/slurm_frontier/_frontier_common.sh`

Nothing under `.superpowers/` was added.

### Verification (real output)

```
$ PYTHONPATH=src .pixi/envs/frontier/bin/python -c "import tokamak_foundation_model.ignite.train_dynamics as t; print('ok')"
ok

$ PYTHONPATH=src .pixi/envs/frontier/bin/python -m pytest tests/ignite -x -q 2>&1 | tail -3
-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
517 passed, 5 skipped, 3 warnings in 257.18s (0:04:17)
```

Baseline taken on `nathan_fm` *before* switching, for comparison: `517 passed, 5 skipped, 3 warnings in 377.60s (0:06:17)`. Identical pass/skip set — no regression, so no failure text to record in the commit body.

`git push -u origin nathan_dev` → `* [new branch] nathan_dev -> nathan_dev`; `git rev-parse HEAD origin/nathan_dev` both `e4fe84c`.

---

## Task A2 — merge `dev-peter`

`git merge --no-ff origin/dev-peter` produced **three** conflicted files, not the two the brief predicted. The extra one is `train_dynamics.py`, which conflicts because A1's carried `_sync_grads` work and Peter's `--dropout` parameter land in the same file.

```
$ git diff --name-only --diff-filter=U
src/tokamak_foundation_model/ignite/eval_dynamics.py
src/tokamak_foundation_model/ignite/maskgit.py
src/tokamak_foundation_model/ignite/train_dynamics.py
```

### Resolutions

**1. `src/tokamak_foundation_model/ignite/train_dynamics.py` — 2 hunks, both UNION (kept both sides).**

Purely additive parameter conflicts, no judgement call. `train()`'s signature keeps the recommender's `ctf_*`, `modality_loss_weight`, `actuator_dropout_p`, `sf_*`, `text_*` block and gains Peter's `dropout: float = 0.0`, with `log=print` kept last; the matching `main()` call site gains `dropout=args.dropout` alongside the recommender's kwargs. Peter's body (`if dropout: cfg.dropout = ...`, line ~903) and his `--dropout` argparse entry (line ~1332) auto-merged and are untouched.

Worth knowing: the recommender already had a *separate* `actuator_dropout_p` knob. The two coexist and are distinct — `dropout` is attention+FFN dropout on `DynamicsConfig`, `actuator_dropout_p` is per-sample actuator dropout for CFG at eval. Peter's commit subject says "dropout" for both, but his actual diff only adds the former.

**2. `src/tokamak_foundation_model/ignite/maskgit.py` — 1 hunk, RECOMMENDER STRUCTURE + Peter's feature grafted on.**

This is the only resolution where I wrote code rather than choosing a side.

Peter's side is the *old* single-pass `generate_frame` loop with his annealed-Gumbel reveal inlined. The recommender side had since refactored that same loop into three passes (sample all / decide reveals / commit) with pluggable reveal policies (`_per_modality_reveal`, `_global_reveal`), `SamplerConfig` (`temp_for`, `top_p`, `global_pool`), and draft-and-revise revision rounds. Taking Peter's side verbatim would have reverted all of that; taking the recommender's side verbatim would have dropped his feature.

Kept the recommender's three-pass structure and moved the Gumbel perturbation into PASS 1, where the confidence used for ranking is produced. **Deliberate deviation from Peter's literal code:** Peter perturbs in log space (`conf = log c + ann·g`), but the recommender's `_global_reveal` scores with `norm_log_confidence(c, V) = 1 + log c / log V`, which takes `log` of `c` and returns NaN on a negative input. So the perturbed score is mapped back through `exp`:

```python
c = (c.clamp_min(1e-12).log() + ann * g).clamp_(max=80.0).exp()
```

`exp` is monotone, so this is *exactly* Peter's ranking under both reveal policies: `_per_modality_reveal` ranks on `c` directly, and `_global_reveal` recovers `1 + (log c + ann·g)/log V`. The `clamp_(max=80.0)` keeps the exponent inside fp32 for a large `IGNITE_MASKGIT_GUMBEL` (it only binds above scale ≈ 4.2, given `g ≤ 20.7` from the `1e-9` clamp on `u`). The `torch.rand` draw stays behind `if _GUMBEL_SCALE > 0.0`, so the default path draws nothing extra and the RNG stream stays bit-identical — Peter's stated invariant. Gumbel is *not* applied to the revision rounds' confidences: those score the already-committed code, and Peter's side has no revision rounds to compare against.

Peter's module-level `import os as _os` and `_GUMBEL_SCALE` (line 41) auto-merged and are used unchanged. `MaskGITDynamics.__init__(self, cfg)` is untouched by both sides, so the brief's constructor check is trivially satisfied — all 40 call sites across `src`, `tests` and `scripts` pass `MaskGITDynamics(cfg)` and nothing else:

```
$ grep -rn "MaskGITDynamics(" src tests scripts   # 40 sites, all single-arg
src/tokamak_foundation_model/ignite/maskgit.py:44:class MaskGITDynamics(nn.Module):
src/tokamak_foundation_model/ignite/maskgit.py:45:    def __init__(self, cfg: DynamicsConfig):
```

**3. `src/tokamak_foundation_model/ignite/eval_dynamics.py` — 2 hunks, MIXED.**

- Hunk 1: kept the **recommender's** SCALE GUARD block (the `_sg`/`_sp`/`_mg`/`_mp` statistics, `gt_static`, `mismatched`, the z-scoring). It is load-bearing — `mismatched` and `gt_static` are both read further down in code that survived the automerge, so dropping it would `NameError`. Removed only the now-dead `diff = img_p - img_g` line, since Peter deleted the difference panel that consumed it.
- Hunk 2: kept **Peter's** `for a in (ax_g, ax_p):` — the honest video panel. The recommender's side here (`im_d = ax_d.imshow(diff, ..., vmin=-dmax, ...)` plus the `max |diff|` annotation) was already broken by the automerge: Peter's cleanly-merging hunks had removed the `dmax = ...`, `ax_d = fig.add_subplot(inner[3])` and `cax_d` definitions and narrowed the grid from `1, 5` to `1, 3`. Keeping the recommender side would have referenced three undefined names.

Net result: GT | Prediction only (Peter), still z-scored with the `(z-scored)` ylabel stamp when the two sides are in different spaces (recommender). Both sides' intent preserved.

Peter's remaining changes all auto-merged and were spot-checked as present: `scripts/slurm_frontier/eval_dynamics.sh:42` `--actuator_mode`, `scripts/slurm_frontier/train_dynamics.sh:130` `--dropout`, and the section gap `0.46` at `eval_dynamics.py:681`/`:689`.

### Verification (real output)

```
$ PYTHONPATH=src .pixi/envs/frontier/bin/python -m pytest tests/ignite -x -q 2>&1 | tail -3
-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
517 passed, 5 skipped, 3 warnings in 246.80s (0:04:06)

$ PYTHONPATH=src .pixi/envs/frontier/bin/python -c "from tokamak_foundation_model.ignite import eval_dynamics, maskgit; print('ok')"
ok
```

Same 517/5 set as A1 and as the `nathan_fm` baseline.

`load_model` contract preserved for Task C3 — `eval_dynamics.py:121`:

```python
def load_model(ckpt_path: Path, device) -> Tuple[MaskGITDynamics, DynamicsConfig, int]:
    ...
    return model, cfg, step
```

**Extra check, not in the brief.** The suite only exercises the default `_GUMBEL_SCALE = 0` path, so my resolution's new branch would otherwise ship untested:

```
$ PYTHONPATH=src IGNITE_MASKGIT_GUMBEL=1 pytest tests/ignite/test_phaseb_maskgit.py tests/ignite/test_sampling.py -q
23 passed in 3.60s

$ # direct probe, both reveal policies, scales 0 / 1 / 4
GUMBEL=0.0 global_pool=False valid_codes=True n_unique=17
GUMBEL=0.0 global_pool=True  valid_codes=True n_unique=15
GUMBEL=1.0 global_pool=False valid_codes=True n_unique=19
GUMBEL=1.0 global_pool=True  valid_codes=True n_unique=20
GUMBEL=4.0 global_pool=False valid_codes=True n_unique=22
GUMBEL=4.0 global_pool=True  valid_codes=True n_unique=20
global_pool=False: greedy vs gumbel identical? False
global_pool=True:  greedy vs gumbel identical? False
```

i.e. no NaN through `norm_log_confidence`, every code finite and in range, and the knob demonstrably changes the reveal order rather than being a silent no-op.

Line lengths: my added lines are all ≤ 88 (`ruff` `line-length = 88`). The one 89-char line in the diff is Peter's own `_GUMBEL_SCALE` comment, carried over verbatim. Note `ruff`'s default rule set does not select `E501`, and this repo does not override `select`, so long lines are not actually enforced — the file already has many.

---

## Task A3 — move Claude material out of `docs/`, delete `origin/recommender`

### Blocker found and fixed first: `.gitignore` ignored all of `.claude/`

`.gitignore:8` was `.claude/`, so ruling 2's `git add .claude/superpowers` would have silently added nothing and the whole A3 move would have produced an untracked tree. Since git cannot re-include a path once its parent *directory* is excluded, the rule had to become the `.claude/*` form:

```
.claude/*
!.claude/notes/
!.claude/superpowers/
!.claude/superpowers-runtime/
```

`.claude/settings.local.json` and any other loose entry stay ignored. Verified:

```
.claude/settings.local.json            IGNORED
.claude/superpowers/specs/x.md         tracked-able
.claude/notes/x.md                     tracked-able
.claude/superpowers-runtime/sdd/x.md   tracked-able
.claude/other.json                     IGNORED
```

### Files moved — 68 renames, 100% similarity, all via `git mv`

**A. `docs/superpowers/plans/` → `.claude/superpowers/plans-recommender/` (8)** — directory `git mv`, per the brief, to avoid colliding with the existing untracked `.claude/superpowers/plans/`:
`2026-08-17-ignite-rollout-quality-phase1.md`, `2026-09-03-labelmaker-phase1.md`, `2026-09-05-labelmaker-phase3.md`, `2026-09-07-recommender-ledger.md`, `2026-09-07-recommender-plan.md`, `2026-09-17-events-shot-rosters-and-verification.md`, `2026-09-18-shot-design-ignite-export.md`, `2026-09-18-verification-ui.md`

**B. `docs/superpowers/specs/` → `.claude/superpowers/specs/` (13)**, merged into the existing directory:
`2026-05-11-e2e-stage1-file-open-profile-design.md`, `2026-09-03-labelmaker-design.md`, `2026-09-05-labelmaker-phase2-design.md`, `2026-09-05-labelmaker-phase3-design.md`, `2026-09-07-recommender-critique.md`, `2026-09-07-recommender-ideate.md`, `2026-09-07-recommender-labelmaker-v2.md`, `2026-09-13-labels-actuation-probe.md`, `2026-09-13-labels-assessment-A.md`, `2026-09-13-labels-workstream.md`, `2026-09-17-events-shot-rosters-and-verification-design.md`, `2026-09-18-shot-design-ignite-export-design.md`, `2026-09-18-verification-ui-design.md`

`docs/superpowers/` was then empty (0 files left, tracked or untracked) and was removed.

**C. `.superpowers/sdd/*` → `.claude/superpowers-runtime/sdd/*` (39)** — per **ruling 1**, file-by-file from `git ls-files .superpowers`, never at directory level:
`profile_l14perf.py`, `review-C3fix.md`, `review-I10.md`, `review-Iprev.md`, `review-L14perf.md`, `task-C3fix-brief.md`, `task-C3fix-report.md`, `task-I10-brief.md`, `task-I10-report.md`, `task-I10fix-brief.md`, `task-I10fix-report.md`, `task-I2-report.md`, `task-Ifix-C1-brief.md`, `task-Iprev-brief.md`, `task-Iprev-report.md`, `task-L12-report.md`, `task-L14perf-brief.md`, `task-L14perf-fix-brief.md`, `task-L14perf-report.md`, `task-LA-report.md`, `task-LC2-report.md`, `task-LD1b-report.md`, `task-LD2-report.md`, `task-LLM1-brief.md`, `task-LLM1-report.md`, `task-LLM2-brief.md`, `task-LLM2-code-report.md`, `task-LLM2-report.md`, `task-LLM2-review.diff`, `task-R1-rename-brief.md`, `task-R1-rename-report.md`, `task-UI1-brief.md`, `task-UI1-report.md`, `task-UI1b-brief.md`, `task-UI2-brief.md`, `task-UI2-report.md`, `task-UI3-brief.md`, `task-UI3-report.md`, `task-fix-C1-report.md`

**D. Working notes → `.claude/notes/` (8)**: `docs/phase_c_step1_status.md`, `docs/spectro_video_status.md`, `docs/spectrogram_step0_findings.md`, `docs/eval_stage1_panels_patch.md`, `docs/eval_stage1_plan.md`, `docs/eval_improvement_recommendations.md`, `docs/shot-design-harness.md`, `fsq_e2e_wiring_scope.md`

**Left in place, untracked, exactly as ruling 1 requires** — the three live `.superpowers/sdd/` workspace directories: `2026-09-19-recommender-frontier-port/` (this task), `2026-08-17-ignite-rollout-quality-phase1/`, `having-latent-text-encodings-enumerated-swan/`. The `.git/info/exclude` entry for `.superpowers/` was left alone.

**Added, not moved** (ruling 2): `.claude/superpowers/specs/2026-09-19-recommender-frontier-port-design.md`, `.claude/superpowers/plans/2026-09-19-recommender-frontier-port.md`.

`docs/` now holds exactly the 13 keep-list files the brief names, and nothing else.

### Links fixed (file:line, as they were before the edit)

**`docs/superpowers/` → `.claude/superpowers/` (with `plans/` → `plans-recommender/`):**
- `docs/LABELER.md:15`, `:16`, `:844`, `:865`
- `outputs/labeler/presentation_continued/README.md:85`
- `scripts/labeler/ae_dataset.py:125` (the path wraps across `:125`–`:126`)
- `src/labeler/__init__.py:5`
- `src/labeler/events/__init__.py:7`
- `src/labeler/models/README.md:50`
- `src/labeler/models/d3d_tearing_time_to_event_dsm/README.md:359`

**`.superpowers/sdd/` → `.claude/superpowers-runtime/sdd/`:**
- `configs/shot_design/ignite_modalities.yaml:93`
- `docs/LABELER.md:117`, `:595` (relative markdown links: `../.superpowers/...` → `../.claude/superpowers-runtime/...`)
- `scripts/labeler/tokeye_masks.sbatch:21`
- `scripts/shot_design/build.sbatch:21`
- `tests/shot_design/test_phenomena.py:692` (comment; re-wrapped to stay inside 88 cols)

Three of those targets (`sdd/progress.md`, `task-I6b-report.md`, `task-I9a-review.md`) do not exist in the repo at all — they were already dangling before this work. The prefix was still rewritten so they at least point at the right directory.

**References to the moved notes — beyond the brief's stated scope, see Deviation 2:**
- `docs/spectrogram_tokenizer_plan.md:209` → `.claude/notes/spectrogram_step0_findings.md`
- `inspect_spectrograms/step0_inspect.py:16` (docstring) and `:45`–`:46`, `:298` (`DOCS_DIR`/`SUMMARY_PATH` → `NOTES_DIR`, so the script writes where its docstring says)
- `scripts/training/eval_e2e_phase1.py:4` → `.claude/notes/eval_stage1_plan.md`
- `scripts/training/eval_e2e_phase2_plots.py:4`, `:79` → `.claude/notes/eval_stage1_plan.md`
- `tests/e2e/test_video_integration.py:5` → `.claude/notes/phase_c_step1_status.md`

**One content fix:** `scripts/labeler/ae_dataset.py:126` named `2026-09-05-labeler-phase3.md`, which has never existed — the file is `2026-09-05-labelmaker-phase3.md`. Corrected while rewriting the prefix, so the reference now resolves.

**Deliberately NOT rewritten:** cross-references *inside* the moved files (the 39 sdd reports and the 21 specs/plans). They are historical review and planning records that quote the exact commands and paths used at the time; rewriting them would falsify that history. The brief scopes step 2 to `-- ':!.claude'`, which excludes them by construction.

### Verification (real output)

```
$ git grep -n "\.superpowers/\|docs/superpowers" -- ':!.claude' && echo "STALE LINKS" || echo "clean"
clean

$ git grep -nE "docs/(phase_c_step1_status|spectro_video_status|spectrogram_step0_findings|eval_stage1_panels_patch|eval_stage1_plan|eval_improvement_recommendations|shot-design-harness)\.md" -- ':!.claude' && echo "STALE" || echo "clean"
clean

$ git ls-files .superpowers | wc -l
0

$ ls .superpowers/sdd/2026-09-19-recommender-frontier-port/progress.md
.superpowers/sdd/2026-09-19-recommender-frontier-port/progress.md

$ git branch -r | grep -c recommender
0
```

Both ruling-1 post-conditions hold.

```
$ PYTHONPATH=src .pixi/envs/frontier/bin/python -m pytest tests/shot_design/test_phenomena.py -q 2>&1 | tail -2
27 passed, 58 errors in 10.05s
```

The 58 errors are **environmental and pre-existing**, not caused by this work — every one is the same fixture-setup `ImportError: Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'`, and neither package exists in the `frontier` env:

```
$ .pixi/envs/frontier/bin/python -c "import pyarrow"      -> ModuleNotFoundError: No module named 'pyarrow'
$ .pixi/envs/frontier/bin/python -c "import fastparquet"  -> ModuleNotFoundError: No module named 'fastparquet'
```

This is exactly the `ideate-frontier` env gap that Task B2 fills. My entire change to that test file is one comment line. The test the brief actually cares about — the one asserting a sentence appears in `docs/SHOT_DESIGN.md`, which did not move — passes:

```
$ pytest "tests/shot_design/test_phenomena.py::test_the_config_and_the_docs_state_the_ranking_the_code_implements" -q
1 passed in 0.28s
```

Syntax-checked every edited non-Python file too: `py_compile` OK on all 8 edited `.py` files, `yaml.safe_load` OK on `ignite_modalities.yaml`, `bash -n` OK on both `.sbatch` files.

### Remote branch deletion

Guarded by the identity check ruling 5 requires, plus a containment check:

```
$ git rev-parse HEAD              -> e968501f6bdeb2b026fa2ca5f5359206aed3442f
$ git rev-parse origin/nathan_dev -> e968501f6bdeb2b026fa2ca5f5359206aed3442f
MATCH -> safe to delete

$ git merge-base --is-ancestor origin/recommender HEAD
YES - every recommender commit is reachable from nathan_dev

$ git push origin --delete recommender
To github.com:PlasmaControl/FusionAIHub.git
 - [deleted]         recommender

$ git branch -r | grep -c recommender
0
```

---

## Late addition — `25667aa`

After A3 was committed and pushed, the coordinator reported editing the two 2026-09-19 plan/spec files. Re-checked: the spec was already committed at its current content (empty diff); the plan had one later edit (a `Delete:` bullet about `scripts/shot_design/serve_llm.sbatch` corrected to a `Note:`). Since `e968501` was already pushed and the global constraints forbid force-pushing, the current content landed as a follow-up commit rather than an amend. Both files are now tracked at their current content.

---

## Final self-review

```
$ git diff a978f0e..HEAD --stat | tail -1
140 files changed, 13676 insertions(+), 58 deletions(-)
```

(`a978f0e` is the old `origin/recommender` tip, kept because the ref itself is now deleted.)

With renames collapsed, only 24 files carry real modifications; the rest are the 47 carried evaluation scripts, `_frontier_common.sh`, the 2 new `.claude` plan/spec files, and the 68 renames. The 58 deletions are the conflict resolutions plus the removed difference-panel code. Nothing outside the three tasks' scope is touched.

Final state re-verified at `HEAD`:

```
$ git status --short --untracked-files=all   -> (empty)
$ PYTHONPATH=src pytest tests/ignite -x -q | tail -1
517 passed, 5 skipped, 3 warnings in 250.80s (0:04:10)
```

---

## Deviations from the briefs

1. **`.gitignore` was modified — not mentioned in any brief, and mandatory.** `.claude/` was fully ignored, so ruling 2's `git add .claude/superpowers` would have been a silent no-op and every future file under `.claude/` would have stayed invisible to `git status`. Detail above. This is the deviation most worth a second opinion.
2. **Fixed references to the moved `.claude/notes/` files as well.** The brief's *Files* section says "every file that links to a moved path", but its step-2 command only greps `docs/superpowers` and its step-3 check only greps `\.superpowers/|docs/superpowers`. Six files referenced the moved notes and would have been left dangling. That included `inspect_spectrograms/step0_inspect.py`'s `SUMMARY_PATH`, which *writes* the findings doc — left pointing at `docs/` it would have recreated a moved file there on any re-run. That path is a hard-coded absolute into another user's Stellar checkout (`/scratch/gpfs/ps9551/...`), already dead on Frontier; only the trailing `docs` → `.claude/notes` segment changed.
3. **A2 had three conflicts, not the two the brief lists**, the third (`train_dynamics.py`) being a direct consequence of A1 carrying `_sync_grads` onto the same file. Resolved by union; no judgement call.
4. **A fourth commit `25667aa`** exists beyond the three the briefs specify, for the reason given above.

## Concerns

1. **The `maskgit.py` resolution is the one place where I wrote new code rather than choosing a side, and it deserves the closest review.** The `log → +Gumbel → exp` round-trip is mathematically the same ranking as Peter's log-space version under both reveal policies, and I verified it empirically, but it is my construction, not his. The alternative, if a reviewer prefers it, is to teach `_global_reveal` to accept a log-domain score and skip the `exp` — cleaner, but a larger change to code neither branch touched.
2. **`clamp_(max=80.0)` silently ties reveal scores above `e^80`.** Unreachable for `IGNITE_MASKGIT_GUMBEL ≲ 4` and the knob's documented value is 1, but it is a real (if remote) behaviour ceiling that Peter's pure-log version does not have.
3. **Gumbel is off by default and therefore untested by CI.** The suite pins the `_GUMBEL_SCALE = 0` path only; my extra runs above are the only coverage that branch has. If the knob is going to be used in anger it wants a real test — out of scope here.
4. **`docs/superpowers` links inside the moved specs/plans are now stale relative to the new layout.** Left alone on purpose (history), but anyone reading `.claude/superpowers/plans-recommender/*` will find paths that no longer resolve. Worth a one-off sweep later if those documents are living rather than archival.
5. **`plans/` vs `plans-recommender/` is an odd split** — the 8 historical plans and the 1 current plan sit in sibling directories for no reason other than avoiding a name collision at move time. The brief mandated it; consolidating is a trivial follow-up if wanted.
6. **`tests/shot_design` cannot actually be run until Task B2.** 58 of 85 tests in the one file the brief names error out on missing `pyarrow`/`fastparquet`. Pre-existing and expected, but it means A3's link edit to that file is only shallowly verified (one targeted test plus `py_compile`).
7. **`CMAP_DIFF` at `eval_dynamics.py:79` is now unused** — the difference panel it served is gone. Harmless (module-level, not flagged by ruff's default rule set) and it arrived that way from Peter's side, so I left it rather than widen the diff.
