# Review — task I-prev, branch `recommender-Iprev` (3cced76..0a4eb9a)

**VERDICT: MERGE WITH FIXES: (1) record the untimed residual/total next to `phase_seconds`; (2) `config.data_root_origin()` mislabels the default when `IDEATE_CONFIG_DIR` is set; (3) `os.replace` of the staged text subset is cross-device-fragile.** None blocks the merge — the guard, the announcement and the docs all do what the brief asked, and all 30 new tests pass under `-W error` with ruff clean.

---

## Findings

### 1. `phase_seconds` is not exhaustive and the manifest has no total to expose the gap — MEDIUM
`src/ideate/shotdb/build.py:1088` (`cov = coverage_report(records)`), the manifest-dict construction at `:1089-1136`, and the whole `if encode:` block at `:1144-1150` (plus the ignite table writes it triggers at `:1210-1220`) fall inside no `_phase(...)` block. On an **encoding** build — which is what the three 65–68 % SLURM builds were — the single largest slice of wall time is outside every phase. Because the manifest carries no `elapsed_s` (check the key list at `:1089-1136`: there is none; total elapsed lives only on the in-process `BuildReport`), a person reading `manifest.json` after the job cannot even see that a residual exists, let alone size it. That defeats part of the diagnostic purpose the brief states.
Report deviation 3 (excluding the IGNITE encode from `write_tables`) is correct and well argued — the problem is that the exclusion is invisible from the artefact.
**Fix:** at the second manifest write (`:1155`), also set `manifest["build_elapsed_s"] = time.perf_counter() - t_start` (a sibling key, so `test_build_phases.py:38`'s "keys are exactly `PHASES`" assertion still holds). Optionally assert in that test that `sum(phase_seconds.values()) <= build_elapsed_s`.

### 2. `data_root_origin()` returns a wrong filename when `IDEATE_CONFIG_DIR` is set — MEDIUM-LOW
`src/ideate/config.py:108` hardcodes `"configs/ideate/paths.yaml default"`, but `CONFIG_DIR` is env-overridable (`src/ideate/config.py:23`) and `load_paths` reads `CONFIG_DIR / "paths.yaml"` (`:87`). With `IDEATE_CONFIG_DIR` exported, the line names a file that did not settle anything. The whole point of the feature is that the resolved path alone "looks plausible either way" — a mislabelled *source* is the same failure one level up.
**Fix:** `return f"{CONFIG_DIR / 'paths.yaml'} default"`. Note this also touches `docs/IDEATE.md:69` and `tests/ideate/test_docs_scratch_db.py:50-53`, which compare the docs prose against the literal return value — good, the drift test will force both.
Related, lower: `data_root_origin()` re-reads the environment instead of being derived from the `paths` object handed to `_announce_root` (`src/ideate/cli.py:326-330`), so under a `using_paths` override the root and the origin could disagree. The docstring (`config.py:104-105`) acknowledges that no writing command runs under one; true today.

### 3. `manifest.json.part` is orphan-prone and can fail a build that already succeeded — LOW
`src/ideate/shotdb/build.py:1156-1158` writes `db/manifest.json.part` then `os.replace`s it. `manifest.json.part` is in neither `BUILD_FILES` (`:903-910`) nor `BUILD_GLOBS` (`:913`), so `_build_owned` (`:916-921`) never claims it: a crash between the two lines leaves it in the published `db/` forever, and no later `_publish` prunes it. Separately, if `part.write_text` raises (ENOSPC), `build()` raises *after* a fully successful publish and `cli.main` (`src/ideate/cli.py:1886-1888`) reports exit 1 for a database that is in fact correct.
**Fix:** add `"manifest.json.part"` to `BUILD_GLOBS`/`BUILD_FILES` so it is pruned, and wrap `:1156-1158` so a timing-only rewrite failure logs a warning instead of failing the build.

### 4. Crash between the two manifest writes understates timing only — LOW (answer to question 3, no fix required)
Both writes serialize the **same** `manifest` dict, so every content field — `shot_source`, `n_shots`, `n_built`, `shots`, `forced_over` (set at `:1138`, before the first write at `:1139-1143`), `ignite` (`:1145-1150`) — is byte-identical between them. The only difference is `phase_seconds`, which the first write emits with `write_tables: 0.0, publish: 0.0` because `_phase` mutates the live dict *after* `_write_tables` serialized it. So a crash in the window leaves a published `db/` whose manifest describes the database correctly (the guard on the next build still works) and its timing incompletely. The rename itself is atomic: `part` and `manifest.json` are in the same directory, hence the same filesystem, and `os.replace` is atomic there. A `labels` block from a later `labels join`, or `adds_since_fit` from a later `add`, cannot be lost — both writes happen inside one `build()` call, seconds apart, and `build` already replaces the manifest wholesale as it always did.
**Fix (optional):** one sentence in the `phase_seconds` comment (`:1131-1135`) saying which two keys are zero if the rewrite never landed.

### 5. New cross-device failure mode in the staged text subset — LOW
`src/ideate/shotdb/build.py:1062` creates the staging dir under `paths.db_dir.parent`, and `:1072` does `os.replace(subset, old_subset)` onto `paths.text_cache_dir`. With the stock `configs/ideate/paths.yaml` both are `${data_root}/…` so this is safe, but an `IDEATE_PATHS` file that puts `text_cache_dir` on a different filesystem — exactly the scratch-DB workflow `docs/IDEATE.md:56-61` now recommends — turns it into `OSError: Invalid cross-device link`. This failure did not exist before the change (`text.build_logs_subset` wrote in place).
**Fix:** `shutil.move(subset, old_subset)`, or create the `TemporaryDirectory` under `paths.text_cache_dir` instead.

### 6. The staging directory leaks into the data root on SIGKILL — LOW
`src/ideate/shotdb/build.py:1062` names it `.build-text-*` inside the data root. `TemporaryDirectory` cleans up on exception and on normal exit, but not on SIGKILL — a SLURM timeout or OOM, precisely the population of builds this task is about — leaving a full copy of `logs_subset.jsonl` in production. Cosmetic, but it accumulates and nothing prunes it.

### 7. Two writing commands still announce nothing — LOW / out of the brief's scope
`ideate model --download` writes `paths.models_dir` (`src/ideate/cli.py:397-412`) and `ideate logs import` writes `paths.shotsummary_raw_dir` / `per_shot_txt_dir` (`src/ideate/cli.py:1178-1190`). Both are outside the brief's six. Risk is lower than for the six because in the stock paths file neither destination is under `${data_root}`, so `IDEATE_DATA_ROOT` does not steer them — but they are the remaining unlabelled writers if the announcement is meant to become a rule.

### 8. Two brittle-ish test assertions — LOW
- `tests/ideate/test_build_phases.py:45` asserts `phases["publish"] > 0.0` against a value rounded to 6 dp. Publishing seven small files on tmpfs is ~10⁻³ s, so it is safe today, but it is a clock assertion and can in principle flake.
- `tests/ideate/test_publish_guard.py:43-45` asserts the bare substring `"2"` is in the refusal message; almost any message containing a `tmp_path` would satisfy it. Weak, not wrong — prefer `"n_shots=2"`.
- `tests/ideate/test_publish_guard.py:24-28` (`snapshot`) walks `is_file()` only, so "no file touched" is proven but directory creation is not. In fact `build()` does create `db_dir.parent` (`build.py:1061`) and the `.build-text-*` dir before the *second* check; both are pre-existing or removed on the refusal path, so the claim holds in substance.

### 9. `_check_publish` has no return annotation — TRIVIAL
`src/ideate/shotdb/build.py:995`. Self-reported by the implementer; ruff does not flag it. Return type is `dict | None`.

---

## Answers

**1. Guard semantics (`src/ideate/shotdb/build.py:995-1027`).**
Impossible to publish smaller or different-source without `--force`, for every case the guard can see. It is called twice: `:1056` on the requested count before anything is read, and `:1069` on `len(records)` — the count actually built — so per-shot failures cannot smuggle a shrink through (pinned by `test_publish_guard.py:59-76`). The refusal condition at `:1019-1021` is `problem or shot_source mismatch or n_shots < previous`; `problem` is evaluated first and short-circuits, so a corrupt manifest never reaches a comparison against a `None` count. `problem` is set for: non-dict JSON, missing `shot_source`, a non-`str`/non-`None` `shot_source`, `type(n_shots) is not int` (this also rejects `bool` and a string `"2"`), a negative `n_shots`, and any `OSError`/`ValueError` while reading (`:1010-1015`). Only `FileNotFoundError` returns `None` (`:1013`) — i.e. **a `db/` with tables but no `manifest.json`, such as one interrupted mid-publish, is still replaceable without `--force`.** That is the brief's spec, not a defect, but it is the one remaining hole.
Same-source growth still publishes: `:1067-1069`/`:1019-1021` compare counts only, and `test_publish_guard.py:47-56` runs 2→2 and 2→3 and asserts no `forced_over`.
`add` and `labels join` never call it: `_check_publish` has exactly two call sites, both in `build()`; `add()` (`build.py:1278`) does not call `build()`; `cmd_labels` (`cli.py:1237-1241`) calls `refresh_frame_codes` + `join.write_tables`. `test_publish_guard.py:118-119` closes it at the CLI level.
Nothing is written to the target before either refusal. The first check is the second statement of `build()` (only `sorted(set(shots))` precedes). The second is reached after the text subset has been built into a temp dir and **before** `os.replace(subset, old_subset)` at `:1072`, so the text cache is intact too; the only side effects are `db_dir.parent.mkdir(exist_ok=True)` and a temp dir the context manager removes on the exception path.
The message (`:1022-1027`) names the db path, both sources, both counts, the parse problem when there is one, and `--force`. Actionable. It does not mention that `--force` records `forced_over`, nor suggest "point `IDEATE_DATA_ROOT` somewhere else" — acceptable, `docs/IDEATE.md:81-89` covers both.

**2. Root announcement.**
Emitted on all six: `build` `cli.py:343`, `add` `:439`, `encode` `:513`, `corpus select` `:925`, `corpus scan` `:1111`, `labels join` `:1225`. Each precedes that command's first write — verified by reading the bodies (e.g. `encode`'s `run_dir.mkdir` is at `:518`, `corpus scan`'s `census.write` at `:1125`, `corpus select`'s first `write_text` at `:1003`, `labels join`'s `refresh_frame_codes` at `:1237`) and by `test_write_root.py:66-89`, which hooks `os.mkdir` and captures what stderr held at the first directory created under the root.
On stderr: `cli.py:324` and `:326-330` both pass `file=sys.stderr`; `--json` stdout consumers are unaffected.
Origin labels correct for the three documented cases (`config.py:105-108`) and they mirror `load_paths`'s precedence (`config.py:84-89`) exactly — with the `IDEATE_CONFIG_DIR` caveat of finding 2. Unit-pinned at `test_write_root.py:27-38`.
Missed writers: see finding 7. Nothing in the brief's scope was missed.

**3. Phase timing.** Six phases, non-overlapping (all `with _phase(...)` blocks are sequential; `scalar_embedding` accumulates across two disjoint blocks, `:1075` and `:962`). **Not exhaustive** — see finding 1. Rename atomic — see finding 4. Second write preserves everything the first wrote — yes, same dict object. A crash between them cannot leave a manifest that misdescribes the *database*, only its timing.

**4. Docs (`docs/IDEATE.md:41-84`).** Accurate against `pyproject.toml:284-289` (`IDEATE_DATA_ROOT`, `LABELMAKER_ROOT`, `IDEATE_CORPUS` all pinned under `[tool.pixi.feature.ideate.target.unix.activation.env]`) and against `pyproject.toml:344-345` (`ideate = ["ideate","cuda"]`, `ideate-cpu = ["ideate","ideate-cpu"]` — so the claim that both environments pin them is right). The example root line at `:72` matches `_announce_root`'s f-string. The guard paragraph at `:81-89` matches `_check_publish` clause for clause, including the corrupt-manifest case.
`tests/ideate/test_docs_scratch_db.py` is **half real drift detection, half prose**: `:31-40` reads the activation-env table out of `pyproject.toml` and fails if a pinned variable disappears from either the toml or the prose; `:48-55` compares the prose against what `config.data_root_origin()` actually returns under two env states. Those two genuinely fail on drift. `:43-46` and `:58-61` are substring checks (`".pixi/envs/ideate-cpu/bin/python -m ideate"`, `"IDEATE_PATHS="`, `"--force"`, `"shot_source"`, `"n_shots"`) — they catch deletion of the words, not a behaviour change. Also, the absolute path in the docs example (`docs/IDEATE.md:53`) is this user's worktree; only its suffix is pinned.

**5. Tests.** Hermetic. `tests/ideate/conftest.py:35-65` builds a tmp `paths.yaml`, sets `IDEATE_PATHS` and — crucially — `monkeypatch.delenv("IDEATE_DATA_ROOT", raising=False)` at `:64`, which neutralises the pixi activation pin; the suite passing under `pixi run -e ideate-cpu` is itself the proof that nothing reached the production root. `embed_texts` is monkeypatched in all three build-driving fixtures (`test_publish_guard.py:14-16`, `test_write_root.py:47-49`, `test_build_phases.py:22-24`), so no model download. `test_docs_scratch_db.py` reads only `pyproject.toml` and `docs/IDEATE.md` from the repo. The only network-ish call is the pre-existing `_blurb_client()` → `LLMClient().available()` localhost probe (`build.py:783-792`) that every build test already makes. All pass under `-W error`.
Behaviour vs implementation: `test_publish_guard.py` and `test_build_phases.py` test behaviour. `test_write_root.py:52-72` (`watch_first_write`) is implementation-coupled — it monkeypatches `os.mkdir` globally and depends on the invariant "every writer in scope reaches its output through `mkdir(parents=True, exist_ok=True)`" — but it is the only way to pin *ordering*, it documents the invariant, and it fails loudly (`seen == []`) rather than silently if the invariant breaks. One hole: it consumes captured stderr at the first `mkdir`, so the trailing `assert "data root" not in capsys.readouterr().err` only covers output emitted after that point. Brittleness: finding 8.

**6. Report honesty.** No claim contradicted by the diff. Verified independently: 13 + 10 + 3 + 4 = 30 tests, matching my run exactly; `_check_publish` called twice; `cmd_labels` never calls `build()`; `--force` writes `forced_over` (`build.py:1137-1138`); the two self-reported observations (no return annotation; three-null `forced_over` over a corrupt manifest) are real and correctly characterised as not contradicting the brief. The full-suite totals (ideate 1206, labelmaker 1580/2) I did **not** re-run — out of scope per the review brief. Deviations 1–5 are all justified: (1) attributing 7e05cd8 to the previous implementer matches the commit's own body; (2) the `config.CONFIG_DIR` monkeypatch it avoided would indeed have redirected six other readers, and the `== [expected]` whole-stderr assertion is wrong for `corpus select`, which prints "no logbook at …" first; (3) excluding the IGNITE encode is right but leaves finding 1 unaddressed; (4) accurate (see finding 4); (5) the `Claude Fable 5.1` trailer on all five commits is the brief's literal instruction applied by a session running on Opus 5 — the commits are now misattributed, flagged for the ledger, not a code issue.
Two things the report omits: the branch has **five** commits, not four (it says "Four commits" then lists four, and `0a4eb9a` adds the report itself); and `0a4eb9a` also newly tracks `.superpowers/sdd/task-Iprev-brief.md`, which the report does not mention. `.superpowers/sdd/` is *not* gitignored in this repo (`git check-ignore` exit 1; ten sibling briefs/reports are already tracked), so this is consistent with prior tasks — but the controller should know this review file will show up as untracked.

**7. Scope.** `git diff --name-only 3cced76..recommender-Iprev | grep -c 'docs/superpowers/plans'` → **0**. Nothing under `docs/superpowers/plans/**` changed. The diff touches exactly 10 files / +790 −19: `docs/IDEATE.md`, `src/ideate/cli.py`, `src/ideate/config.py`, `src/ideate/shotdb/build.py`, the four test modules, plus `.superpowers/sdd/task-Iprev-brief.md` and `-report.md`. The report accounts for eight of the ten (the brief file is the one unmentioned addition). Working tree clean (`git status --porcelain` empty before this review file was written).

---

## Commands run

```
$ cd /scratch/gpfs/nc1514/FusionAIHub-Iprev
$ export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src HF_HUB_OFFLINE=1
$ pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e ideate-cpu \
    python -m pytest tests/ideate/test_publish_guard.py tests/ideate/test_write_root.py \
    tests/ideate/test_build_phases.py tests/ideate/test_docs_scratch_db.py -q -W error -p no:cacheprovider
..............................                                           [100%]
30 passed in 11.00s
exit 0

$ pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker \
    ruff check src/ideate tests/ideate
All checks passed!
exit 0
```

Read-only throughout: no `ideate` writing command was run, `/scratch/gpfs/EKOLEMEN/` was not touched, nothing was edited, staged, committed, checked out or merged, and the only file written is this one.
