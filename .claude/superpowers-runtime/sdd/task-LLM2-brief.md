# Task LLM2 — blurb prompt v6, gate hardening, targeted regeneration

Worktree: `/scratch/gpfs/nc1514/FusionAIHub-LLM2` (branch `recommender-LLM2`, from `recommender` @ 9cec3d6).
Work ONLY in this worktree. Package `shot_design` (`src/shot_design`), tests `tests/shot_design`.

## Why

The first production backfill (prompt v5, Gemma 4 26b, 504 shots, all gate-accepted) was audited
against each blurb's `blurb.source_text`. The gate (numbers, shot numbers, quotes, three
sentences, word cap) let two classes of error through:

1. **Wrong abbreviation expansions** — the model "explained" an abbreviation it did not know:
   `AE` → "aeroelastic instabilities" (200729; source: "AE-induced zonal flows"),
   `EHO` → "edge helical oscillation" (190511), `QH` → "quasi-high confinement mode" / "quasi-H-mode"
   (189867, 189869, 189890), `FPP` (Fusion Pilot Plant) → "flat pressure profile" / "flat plasma
   profile" / "fully populated plasma" (203872, 203876, 204163, 204170, 204173), `PCS` (Plasma
   Control System) → "power control system" (185786, 189472, 199516), `betan` → "poloidal beta"
   (203462), `V.s` (volt-seconds) → "volts" (192090), `LM` → "low mode" (194402), `RE` (runaway
   electron) → "relativistic electron" (198913), `LH` (L-H transition) → "lower hybrid waves"
   (186563), `li` (internal inductance) → "lithium" (189262).
2. **Numbers written as words** evade the numeric gate: "thirteen pellets" (source "13 pellets"),
   "approximately eighteen milliseconds" ("About 18 ms"), "thirty kPa", "from five to two hertz",
   "several hundred kilowatts" ("~500kW"), "two-one neoclassical tearing modes" ("2/1 NTM"),
   "three-halves mode ... between two and five seconds" ("3/2 mode from 2 to 5 s").

Correct expansions were common too (RMP → resonant magnetic perturbation, SOL → scrape-off layer,
W → tungsten). The fix is not a glossary: the model must **copy abbreviations verbatim and never
expand them**, and **never write a number as a word**; the gate must catch what it can.

## Deliverables (TDD: failing tests first, then code)

### 1. Prompt v6 — `src/shot_design/retrieval/blurb.py::_system`
Keep every v5 rule (three sentences: goal / success-or-failure incl. disruption, quench, early
termination plainly / one finding or the literal `No notable findings were logged.`; own words;
no quotation marks; no invented numbers; word cap). ADD, in plain prose inside the same prompt:
- Copy every abbreviation, acronym and symbol exactly as the text writes it (for example AE, EHO,
  QH, FPP, PCS, RMP, LM, li, V.s, betan). Never expand, translate or explain one, even when you
  think you know what it stands for, and never introduce an abbreviation the text does not use.
- Write a quantity with digits and the unit exactly as the text gives it, or leave it out. Never
  write a number as a word (not "thirteen", not "several hundred", not "two-one").
Update the docstring ("Prompt v6: ...") and the module docstring's gate summary.

### 2. Config — `configs/shot_design/llm.yaml`
`blurb.prompt_version: 6`, with a comment line in the existing style: v6 = abbreviations verbatim,
never expanded; numbers never as words; gate adds number-word and unknown-acronym rules (audit of
the 2026-09-15 v5 backfill found AE→aeroelastic, EHO→edge helical, QH→quasi-high, FPP→flat
pressure profile, PCS→power control system, betan→poloidal beta, V.s→volts, and spelled-out
numerals).

### 3. Gate — `blurb.gate(source, candidate, rec, max_words)`
Two new checks, placed AFTER the existing number/shot checks and BEFORE the sentence-count check.
Return strings in the existing style.
- **Number words.** Tokenise the candidate on whitespace AND hyphens; lower-case; strip
  punctuation. Number words = two … twelve, thirteen … nineteen, twenty, thirty, forty, fifty,
  sixty, seventy, eighty, ninety, hundred, thousand, million, dozen. `one` is deliberately NOT
  in the set (ordinary English: "one of the beams"). Ordinals (first, second, third) are NOT in
  the set. A number word that does not occur as a whole word (case-insensitive) in `source` →
  `f"number written as a word: {sorted(...)}"`. A number word that IS in the source
  ("two-day experiment") passes.
- **Unknown acronyms.** Tokens matching `\b[A-Z]{2,6}s?\b` in the candidate (strip the trailing
  `s`); compare hyphen-stripped and lower-cased against the hyphen-stripped, lower-cased source
  by SUBSTRING (so `L-H` passes when the source has `PLH`, `ELMs` passes for `ELM`, `NTMs` for
  `NTM`). Always allowed: `DIII`, `D` handled by the regex minimum length (single letters never
  match). Anything else absent → `f"abbreviation not in source: {sorted(...)}"`. Keep the module
  constant names descriptive (`_NUMBER_WORDS`, `_ACRONYM`).

### 4. Targeted regeneration — `shotdb/build.py::write_blurbs` and the CLI
- `write_blurbs(paths, client, only_missing=True, *, limit=None, dry_run=False, shots=None)`.
  When `shots` (sequence of int) is given, `todo` = exactly those shots in ascending order,
  regardless of `only_missing`; a shot absent from `shots.parquet` raises `ValueError` naming it
  BEFORE any model call; `limit` still applies after; `dry_run` semantics unchanged. Rewritten
  rows get the client's current model and prompt version like today.
- Manifest: `manifest["blurbs"]` gains `"prompt_versions": {"5": n, "6": m}` (string keys,
  counts of non-null `blurb_prompt_version` in the whole table), so a mixed database is visible.
  Extend `_blurb_counts` (or wherever the counts are built) so `build()`/`add()` paths that write
  the same block stay consistent — check them and add the key there too if they build it
  separately.
- CLI `shot_design blurb --shots N [N ...]` (`type=int, nargs="+"`). `--shots` with `--all` →
  message on stderr, exit 2. `--shots` with `--dry-run` prints only those shots. Help text in
  the existing style. `scripts/shot_design/blurb_all.sh` already forwards `"$@"`; leave it.

### 5. Tests — `tests/shot_design/test_blurb.py` (+ `test_blurb_provenance.py` where versions are pinned)
Hermetic (tmp_path, fake client as the existing tests do), `-W error`. At least:
- gate rejects "thirteen pellets" when the source says "13 pellets", reason names `thirteen`;
- gate rejects "two-one" for a source with "2/1"; accepts "one of the beams"; accepts a number
  word present in the source; ordinals pass;
- gate rejects an acronym absent from the source (`NBI` when the source says "beams"), accepts
  `L-H` for source `PLH`, accepts `ELMs` for source `ELM`, accepts `DIII-D` always;
- new checks run after the number/shot checks (an invented digit still reports "number not in
  source"), before the sentence check;
- `_system(90)` mentions never expanding abbreviations and never writing numbers as words; the
  prompt string contains `(prompt v6)` when built through `make` with the yaml config;
- `llm.yaml` `blurb.prompt_version == 6` (update any test pinning 5);
- `write_blurbs(..., shots=[a, c])` rewrites exactly those rows (text, model, version) and leaves
  every other row's text and version untouched; unknown shot → `ValueError` with no model call;
  manifest `blurbs.prompt_versions` histogram correct on a mixed table; `only_missing` ignored
  when `shots` is given; `limit` applies after;
- CLI: `--shots` + `--all` exits 2; `--shots ... --dry-run` prints only those shots and writes
  nothing.

### 6. Docs — `docs/SHOT_DESIGN.md`, section "Local LLM and per-shot blurbs"
Describe v6's two rules, the two new gate reasons, `blurb --shots`, the manifest histogram, and
one sentence on the audit that motivated v6 (with 200729 as the example).

## Verification (run from the worktree; paste outputs into the report)
```
cd /scratch/gpfs/nc1514/FusionAIHub-LLM2
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e ideate-cpu python -m pytest tests/shot_design -q -W error -p no:cacheprovider
/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/ruff check --no-cache src/shot_design scripts/shot_design tests/shot_design
```
Baseline before your change: 1479 passed, 0 skipped, ruff clean. Both must be green.

## Hard rules
- Every `pixi run` MUST be exactly `pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e ideate-cpu ...` with `PYTHONPATH` set to THIS worktree's `src`. Never `pixi install`, `pixi lock`, `pixi update`, never a bare `pixi run`.
- Never run `shot_design blurb` / `build` / `add` / `labels join` against the production data root (`/scratch/gpfs/EKOLEMEN/nc1514/ideate`). Tests write only under `tmp_path`.
- Do not edit `docs/superpowers/plans/**`, `pixi.lock`, `pyproject.toml`, anything under `src/labeler`, `tests/labeler`, `data/`, or `configs/shot_design/phenomena.yaml`.
- Do not touch `/scratch/gpfs/nc1514/FusionAIHub` (the main checkout) or any other worktree.
- Commit in this worktree when green: subject `shot_design: blurb prompt v6 (abbreviations verbatim, no number words), gate rules, blurb --shots`; body states the test counts; end the message with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Write `.superpowers/sdd/task-LLM2-report.md` (what changed, test/ruff output, anything left) and include it in the commit.
