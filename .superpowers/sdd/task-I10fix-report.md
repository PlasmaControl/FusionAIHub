# Task I10-fix — report

Worktree `/scratch/gpfs/nc1514/FusionAIHub-I10`, branch `recommender-I10`, base for this loop
`c52332d`. Review applied: `.superpowers/sdd/review-I10.md` (verdict FIX-THEN-MERGE), brief
`.superpowers/sdd/task-I10fix-brief.md`. Findings F1–F8 addressed; one part of F3 skipped and
reported below. TDD throughout: every behavioural test was written first and observed failing for
the right reason before the implementation existed (6 failures on the first run: F1×2, F2×2, F3, F4).

## Commits

| commit | finding group |
| --- | --- |
| `c6a9016` | F1 (+ F8) |
| `05b7764` | F2 |
| `01773db` | F3 |
| `890bb65` | F4–F7 |

## Per finding

### F1 (blocking) — a missing or unreadable evidence table is now named

`src/ideate/mcp/tools.py:974-981`. After `ph.locate(...)` returns and the tier caveat is
appended, the tool walks `db.load_errors` for `events`, `labels_wide`, `text_claims`,
`event_sources` and appends `could not read <name>.parquet: <message>` for each
(**tools.py:979**), then appends the existing `NO_EVENTS` constant (tools.py:50) when
`config.load_paths().db_dir / "events.parquet"` does not exist (**tools.py:981**). This mirrors
`get_events` (tools.py:545-552) verbatim in wording; no new sentence was invented.

Tests: `tests/ideate/test_mcp.py:863` (`..._without_an_events_table_says_the_join_has_not_run`,
modelled on the `get_events` test at test_mcp.py:254) and `tests/ideate/test_mcp.py:873`
(`..._names_an_evidence_table_it_could_not_read`, which writes garbage bytes over
`events.parquet` so `ShotDB.load` records a real `load_errors["events"]`, and asserts
`NO_EVENTS` is *not* also emitted — a torn table is not a missing one).

### F2 (blocking) — an empty `hits` list says which absence it is

`src/ideate/mcp/tools.py:986-989`. `ph.NO_DETECTOR.format(id=top)` is appended whenever the
resolved phenomenon has no `covering_sources` (tools.py:987) — emitted regardless of whether
`hits` is empty, as the review asked, because on a non-empty list it is the reason every hit
rests on text or a curated row. When `hits` is empty, `NO_EVIDENCE` is appended
(tools.py:989).

`NO_EVIDENCE` is the one new constant, defined next to `NOTHING_RESOLVED` at
**`src/ideate/mcp/tools.py:851`**:

```
no shot carried evidence for this phenomenon under these filters; absence of evidence here is
not a negative
```

Its docstring records why `phenomena.py` has no constant to reuse: that module's vocabulary is
per-shot (`NO_DETECTOR`, the coverage states) and this is a fact about the result set.

Tests: `tests/ideate/test_mcp.py:884` (`ip_mean` lower bound of `1e12` ⇒ `hits == []`,
`NO_EVIDENCE` present, and at least one caveat that is not the ranking sentence) and
`tests/ideate/test_mcp.py:896` (`"resistive wall mode"` ⇒ `rwm`, whose `covering_sources` is
empty ⇒ `ph.NO_DETECTOR.format(id="rwm")` present). `detachment`, `rwm` and `fast_ion` are the
three registry entries with no covering source today.

### F3 (non-blocking) — CLI flag spellings, **partly done**

Done:
* `locate` gains keyword-only `option: str = "--avoid"`
  (**`src/ideate/retrieval/phenomena.py:1316`**, documented in the docstring) and passes it to
  `_avoid_ids(avoid, option=option)` (**phenomena.py:1343**) — the kwarg `_avoid_ids` has taken
  all along for exactly this purpose.
* `phenomenon_locate` calls `ph.locate(..., option="avoid")`
  (**`src/ideate/mcp/tools.py:953`**). A bad token now reads `avoid 'phenomenon:banana': ...`
  to an MCP caller instead of naming a flag that is not in its schema.
* `DROPPED_UNSCORED` reworded from `--min-confidence {limit}` to `min_confidence {limit}`
  (**`src/ideate/retrieval/phenomena.py:237-240`**) — the parameter name the CLI flag and the
  MCP argument share.

Test: `tests/ideate/test_mcp.py:783` widened — `"avoid 'phenomenon:banana'" in got["error"]`
and `"--avoid" not in got["error"]`.

**Skipped, with the reason.** The review's third item was to give `AVOID_DROPPED`
(phenomena.py:231) an `{option}` slot. **Not done.** `src/ideate/shotdb/store.py:346-348`
formats that same constant with `token`/`n`/`title` only, so adding an `{option}` field makes
`str.format` raise `KeyError('option')` on the `search_shots`/`ShotDB` label-avoid path —
and `store.py` is outside this task's editable set (`tools.py`, `phenomena.py`,
`test_mcp.py`, `.superpowers/sdd/*`). Breaking a sibling path was the worse of the two
outcomes, so the constant is unchanged and the `notes` sentence an MCP caller sees still
spells `--avoid`. The same is true of the `AVOID_COVERAGE_CAVEATS` templates
(phenomena.py:207-224), which the review did not raise and which `tests/ideate/test_phenomena.py`
pins by literal at lines 538, 540, 753, 767 and 1064. Both belong on the ledger as one small
follow-up that touches `store.py` and `phenomena.py` together.

The `DROPPED_UNSCORED` reword did **not** need the brief's escape hatch: nothing in the
committed `tests/ideate/test_phenomena.py` or `tests/ideate/test_ui.py` pins that literal
(checked by grep and by running both suites). `docs/IDEATE.md:261` describes the caveat as
"dropped by `--min-confidence`", which remains a true description of the CLI flag and is not
pinned equal to the constant; `docs/` is outside the editable set in any case.

### F4 (nit) — one wording for a malformed constraint

`src/ideate/mcp/tools.py:962-965`: the `(ValueError, TypeError)` arm is now
`_error(str(exc), caveats)`, matching `search_shots` (tools.py:240). The `ValueError:` prefix is
gone. Test: `tests/ideate/test_mcp.py:797` asserts the two tools return the **same** error string
for the same malformed constraint, which pins the equality rather than one spelling.

### F5 (nit) — the server's copy of the ranking rule is pinned

Test only: `tests/ideate/test_mcp.py:907` asserts
`ph.RANKING_SENTENCE in " ".join(server_mod.INSTRUCTIONS.split())` (whitespace-normalised,
because the copy is wrapped across a newline). `server.py` is unchanged — the copy is correct
today; it was merely unpinned.

### F6 (nit) — every registry id is accepted

Test only: `tests/ideate/test_mcp.py:916` loops all `ph.registry()` ids (17 today, including the
underscored `qmin_hybrid` / `fast_ion`) through `phenomenon_locate(pid, n=1)` on the small
fixture database and asserts a payload with `"phenomenon"` and no `"no phenomenon resolved"`
error. This pins the tool's own error message against the lexicon edit another front is making.

### F7 (nit) — the payload is JSON

Test only: `tests/ideate/test_mcp.py:756`, `json.loads(json.dumps(got)) == got` in the success
test (round-tripped, so it pins equality and not just that `dumps` did not raise).

### F8 (nit) — section break

`tests/ideate/test_mcp.py:929`: the `# --- the registry` section comment now has the two blank
lines above it that every other section comment in the file has. Taken in the F1 commit.

### F9 / F10

Out of remit by the review's own words (F9: a docs census over the production database, a
separate pass; F10: a note, no action). Neither touched. F10's substance — the `eval/latency.py`
row named `phenomenon_locate` measures `phenomena.locate`, not this MCP tool, and no `avoid`
case is covered — belongs on the ledger for the next latency pass.

## Verification

Full `tests/ideate` suite, the mandated command, from the worktree root:

```
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src HF_HUB_OFFLINE=1 pixi run --frozen --no-install \
  --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e ideate-cpu \
  python -m pytest tests/ideate -q -W error -p no:cacheprovider
```

last line:

```
1269 passed in 170.59s (0:02:50)
```

(1262 before this loop; +7 = six new tests plus one widened, all in `tests/ideate/test_mcp.py`,
which goes 61 → 68.)

Ruff:

```
/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/ruff check --no-cache src/ideate tests/ideate
All checks passed!          (exit 0)
```

Scope, `git diff --name-only c52332d..HEAD`:

```
src/ideate/mcp/tools.py
src/ideate/retrieval/phenomena.py
tests/ideate/test_mcp.py
```

Nothing under `docs/`, `configs/`, `data/`, `src/labelmaker/`, and
`tests/ideate/test_phenomena.py` untouched. Working tree clean apart from this report. No
`pixi install`, no bare `pixi run`, no `-e ideate`; no production database or
`/scratch/gpfs/EKOLEMEN/...` path was read or written — every test runs on `tmp_path` with
`IDEATE_DATA_ROOT` monkeypatched. Nothing was created under `/scratch/gpfs/nc1514` outside this
worktree.

## Carried forward

1. `AVOID_DROPPED` and `AVOID_COVERAGE_CAVEATS` still spell `--avoid` to an MCP caller; fixing
   them means editing `src/ideate/shotdb/store.py:346` and `tests/ideate/test_phenomena.py`
   together, both outside this task.
2. F9's stale census claims at `docs/IDEATE.md:165` and `:478` (two occurrences, not one) need a
   census over the production database.
3. F10: `eval/latency.py`'s `WHAT["phenomenon_locate"]` string should say the row is not the MCP
   tool of that name, and the next latency pass should carry an `avoid` case (roughly twice the
   work of the measured one).
