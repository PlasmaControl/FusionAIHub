# Review — task I10, `phenomenon_locate` (read-only)

Branch `recommender-I10` @ 9d7d525 in `/scratch/gpfs/nc1514/FusionAIHub-I10`, base `recommender` @ a3db1e7.

## Verdict: **FIX-THEN-MERGE**

The contract is met item for item. The tool is `@never_raises`-registered like its three siblings,
every reply carries `caveats`, no-resolution is an error dict listing all 17 `id (title)` pairs and
carries no `hits` key at all, `notes` is a separate list with its own top-level caveat, `constraints`
goes through the same `_range` and the unknown column comes back as an error, `segment` goes through
`_segment` (alias correction included), `ALL_FORECASTS` is emitted on a forecast-only database, per-hit
caveats are passed through untouched, the payload is plain JSON, and the order `locate` produced
reaches the caller unre-sorted. No caveat sentence was invented: `_TIER_CAVEAT` (tools.py:851) *formats*
`ph.RANKING_SENTENCE` / `ph.TEXT_ONLY` / `ph.DATABASE_ONLY` rather than restating them, and
`ALL_FORECASTS` is `ph.ALL_FORECASTS.format(n=…)`. The two genuinely new strings (`NOTHING_RESOLVED`,
`_NOTES_CAVEAT`) say things `phenomena.py` has no sentence for. Scope is clean; the suite and ruff are
green; the report's numbers reproduce.

Two findings hold it back, and they are one edit and one test between them: on a database where
`labels join` has not run, or where an evidence table failed to load, or where the constraints filtered
everything out, this tool answers `{"n": 0, "hits": [], "caveats": [<the ranking rule>]}` — which reads
as "no shot has this phenomenon". That is the exact misreading the tool refuses to allow on the
*resolve* path (tools.py:920-927, and `NOTHING_RESOLVED` itself), and which `get_events` refuses on the
same data (`NO_EVENTS` at tools.py:552, the `load_errors` caveat at tools.py:545-546). Fix F1 and F2,
re-run `tests/ideate`, merge.

---

## Findings, by severity

### F1 (should-fix, blocking the merge) — a missing or unreadable evidence table is silent here, and named by every sibling

`src/ideate/mcp/tools.py:933-964`

`phenomenon_locate` reads `db.events` / `db.labels_wide` / `db.text_claims` / `db.event_sources`
indirectly through `ph.locate`, and never looks at whether those tables are actually there.
`ShotDB.load` (`src/ideate/shotdb/store.py:149-160`) deliberately does *not* raise on a missing or
torn optional table: a missing one is skipped, an unreadable one is recorded in `db.load_errors`, and
in both cases `db.events` is the **empty typed frame** (`store.py:110-115`). The comment at
`store.py:145-148` says why in as many words — "the reader that needs the table (`mcp.tools.get_events`,
`retrieval.phenomena`) reports the failure in its own words". `get_events` does exactly that
(`tools.py:545-546`, `tools.py:551-558`). `phenomenon_locate` does not, so:

* on any database built without `ideate labels join`, the reply is an empty or text-only hit list with
  no `NO_EVENTS` caveat — and no `ALL_FORECASTS` either, because the guard at `tools.py:960-961`
  short-circuits on `len(db.events) == 0`;
* on a torn `events.parquet`, the same reply, with `db.load_errors["events"]` sitting unread. Worse than
  silence: `evidence()` then reports `coverage_state == "unprocessed"`, whose established meaning is
  "no detector ran; absence is not evidence" — a claim about the *machine* standing in for a claim about
  a file that would not open.

Why it matters: this is the tool's headline promise (server.py:45-49, "Every reply the tools THEMSELVES
produce carries `caveats`, and they change what the reply means") failing in the one state a fresh
database is actually in.

Fix — after the `locate` call, before the return (tools.py:952):

```python
for name in ("events", "labels_wide", "text_claims", "event_sources"):
    if name in db.load_errors:
        caveats.append(f"could not read {name}.parquet: {db.load_errors[name]}")
if not (config.load_paths().db_dir / "events.parquet").exists():
    caveats.append(NO_EVENTS)
```

`NO_EVENTS` (tools.py:49) is already the exact string a caller keys on, so reuse it rather than writing
a second spelling. Test: build the fixture without `write_events` and assert `NO_EVENTS in got["caveats"]`
— `tests/ideate/test_mcp.py:255` (`test_get_events_without_an_events_table_says_the_join_has_not_run`)
is the model.

### F2 (should-fix, blocking the merge) — an empty `hits` list carries no caveat saying which absence it is

`src/ideate/mcp/tools.py:953-975`

When `hits` is empty the only top-level caveat is the ranking rule. Three very different facts produce
that reply and the caller cannot tell them apart: (a) the constraints or the segment left no candidate
at all; (b) candidates existed and none carried any evidence; (c) the phenomenon has **no detector
registered** — `ph.covering_sources` empty, true today for several registry entries — so an empty list
could never have contained an observed hit in the first place. `search_shots` distinguishes exactly this
family (`tools.py:266-279`: "nothing passed the filters" / "no channel had anything to search on" /
"N candidate(s) passed the filters, none ranked"), and `get_events` emits `ph.NO_DETECTOR` for case (c)
at `tools.py:541-542`.

Why it matters: an unqualified empty list from a phenomenon tool reads as "no shot has one", which is
the very answer `NOTHING_RESOLVED` (tools.py:842-845) exists to stop the tool from giving by accident.

Fix — reuse the existing vocabulary, invent nothing:

```python
if not reg[top].covering_sources:
    caveats.append(ph.NO_DETECTOR.format(id=top))
if not hits:
    caveats.append("no shot carried evidence for this phenomenon under these filters; "
                   "absence of evidence here is not a negative")
```

(The first line is worth emitting whether or not `hits` is empty — it is the `ALL_FORECASTS` fact
restricted to one phenomenon, and on a hit list it explains why every hit says `NO_OBSERVED`.) Test:
`phenomenon_db` plus `constraints={"ip_mean": {"lo": 1e12}}` → `hits == []` and a caveat that is not
just the ranking sentence.

### F3 (should-fix, non-blocking) — CLI flag spellings reach an MCP caller that has no flags

`src/ideate/mcp/tools.py:941` and `:944-945`; strings at `src/ideate/retrieval/phenomena.py:231`,
`:235-238`, `:1281-1291`

Three of the sentences this tool returns are written for `ideate phenomenon` at a terminal:

* `notes` carries `AVOID_DROPPED` = `"--avoid {token}: dropped {n} shot(s) …"` (phenomena.py:231) — so the
  tool's own `notes` list tells a model about a `--avoid` flag;
* a bad avoid token comes back as `"--avoid 'phenomenon:banana': …"` — `_avoid_ids` takes an `option=`
  kwarg for precisely this reason (phenomena.py:1281), but `locate` does not expose it;
* per-hit caveats carry `DROPPED_UNSCORED` = `"… cannot be shown to reach --min-confidence {limit}"`
  (phenomena.py:235-238), which the MCP surface reaches for the first time with this tool.

Why it matters: a model told to fix a flag that is not in its schema has no move; and this is the first
MCP surface for these strings, so today is when the wording stops being CLI-only.

Fix (smallest correct one): add `option: str = "--avoid"` to `locate`'s keyword-only signature, pass it
to `_avoid_ids`, and have the tool call `ph.locate(..., option="avoid")`; do the same for the
`--min-confidence` literal by making it a `{option}` slot. `phenomena.py` is not on the brief's excluded
list, but if this is judged scope creep it belongs on the ledger as an I-follow-up rather than being
dropped.

### F4 (nit) — the same malformed constraint is worded two ways across sibling tools

`src/ideate/mcp/tools.py:950-951` vs `:239-240`

`_range` raising `ValueError` is reported by `search_shots` as `_error(str(exc), caveats)` and by
`phenomenon_locate` as `_error(f"{type(exc).__name__}: {exc}", caveats)`, so the identical failure reads
`constraint … is neither …` in one tool and `ValueError: constraint … is neither …` in the other. The
brief's stated reason for reusing `_range` ("a model that learnt the constraint shape from `search_shots`
need not learn a second one") applies to the failure text too. Fix: drop the type prefix on the
`(ValueError, TypeError)` arm, or accept the divergence deliberately and say so in a comment.

### F5 (nit) — `INSTRUCTIONS` hand-copies `RANKING_SENTENCE`, and nothing pins the copy

`src/ideate/mcp/server.py:71-80`

The paragraph is accurate today — it is `phenomena.RANKING_SENTENCE` character for character — but it is
a literal in a Python string, wrapped across a newline, while `docs/IDEATE.md:162-164` and
`configs/ideate/retrieval.yaml` are pinned equal to the constant by a test (`test_phenomena.py`). If the
constant is ever reworded, the server's instructions drift silently and the server is the surface that
matters most. Fix: a test in `test_mcp.py` asserting `ph.RANKING_SENTENCE in " ".join(server_mod.INSTRUCTIONS.split())`
(whitespace-normalised, because of the wrap). `test_phenomena.py` is excluded from this task; `test_mcp.py`
is not.

### F6 (nit) — nothing pins that a registry **id** is accepted, though the tool's own error hands ids out

`src/ideate/mcp/tools.py:875-878` and `:923-926`

The docstring promises "free text or a registry id", and the no-resolution error tells the caller to
"try one of: tearing (Tearing mode), …" — so the next call a model makes is with a bare id. That works
today for all 17 (verified: `resolve(pid) == [(pid, 1.0)]` for every id in the registry, including the
underscored `qmin_hybrid` / `fast_ion`), but only because each id happens to be listed as its own alias
in `src/labelmaker/events/lexicons.yaml`, which another front is editing in the user's working tree right
now. A lexicon edit that drops an id-as-alias puts a model into a loop: the error offers an id the tool
then rejects. Fix: one test in `test_mcp.py` — for every `ph.registry()` id, `phenomenon_locate(pid)`
returns a payload and not `"no phenomenon resolved"`.

### F7 (nit) — the payload's JSON-serialisability is asserted nowhere

`src/ideate/mcp/tools.py:965-975`

It is in fact safe: every `PhenomenonHit` field is a typed scalar or a model of typed scalars
(`schema.py:245-319`), so `model_dump(mode="json")` yields plain types, and `resolved`/`n`/`title` are
plain. But the CLI hedges the same objects with `json.dumps(..., default=str)` (`cli.py:1493`), the brief
names JSON-serialisability as contract, and a later field of a less tame type would surface as a
protocol error rather than a caveated one. Fix: `json.dumps(got)` in the success test — one line.

### F8 (nit) — test-file section break has one blank line where the file uses two

`tests/ideate/test_mcp.py:842`

Cosmetic; ruff's default rule set does not include E302 so it passes. Every other section comment in the
file has two blank lines above it.

### F9 (docs) — the stale "all 1,037 event rows are forecasts" claim: **not I10's to fix**

`docs/IDEATE.md:165` and `docs/IDEATE.md:478`

The implementer's finding is correct, and there are **two** occurrences, not one; the report names only
the first:

* :164-168 — "**On the `recommender_v1` database today every one of the 1,037 event rows is a forecast**,
  so `ideate phenomenon` returns forecast-, label- and text-class hits and no observed ones at all";
* :478-481 — "(On the `recommender_v1` database today *all* 1,037 event rows are forecasts and no shot
  has an observed-event product, so `get_events` answers `unprocessed` for every one of the 500 …)".

Both are contradicted by the read-only smoke in the report (five **observed** `tokeye_track` hits on
186036/186034/186055/186059/185956, and `ALL_FORECASTS` correctly absent), and by the ledger's own note
that detector products now exist on the L12 pilot shots.

Remit: **outside I10.** The brief's docs remit is exactly two edits — "the '### The tools' table: one row"
and "the 'The MCP server' prose" — and both were made, correctly (docs/IDEATE.md:392, :429; the row's
column format, argument list and described payload all match the implementation). Line 165 sits in the
ranking section and line 478 in the `get_events` coverage section; both carry a *census* claim whose new
true value nobody has measured in this task (the smoke read five rows, not a count), and fixing them means
re-running a census and touching two sections this branch otherwise does not. That is a separate docs
pass, and the numbers should come from a census over the production database, not from five hits. Carry
it to the ledger as such. Do not let it hold the merge.

### F10 (note, no action) — what the 175 ms does and does not say

`src/ideate/eval/latency.py:65-79`, `:168`

Recorded so the next reader is not misled, not as a defect: the harness row **named** `phenomenon_locate`
measures `phenomena.locate` over the whole database on `elm` (`WHAT["phenomenon_locate"]`, and the lambda
at :168), not this MCP tool, and `latency.py:25-26` records that row over budget in 6 runs out of 6 on a
loaded node. The report's 175.0 / 174.5 / 173.4 ms is the MCP call on `eho`, `n=5`, warm, quiet node, and
**without `avoid`** — `locate` runs a second `evidence()` per candidate per avoided phenomenon
(phenomena.py:1355-1360), so an `avoid` call is roughly twice the work measured. The report discloses the
first of these plainly; the `avoid` cost is the part it does not mention. Cost scales with candidate count,
not with `n` (phenomena.py:1350 iterates all candidates, truncating only at :1390), so it is linear in
database size. Nothing to tune here — the brief forbade it — but the next latency pass should carry an
`avoid` case, and the `WHAT` string should say the row is not the MCP tool of the same name.

---

## Answers to the review questions, in brief

1. **Contract fidelity** — met. `never_raises` at registration (server.py:32-35), not at definition, as
   the module requires; error dicts carry the accumulated caveats on every arm except the `_db()` arm,
   which returns `_db()`'s own error — identical to `search_shots:225-227`, so the segment-alias caveat is
   dropped there in both tools, deliberately. No-resolution is an error with the titles and no `hits` key
   (tools.py:920-927). `notes` is its own list plus `_NOTES_CAVEAT` (tools.py:963-964, :973). `_range` and
   the `KeyError` arm are `search_shots`' verbatim. `ALL_FORECASTS` guard is `db.events`-safe (empty typed
   frame ⇒ `len == 0` ⇒ short-circuit) and uses `ph.FORECAST_KIND` where the CLI used the literal
   `"forecast"` — an improvement. Exception order is right: `PhenomenaError` subclasses `ValueError`
   (phenomena.py:248) and is caught first, at :944 and :918.
2. **Invented caveats** — none. `_TIER_CAVEAT` formats three existing constants; `ALL_FORECASTS` is
   formatted, not restated; hit caveats are untouched. `NOTHING_RESOLVED` and `_NOTES_CAVEAT` are new
   sentences for facts `phenomena.py` has no constant for (the CLI expressed both as *placement* — stderr,
   exit 2 — which a payload cannot).
3. **Test quality** — good. All ten fail without the implementation for a real reason (`phenomenon_locate`
   does not exist at `dc110d2^` in either `tools.py` or `server.py`: 0 occurrences in each), and they
   assert behaviour rather than mirror it — the ordering test (test_mcp.py:733) pins `[100, 101, 200]`
   over a fixture built to hold one shot per evidence class, which is the only shape that can prove the
   *class-before-score* order survives the call; the constraint test (:789) pins both spellings against the
   same result set and genuinely filters (shots at 1.0/1.05e6 drop out); `ALL_FORECASTS` is asserted through
   the constant, not a copy. The incomplete-db path **is** tested, and correctly through `_registered(...)`
   rather than the plain function (:820-832), which is the only way the `never_raises` promise is under
   test. No test touches the production database or wall-clock: every one is on `ideate_db`/`tmp_path` with
   `IDEATE_DATA_ROOT` monkeypatched and the `_no_cached_db` autouse fixture clearing the lru_cache. Gaps:
   F1, F2, F6, F7 above.
4. **Docs** — the table row (IDEATE.md:429) and the prose (:392) are accurate to the payload and in the
   siblings' format. The `INSTRUCTIONS` paragraph (server.py:71-80) is accurate: class order, rank ≠
   strength, unresolved text is an error not an empty result, `notes` is about excluded shots, and it does
   not restate the `get_events` text. Stale-census sentence: see F9 — real, two occurrences, separate pass.
5. **Scope** — clean. `git diff --name-only a3db1e7..recommender-I10` is six files, all allowed; nothing
   under `configs/ideate/`, `tests/ideate/test_phenomena.py`, `data/events/**`, `src/labelmaker/**`,
   `scripts/labelmaker/**`, `tests/labelmaker/**`, `docs/superpowers/plans/**`. Working tree clean. Note
   for the merger: `git diff recommender..recommender-I10` *also* shows
   `docs/superpowers/plans/2026-09-07-recommender-ledger.md` — that is base drift, not an I10 edit
   (`recommender` gained 6c374b0 and 7aef67e after the branch was cut at a3db1e7), and it appears as a
   **reversal** of two ledger entries. Merge `recommender` into the branch (or merge the branch forward)
   rather than applying that diff.
6. **Correctness risks** — `avoid` token validation is real and tested (F3 is its wording, not its
   existence). `n` is unbounded upward but `locate` clamps at `max(int(n), 0)` (phenomena.py:1390) so a
   negative is an empty list and a huge one is the whole database; `int(n)`, `float(min_confidence)` and
   the `_range` comprehension are all *inside* the try at :933, so a bad value is a message and not a
   `never_raises` stack-trace string. `min_confidence` is unvalidated: a value above 1.0 silently drops
   every scored event, and unscored events are only dropped when `min_confidence > 0` — which makes the
   docstring at :890-893 true as written, though a reader could take it to apply at the default 0.0. No
   mutable default arguments (`None` for both `constraints` and `avoid`; `notes` is a fresh list per call
   at :932). Blocking cost: see F10 — linear in candidates, same order as the existing `search_shots`, so
   not new to this tool.

## Files in the diff

From the branch's own commits (`git diff --name-only a3db1e7..recommender-I10`):

| file | commit |
| --- | --- |
| `.superpowers/sdd/task-I10-brief.md` | f404f9a (the controller's brief) |
| `tests/ideate/test_mcp.py` | dc110d2 |
| `src/ideate/mcp/tools.py` | c7dd9b0 |
| `src/ideate/mcp/server.py` | c7dd9b0 |
| `docs/IDEATE.md` | 9d7d525 |
| `.superpowers/sdd/task-I10-report.md` | 9d7d525 |

`docs/superpowers/plans/2026-09-07-recommender-ledger.md` appears only in `git diff recommender..recommender-I10`
and only as a reverse diff: base drift, not a change made by this task (see §5).

This review file, `.superpowers/sdd/review-I10.md`, is written and left uncommitted. Nothing else was
edited, no branch was checked out, no file was created anywhere else under `/scratch/gpfs/nc1514`.

## What I ran

| command | exit |
| --- | --- |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src HF_HUB_OFFLINE=1 pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e ideate-cpu python -m pytest tests/ideate/test_mcp.py -q -W error -p no:cacheprovider` → `61 passed in 7.68s` | **0** |
| `/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/ruff check --no-cache src/ideate tests/ideate` → `All checks passed!` | **0** |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src <ideate-cpu python> -c "for pid in ph.registry(): print(ph.resolve(pid))"` — read-only, no database opened; all 17 ids resolve to themselves at weight 1.0 (evidence for F6) | **0** |
| `git` reads: `log`, `show`, `diff --stat/--name-only`, `merge-base`, `status --porcelain`, `show dc110d2^:…` | 0 |

Both `pixi run` invocations used the mandated `--frozen --no-install --manifest-path … -e ideate-cpu`
form; ruff was run as the labelmaker env's binary with `--no-cache` (it is not in `ideate-cpu`), the same
deviation the report discloses. No write command, no `pixi install`, no `.pixi` created in the worktree.
The full `tests/ideate` suite was not re-run here (the brief's reviewer command is the `test_mcp.py`
subset); the report's `1262 passed` stands unreproduced by me.
