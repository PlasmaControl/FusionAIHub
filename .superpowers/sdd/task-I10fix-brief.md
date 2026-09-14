# Task I10-fix — apply review findings F1–F8 to `phenomenon_locate`

Worktree `/scratch/gpfs/nc1514/FusionAIHub-I10`, branch `recommender-I10` @ 9d7d525. Review: `.superpowers/sdd/review-I10.md` (verdict FIX-THEN-MERGE). Commit review + this brief first (`ideate: I10 review and fix brief`), then one commit per finding group, trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Blocking
- **F1** `src/ideate/mcp/tools.py::phenomenon_locate` — after `ph.locate(...)` returns, before the payload: for each of `events, labels_wide, text_claims, event_sources` in `db.load_errors` append `f"could not read {name}.parquet: {db.load_errors[name]}"`; if `config.load_paths().db_dir / "events.parquet"` does not exist append the existing `NO_EVENTS` constant (tools.py:50). Mirror `get_events` (tools.py:545-552). Test (TDD, `tests/ideate/test_mcp.py`): fixture without an events table → `NO_EVENTS in got["caveats"]`; model `test_get_events_without_an_events_table_says_the_join_has_not_run`. A second test: a torn table recorded in `db.load_errors` (monkeypatch or write garbage `events.parquet` and build the db the way the store does) → caveat names the file.
- **F2** same function — if `not reg[top].covering_sources`: `caveats.append(ph.NO_DETECTOR.format(id=top))` (emit regardless of hits). If `not hits`: append a plain sentence that the empty list is not a negative, e.g. `"no shot carried evidence for this phenomenon under these filters; absence of evidence here is not a negative"`; define it as a module constant next to `NOTHING_RESOLVED`. Test: `phenomenon_db` + `constraints={"ip_mean": {"lo": 1e12}}` → `hits == []` and a caveat beyond the ranking sentence; a phenomenon with no covering sources → `ph.NO_DETECTOR.format(id=...)` present.

## Non-blocking (do them; all small)
- **F3** `src/ideate/retrieval/phenomena.py`: add keyword `option: str = "--avoid"` to `locate` and pass it to `_avoid_ids(avoid, option=option)`; make `AVOID_DROPPED = "{option} {token}: dropped {n} shot(s) with observed {title} evidence"` and pass `option=option` at its `.format` (~line 1382). Tool calls `ph.locate(..., option="avoid")`. `DROPPED_UNSCORED` (line 235): reword the `--min-confidence` literal to `min_confidence` (the parameter name shared by CLI and tool). Constraint: **do NOT edit `tests/ideate/test_phenomena.py`** (the user has uncommitted edits to it in the main checkout; a branch touching it cannot merge). If the committed `test_phenomena.py` pins a literal that would break, keep behaviour and report which part you skipped. Test in `test_mcp.py`: a bad avoid token's error message contains `avoid '…'` and not `--avoid`.
- **F4** tools.py `(ValueError, TypeError)` arm: `_error(str(exc), caveats)` to match `search_shots` (~line 239).
- **F5** test in `test_mcp.py`: `ph.RANKING_SENTENCE in " ".join(server_mod.INSTRUCTIONS.split())`.
- **F6** test: for every `pid in ph.registry()`, `phenomenon_locate(pid)` (via the registered/never_raises wrapper or plain, as the existing tests do) returns a payload with `"phenomenon"` and no `"no phenomenon resolved"` error — on the small fixture db.
- **F7** `json.dumps(got)` in the success test.
- **F8** two blank lines above the section comment at `tests/ideate/test_mcp.py:~842`.

## Rules
- Only these files: `src/ideate/mcp/tools.py`, `src/ideate/retrieval/phenomena.py`, `tests/ideate/test_mcp.py`, `.superpowers/sdd/*`. Nothing under `docs/superpowers/plans/**`, `configs/`, `data/`, `src/labelmaker/`, `tests/ideate/test_phenomena.py`. Nothing new under `/scratch/gpfs/nc1514` except files in this worktree.
- Every `pixi run` MUST be: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src HF_HUB_OFFLINE=1 pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e ideate-cpu python -m pytest tests/ideate -q -W error -p no:cacheprovider` (full ideate suite at the end; `tests/ideate/test_mcp.py` while iterating). Never a bare `pixi run`, never `pixi install`.
- Ruff: `/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/ruff check --no-cache src/ideate tests/ideate`.
- No production database writes; tests only on `tmp_path` fixtures.
- Report to `.superpowers/sdd/task-I10fix-report.md` (commit it): per finding what changed + file:line, full-suite count, ruff, anything skipped and why.
