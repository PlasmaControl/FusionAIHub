### Task E3: Parallel blurb backfill with Gemini Flash

**Files:**
- Modify: `src/shot_design/shotdb/build.py` (`write_blurbs(..., workers: int = 1)`), `src/shot_design/cli.py` (`blurb --workers N`, default 1)
- Create: `scripts/shot_design/blurb_frontier.sh`
- Note: no Frontier serve script exists (nothing is served on Frontier); `scripts/shot_design/serve_llm.sbatch` is Stellar-only and stays.
- Test: `tests/shot_design/test_blurb_backfill.py`

**Interfaces:**
- `write_blurbs(paths, client, only_missing=True, *, limit=None, dry_run=False, shots=None, workers=1) -> int`. With `workers > 1` the `_blurb.make(rec, client, ...)` calls run in a `concurrent.futures.ThreadPoolExecutor(max_workers=workers)`; results are written back into `df` in shot order after all futures resolve; the parquet/manifest rewrite is unchanged and single-threaded. The `agy` provider is a subprocess per call so threads are safe; the request cache must be safe for concurrent writers (write `<sha>.json.part` then `os.replace`).
- `scripts/shot_design/blurb_frontier.sh`: `source _shot_design_common.sh`; `"$PY" -m shot_design blurb --workers "${WORKERS:-8}" "$@"`; runs on the login node (no sbatch — `agy` needs the network and the OAuth cache in `~/.gemini`).

- [ ] **Step 1: Failing test**

```python
def test_write_blurbs_workers_writes_every_row(tmp_db, fake_client):
    from shot_design.shotdb.build import write_blurbs
    n = write_blurbs(tmp_db.paths, fake_client, workers=4)
    df = pd.read_parquet(tmp_db.paths.db_dir / "shots.parquet")
    assert n == len(df) and (df["blurb_source"] == "llm").all()
    assert fake_client.max_concurrent >= 2
```
(`fake_client` returns a gate-passing three-sentence blurb after `time.sleep(0.05)` and records the peak number of in-flight calls; reuse `tests/shot_design/conftest.py` fixtures for `tmp_db`.)

- [ ] **Step 2–4:** fail → implement → pass (whole `tests/shot_design`).
- [ ] **Step 5: Dry run on 5 shots** (after F2's build): `bash scripts/shot_design/blurb_frontier.sh --dry-run --limit 5` — read the five candidates and gate verdicts; every acronym must be verbatim from the source text.
- [ ] **Step 6: Commit** `git commit -m "shot_design blurb: --workers thread pool; Frontier backfill script"`

The full backfill itself is F2 Step 4.

