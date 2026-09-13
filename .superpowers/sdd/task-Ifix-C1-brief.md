# Task Ifix-C1 — iteration-0 critic fixes, ideate side (defects 3 and 2b; report items 5, 6, 7, 8)

Worktree /scratch/gpfs/nc1514/FusionAIHub-Ifix, branch `recommender-Ifix` (from `recommender` HEAD); absolute paths only; never touch /scratch/gpfs/nc1514/FusionAIHub or any other worktree.
Test command: `cd /scratch/gpfs/nc1514/FusionAIHub-Ifix && HF_HUB_OFFLINE=1 PYTHONPATH=$PWD/src IDEATE_DATA_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/ideate LABELMAKER_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker IDEATE_CORPUS=/scratch/gpfs/EKOLEMEN/foundation_model /scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/ideate-cpu/bin/python -m pytest tests/ideate -q -W error` (778 passed at HEAD; `PYTHONPATH=$PWD/src` FIRST so this worktree shadows the installed main checkout).
Context: the independent critic (Codex gpt-6-astra) scored iteration 0 at 4.0/10. Its findings for your defects are reproduced VERBATIM below; fix exactly those with the tests the critic asks for. Full report (read-only): /scratch/gpfs/nc1514/FusionAIHub/.superpowers/sdd/critic-iteration-0-report.md.
Data (read-only unless stated): `$IDEATE_DATA_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/ideate` (`db/`, `frame_codes/*.pt` 500 production caches, `gates/g_enc.json` + gate caches, `runs/encode/*.json` run manifests naming which shots each CPU task encoded; the 10 GPU pilot caches are 185786 … 186036 — the encode_cpu.sbatch header names them). Shipped reference caches: `/scratch/gpfs/EKOLEMEN/nc1514/shot-recommender/models/IGNITE/frame_codes/{190090,202537,204346}.pt`. Write nothing under /scratch/gpfs/nc1514 except repo source; tests write only to `tmp_path`. You MAY write under `$IDEATE_DATA_ROOT` ONLY for: the frame-code provenance sidecars (item 3) and a `labels join` re-publication (item 2c) — state exactly what you wrote in the report. Do not re-encode anything on the GPU; do not submit jobs; do not modify labelmaker code (another agent is adding `events/<shot>_sources.parquet` there — build against the CONTRACT below with synthetic fixtures).
NOTE: another agent (task I9a) is concurrently adding `retrieval/phenomena.py`, schema types `Interval/PhenomenonHit/EventRef`, and `ShotDB` loading of events/labels_wide/text_claims in a different worktree. Avoid those files where you can; if you must touch `shotdb/store.py` or `schema.py`, keep the change minimal and additive.

## Deliverables
1. **G-ENC strictness** (defect 3; `scripts/ideate/g_enc.py:145,185`, `src/ideate/design/seed.py:145`, `tests/ideate/test_seed.py:258`): (a) `verdict` FAILS when any REQUESTED modality is missing from either side (`equal is None` is a failure with note "not encoded", never a pass); (b) `encode_frame_codes` must not silently reduce the requested modality set to the available codecs — a requested modality without a codec raises (or, with an explicit `allow_partial=True` used only by diagnostics, is recorded as `missing` and makes the verdict fail); (c) `compare` validates tensor dims, frame counts (239), vocabulary bounds, dtypes (`int32` codes, `float16` 239×88 actuators) and fails on inconsistency; (d) tests that FAIL for: a missing codec/modality, one changed token, 81/88 actuator channels within tolerance, a frame-count mismatch, a dtype mismatch — each asserted against the real 202537 reference cache structure (load it read-only) or a faithful synthetic copy; (e) narrow the causal wording at `g_enc.py:83` ("demonstrated" input difference for 190735/190736 → "output disagreement; input difference not established without matched input hashes"); (f) keep `gates/g_enc.json`'s historical FAIL untouched; document in the script header that a partial diagnostic run is not the three-shot gate.
2. **MCP/DB coverage states** (defect 2b; `src/ideate/mcp/tools.py:365`, `labels/join.py`, `shotdb/store.py`): 
   (a) CONTRACT (labelmaker writes it; you consume it): `$LABELMAKER_ROOT/events/<shot>_sources.parquet` with columns exactly `shot int32, source str, status str {ok, skipped, error}, reason str, t_cov0_s float64, t_cov1_s float64, n_events int32, diag str, channel int16, pass_name str, run_id str, git_sha str, written_at str` — one row per (source, diag, channel, pass) that ran or was skipped. `ideate labels join` ingests every existing file into `db/event_sources.parquet` (same columns) and records per shot in the join summary `n_sources_ok/skipped`, `has_observed_products: bool`. Missing file → no rows (shot "unprocessed"). Provide a test fixture writer for the contract.
   (b) `get_events(shot, …)` returns FOUR explicit states in a `status` field and a caveat for each: `unindexed` (shot not in `shots.parquet` → the error dict, as `describe_shot` does), `unprocessed` (indexed, no `event_sources` rows and no non-forecast events → caveat "no observed-event product for this shot; absence is not evidence"), `uncovered` (a requested window outside every source's coverage → caveat naming the covered span), `observed` (sources ran; report `n_sources_ok`, and "0 detections inside coverage" explicitly when empty). Split the payload: `events` (detector/heuristic only), `text_mentions` (evidence_kind text — a lexicon hit, not an assertion that the phenomenon occurred; say so), `forecasts` (unchanged). Validate `t0_s < t1_s` and finiteness → error dict with caveat (never a silent empty). Tests over a synthetic DB for all four states, the split, and the reversed window.
   (c) `labels join` also refreshes `has_frame_codes` in `shots.parquet` from the `frame_codes/` directory (today 13 true / 487 false while 500 caches exist) and writes `manifest.json`'s `labels_join` block with the refreshed count; re-run it once on the real DB on the login node (57 s) and report the new flag count (expected 500/500) — this is the one allowed production write besides item 3.
   (d) Fix the server/`.mcp.json` instructions and tool docstrings: "every reply carries caveats" holds for application-level results; malformed arguments surface as MCP schema-validation errors (document it, or normalise by catching and returning the error dict — pick one and test it).
3. **Frame-code provenance** (item 7): `ideate encode` writes a sidecar `frame_codes/<shot>.json` next to each `.pt` (`device, torch_threads, torch_version, git_sha, ignite_bundle_sha_or_path, input_file_sha256 (of the corpus h5) or mtime+size if hashing 4 GB is too slow — say which, encoded_at, run_manifest`); add `scripts/ideate/frame_codes_provenance.py --backfill` that writes sidecars for the existing 500 from the run manifests under `runs/encode/*.json` (device = cuda for the 10 pilot shots, cpu for the rest; threads from the sbatch; git sha from the manifests; mark `backfilled: true`) — run it once (production write) and report counts; `ideate coverage`/`describe_shot` expose `frame_codes_device`. Tests: sidecar schema; backfill on a synthetic dir; the `.pt` payload itself is UNCHANGED (compatibility with the shipped bundle — assert the four keys).
4. **Determinism wording** (item 4): the `corpus select --finalize` output differs in `created`, `summary.from_list` and feature-store metadata while the `shots:` rows are byte-identical — make the CLI's own summary line say exactly that (rows deterministic; header metadata regenerated), and note that the feature-store inventory counts 507 frame-code shots (production 500 + non-overlapping bundled caches) — label it as such in the output.

## Rules
TDD; commits on `recommender-Ifix`, subject prefix `ideate:`, trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Report to `/scratch/gpfs/nc1514/FusionAIHub-Ifix/.superpowers/sdd/task-Ifix-C1-report.md` with the suite tail, the exact production writes you made (paths, counts), the new G-ENC failing-test names, and deviations with reasons.

## Critic findings (verbatim)
### 2. Coverage truth is lost between detectors, storage, and MCP

**Locations:** [pipeline.py:427](/scratch/gpfs/nc1514/FusionAIHub/src/labelmaker/events/pipeline.py:427), [pipeline.py:614](/scratch/gpfs/nc1514/FusionAIHub/src/labelmaker/events/pipeline.py:614), and [tools.py:365](/scratch/gpfs/nc1514/FusionAIHub/src/ideate/mcp/tools.py:365).

There are two connected failures.

**Actuator events receive the union of unrelated time axes.** `_span` takes the earliest start and latest end across all actuator inputs, then supplies that same coverage to every actuator event.

For shot 198658:

| Quantity | Independently inspected source axis | Coverage assigned through the shared actuator span |
|---|---|---|
| Gas | Approximately −10 to 94.8576 s | −10 to 94.8576 s |
| NBI | Approximately 0 to 13.1001 s | −10 to 94.8576 s |
| RMP | Approximately −1.06286 to 10.20114 s | −10 to 94.8576 s |

The suspicious **gas** span is substantially real: the gas recorder actually has that long axis. The bug is borrowing it as coverage for NBI and RMP. The endpoints also include trailing padding rather than exclusively finite observations.

**MCP cannot distinguish missing observation products from observed silence.** The real stdio call returned:

```json
{
  "shot": 198658,
  "n": 0,
  "n_forecasts": 0,
  "nan_excluded": 0,
  "caveats": [],
  "events": [],
  "forecasts": []
}
```

Yet `describe_shot(198658)` correctly says that shot is not in the 500-shot database. For the selected set, the join reports **500 missing observed-event products**. Merely finding a global `events.parquet` does not establish that anybody examined a particular shot or diagnostic.

**Required fix:** Compute coverage per source and quantity, using the required-input intersection where appropriate. Persist completion and coverage even when a detector emits zero events. Have MCP distinguish unindexed, unprocessed, uncovered, and observed-with-no-detection states. Validate finite, ordered time windows. Correct the API documentation’s claim that every nonforecast row describes what a diagnostic showed: text rows do not satisfy that description.

### 3. G-ENC passes when a required modality is omitted

**Locations:** [g_enc.py:145](/scratch/gpfs/nc1514/FusionAIHub/scripts/ideate/g_enc.py:145), [g_enc.py:185](/scratch/gpfs/nc1514/FusionAIHub/scripts/ideate/g_enc.py:185), [seed.py:145](/scratch/gpfs/nc1514/FusionAIHub/src/ideate/design/seed.py:145), and [test_seed.py:258](/scratch/gpfs/nc1514/FusionAIHub/tests/ideate/test_seed.py:258).

A missing modality gets `equal=None`; `verdict` rejects only `equal is False`. The encoder also reduces the requested names to whatever codecs are available, then checks completeness against that reduced dictionary.

**Fresh controlled reproduction using the real 202537 reference:**

```text
Remove ece from an otherwise identical cache.

compare(...).modalities.ece:
    equal: None
    agreement: None
    note: "not encoded"

verdict(...):
    True
```

The comment claiming an unexpectedly missing requested modality cannot reach this code is therefore unsafe. The loader can return a partial codec collection, and the seed path accepts that reduction.

The numerical constants remain **0.002 z and 82/88**; they were not relaxed. Nevertheless, omission weakens the effective gate. The saved full report correctly remains failed.

**Required fix:** Require every requested modality and validate actual tensor dimensions, frame counts, vocabularies, and dtypes. Distinguish partial diagnostic runs from the full three-shot gate. Add tests that fail for a missing codec/modality, one changed token, fewer than 82 passing actuator channels, and inconsistent tensor/frame metadata. Preserve the historical full-gate failure until it is resolved or superseded by a separately justified acceptance criterion.


### Report items 5–8 (verbatim)
### 5. IDEATE database — artifact audit plus fresh in-memory join

Manifest:

```text
git_sha:      58e664f
shot_source:  list:recommender_v1
n_requested:  500
n_shots:      500
n_segments:   1977
failed:       {}
```

Coverage inspected in `record_json`:

| Features | Present | Unavailable |
|---|---:|---:|
| ip, bt, betan, kappa, qmin, li, pcbcoil, tritop, tribot, aminor, volume, gapin | 500 each | 0 |
| ne0 | 489 | 11 |
| te0 | 489 | 11 |
| r0 | 499 | 1 |

The unavailable inputs remain missing rather than becoming zero. Their source files record `TreeNODATA`; for example, the unavailable density/temperature inputs on 186389 and `r0` on 203905. Current records contain no pending values. The test suite separately exercises retryable failure versus permanent unavailability; the absence of pending rows today alone would not prove that distinction.

Joined tables:

```text
labels_wide: 16800
thr_source:
    config 3920
    ""    12880
    card      0

events: 1037
evidence_kind=forecast: 1037
finite horizon_s:       1037

text_claims: 13445
polarity:
    pos       12030
    neg         499
    uncertain   916
temporality:
    planned    9460
    observed   3934
    historical   51
scope:
    run       13199
    shot        246
```

Fresh `card_thresholds()` returned an empty mapping: no operating thresholds were recovered from cards. Reporting configuration thresholds as `config` is correct.

I also ran the actual join over all 500 shots **in memory**, without calling its writers:

```text
labels_wide 16800: values reproduce
events       1037: values reproduce
text_claims 13445: values reproduce
elapsed: 59.13 s
```

Equality excluded regenerated execution provenance (`run_id`, `git_sha`, `written_at`) and normalized parquet-related dtype differences.

Two material product qualifications:

```text
n_shots_with_events:      0
n_shots_missing_events: 500
```

Also, the database’s `has_frame_codes` flag is stale: **13 true / 487 false**, despite all 500 production caches now existing. The completed encode must be reflected in a refreshed database publication.

### 6. G-ENC — artifact comparison, exit-path replay, and fresh CPU encode

The saved report says:

```text
passed: false
failed_shots: [190090, 204346]
include_video: true
actuator_tolerance_z: 0.002
actuator_min_pass: 82
```

I recomputed comparisons directly from saved gate caches and the shipped bundle:

| Shot/modality | Mismatched tokens | Affected frames | Maximum mismatches/frame | Agreement |
|---|---:|---:|---:|---:|
| 190090 tangtv_lower | 145 | 63 | 14 | 99.4382% |
| 190090 tangtv_upper | 161 | 45 | 31 | 99.3763% |
| 204346 ece | 4 | 4 | 1 | 99.9913% |
| 204346 mhr | 9 | 9 | 1 | 99.9804% |
| 204346 co2 | 367 | 88 | 49 | 99.2002% |

All 14 modalities match for 202537. All three shots have **88/88 bit-identical actuator channels**, with maximum delta **0**.

I exercised the actual `main`/`compare`/`verdict` exit path using the saved caches in place of re-encoding and a temporary report destination:

```text
CACHED-ARTIFACT REPLAY
190090 FAIL
202537 PASS
204346 FAIL
G-ENC FAIL
replayed main exit status: 1
```

That confirms the exit behavior on these artifacts; it is not a new GPU encoding run.

I also performed the optional **fresh CPU encode**, with outputs under `/tmp`:

```bash
python scripts/ideate/g_enc.py \
  --shots 202537 --device cpu \
  --out-dir /tmp/.../codes --json /tmp/.../g_enc.json
```

```text
202537: 239 frames; 39.1 s; PASS
14/14 modalities bit-identical
88/88 actuators bit-identical
exit status: 0
```

This one-shot smoke result does not satisfy the three-shot gate.

A supplemental direct-PyTorch probe initially encountered the system `libstdc++`/`GLIBCXX_3.4.29` mismatch; the cached replay ran with the IDEATE environment’s library directory supplied. That import failure is not counted as a gate verdict.

**CPU/device reproducibility:** Independently comparing the preserved caches reproduced:

```text
185786:
    CUDA == CPU(4 threads) == CPU(8 threads), all 14 modalities

185955:
    CUDA vs CPU(4): bes 1 token; mhr 4 tokens
    CUDA vs CPU(8): bes 4 tokens; mhr 4 tokens
    CPU(4) vs CPU(8): bes 3 tokens
    actuator differences: 0
```

These are artifact comparisons of prior experiments. They refute unconditional bit-reproducibility across device/thread settings. Two shots do not establish a corpus-wide maximum disagreement.

**Attribution:** The revised text correctly admits that production inputs were never compared, and no longer rules out arithmetic differences for 204346. However, [g_enc.py:83](/scratch/gpfs/nc1514/FusionAIHub/scripts/ideate/g_enc.py:83) still says an input difference is “demonstrated” for 190735/190736. I reproduced their 78/88 and 77/88 actuator agreements and approximately 0.04%/0.10% upper-video agreements. Those demonstrate output disagreement. Without matched input hashes and preprocessing provenance, they do not uniquely prove different input files. That causal wording still needs narrowing.

**Trailing-NaN compatibility:** `_record_length` deliberately reproduces production’s zero-averaged trailing pad. This is prominently documented at [actuators.py:42](/scratch/gpfs/nc1514/FusionAIHub/src/ideate/design/actuators.py:42) and pinned by tests. The distinction between compatibility inputs and physically valid sample means is explicit. This is a justified compatibility choice, not an undisclosed missing-data imputation.

The saved gate JSON predates the newly added mismatch-count fields. The I8 fix report acknowledges that it was not regenerated. Current code emits those fields; the fresh CPU report did so. “The saved JSON now contains them” would still be inaccurate.

### 7. Encode product — full artifact census and load validation

I loaded **all 500** production caches, exceeding the brief’s one-file structure check:

```text
files:              500
selected shots:     500
missing:              0
extra:                0
invalid structures:   0
```

Every cache has:

- Exactly `codes`, `actuators`, `n_frames`, and `vocabs`.
- **239 frames** and **14 modalities**.
- Finite `float16` actuators of shape **239 × 88**.
- Valid `int32` token arrays with the expected frame counts, widths, and vocabulary bounds.

The structure matches the shipped 190090 bundle cache.

The product is device-mixed: **10 GPU pilot caches plus 490 CPU caches**. This is disclosed in the ledger and `encode_cpu.sbatch`. It is not adequately exposed through the four-key cache payload or database provenance. The payload does not record device, thread settings, code revision, or input hashes.

Keep those attributes in a compatible sidecar or manifest. A consistent backend may help, but the measured thread sensitivity means “all CPU” alone does not establish reproducibility.

### 8. MCP stdio server — fresh real protocol exercise

I launched the actual `python -m ideate.mcp` subprocess with the real data root, using the repository’s `mcp.client.stdio` client pattern.

```text
tools:
    search_shots
    describe_shot
    get_events
resource:
    ideate://manifest
```

The resource exposed the real manifest.

Observed calls:

| Call | Result |
|---|---|
| `describe_shot(198658)` | Application error dictionary: shot absent from database |
| `get_events(198658)` | Empty events/forecasts, **no caveat** |
| `describe_shot(185955)` | Real database description |
| `get_events(185955)` | 0 events, 37 forecasts, explicit forecast caveat |
| Invalid segment value | Application error dictionary, no transport failure |
| Invalid shot argument type | MCP schema-validation error; no application `caveats` field |
| Reversed time window | Successful empty result without validation caveat |

Forecast separation works. Universal coverage honesty does not.

The documentation’s “every reply” caveat promise also exceeds the implementation: application errors are handled, but malformed schema arguments follow framework error handling. That distinction should be documented or deliberately normalized.

This verifies real stdio transport. I did **not** attach the server from a live Claude Code client, so I do not claim that exact exit criterion was independently exercised.

