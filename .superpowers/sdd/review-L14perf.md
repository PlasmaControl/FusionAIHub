# Review — L14-perf (`recommender-L14perf` @ fa63f7a, base 67f47e3)

Reviewer: read-only pass in `/scratch/gpfs/nc1514/FusionAIHub-L14perf`.
Only write: this file (uncommitted).

## Verdict: **FIX-THEN-MERGE**

The engineering is real and the evidence is honest. Every headline number in
the report reconciles with the files on disk (I re-hashed four of them and
re-read the jobstats/sacct captures byte for byte); the identity contract is
genuinely pinned by hermetic tests that compare NPZ file bytes and the exact
events/sources frames including `intervals`/`min_gap_s`; the CUDA
forward-boundary finding is a real discovery, correctly diagnosed and
correctly acted on rather than tolerated; the jobstats FAIL is reported as a
FAIL; and no production store, no `docs/superpowers/plans/**` and nothing
outside `runs/l14perf/` + the prescribed `runs/slurm/` captures was written.

Two things stop it being a straight MERGE:

* **F1** is a live crash path on the production device: a transfer OOM on the
  CUDA path is "handled" by retrying the identical failed allocation with no
  guard, so the second failure escapes the generator and kills the whole
  array task instead of one block.
* **F2**: the production sizing proposal — the one artefact the controller
  would act on — is contradicted by the task's own measurements in two
  independent places (prep workers and tail workers are both *below* measured
  demand), so acting on it as written would very likely reproduce the GPU-util
  collapse this task existed to fix.

Neither is deep. F1 is a `try`/`except` plus a fallback; F2 is a corrected
table and an explicit "second pilot required" in the report.

---

## Findings, by severity

### F1 — BLOCKING. Transfer OOM on the CUDA path is retried at the same size and uncaught

`src/labelmaker/events/masks.py:628-632`

```python
if reference is not None:
    if copied is None:
        # A full input allocation failed. Preserve group boundaries
        # even in this exceptional path; one input group is small
        # relative to the network's activations.
        copied, ready = _copy_batch(host, device, copy_stream)
```

`copied is None` can only be reached because `stage()`
(`masks.py:595-600`) swallowed a `torch.cuda.OutOfMemoryError` from
`_copy_batch(host, ...)`. The recovery then calls `_copy_batch` again **with
the same `host`**, i.e. requests the identical allocation that just failed,
with no `torch.cuda.empty_cache()` in between and no `try`. The comment's
reasoning ("one input group is small") describes *forward* groups, but the
call transfers the whole pooled batch, so nothing has been made smaller.

The second `OutOfMemoryError` propagates out of `infer_pooled`, past the
`finally`, into `_run_pooled`'s `output = next(inferred, None)`
(`driver.py:964`), which is not inside a `try`, and out of `run_shots` — one
transient allocation failure destroys every shot in the chunk, including the
ones already inferred but not yet flushed. The legacy path halved and
recovered per block.

This is exactly the branch `test_pooled_oom_retries_transfer_and_forward`
(`tests/labelmaker/test_l14perf_identity.py:181`) does **not** reach: it runs
on `"cpu"`, where `preserve_batches` defaults to `False`, so it exercises the
`while offset < len(host)` slice-and-halve path at `masks.py:642-671`, which
is correct. The reference path has no OOM test at all.

**Fix.** In the `copied is None` branch: `torch.cuda.empty_cache()`, retry
once inside `try/except torch.cuda.OutOfMemoryError`; on a second failure do
not raise — set `state.error = OutOfMemoryError(...)` on every `state` in
`spans`, `continue` to the next batch, and let `collect()` yield those blocks
as errors. That keeps the per-block isolation contract the module's docstring
claims ("Prep and tail failures are isolated"). A stronger variant, if you
want the boundaries preserved *and* a smaller allocation: copy each forward
group separately (`_copy_batch(host[begin:begin+count], ...)`) and feed
`consume` per group. Add the missing regression with
`preserve_batches=True` and a `_copy_batch` stub that fails the first N calls.

### F2 — BLOCKING (report, not code). The production proposal contradicts its own measurements

`.superpowers/sdd/task-L14perf-report.md:~390-410`

The proposal is **10 CPUs / 31G / 00:04:00**, with "a matching worker
configuration would be 6 prep, prefetch 6, 3 tail, 1 parent". The CPU, memory
and time numbers are all derived from the pilot, which ran **37 prep /
prefetch 37 / 12 tail** (21 and 8 live). You cannot hold a run's measured
aggregate demand fixed and simultaneously cut the pools that produced it. The
task's own evidence says both pool sizes are too small:

1. **Tail: 3 workers is below measured demand, arithmetically.**
   `pilot-l14perf/runs/events/tokeye-2931999_c0of1_r0.json` gives
   `describe_s = 27.772` + `finish_s = 134.626` = **162.40 tail worker-seconds**
   against a **53.49 s** driver wall → **3.04 tail cores at 100 % efficiency**.
   Three workers cannot keep up even in the ideal case, and `tail_wait_s` was
   already **8.344 s** *with twelve* configured. When the tails fall behind,
   `dispatch_ready`'s bound `len(pending) >= max(2, 2 * tail_workers)`
   (`driver.py:940`) drops to 6 and `flush()` blocks the GPU-feeding loop.
2. **Prep: 6 workers is below every estimate.** The AFTER profile measured
   `prep_work_s = 60.277594` for 1,170 tiles = 0.0515 core-s/tile; at the
   pilot's 139.81 tiles/s that is **7.2 prep cores**. The report itself says
   the AFTER profile "exposed waits at six prep workers" (`prep_wait_s` went
   **6.64 → 11.69 s** on a V100S) — and the A100 is ~2.9× faster, so it needs
   tiles ~2.9× sooner. The pilot's `prep_wait_s = 0.002` was bought with 21
   live workers; proposing 6 discards precisely the knob that produced the
   result.
3. **Memory is not derivable at the new size.** 23.601597 GiB MaxRSS was
   measured at 37/12 with a 24-shot pending bound. At 6/3 the bound is 6 and
   there are 15 fewer prep workers, so real use will be well under 31G and the
   CPU-memory gate will fail from the *other* side. (The ×1.3 arithmetic is
   right — `23.601597 × 1.3 = 30.68 → 31G` — it is the transferability that
   is wrong.)
4. **The time budget is self-defeating.** `00:04:00` per chunk with the
   sbatch's own `--timeout 240` means **one** hung shot consumes the entire
   wall and the chunk dies with nothing written. Even ignoring that, 4 min is
   `61/20*60*1.3` computed from a run at 21/8 workers; at 6/3 the wall will be
   longer, not the same.

Answering the brief's sub-questions directly: **(i) yes**, GPU-util collapse at
6 prep / 3 tail is the likely outcome, and the AFTER profile is the direct
evidence. **(ii) yes**, a second ≤20-shot pilot is warranted and is the cheap,
rule-compliant way to get there — a ≤20-shot pilot is exempt-but-reported, so
it costs nothing against the "all four ≥70 %" production rule, and right now
*neither* the tested configuration (CPU 12.8 %, CPU-mem 32.9 %) nor the
proposed one is production-ready. **(iii)** the memory *derivation* (sacct
MaxRSS × 1.3, with the explicit statement that per-worker VmHWM is not
additive because of shared pages) is sound as a method and the report deserves
credit for saying so out loud; it is the *transfer* to a different pool size
that is unsound. **(iv)** the time budget is unsound for the two reasons above.

**My recommended production request** (to be *validated by one more ≤20-shot
pilot at exactly these numbers* before the array is submitted):

| Knob | Value | One-line justification |
|---|---:|---|
| `--cpus-per-task` | **12** | The sbatch's own `PREP + TAIL + 1` rule; predicted efficiency `454.406 TotalCPU s / 53.49 s wall / 12 = 71 %`, which is measured demand, not a shrunk allocation. |
| `PREP_WORKERS` | **7** | Prep demand is bracketed 4.5–7.2 cores by two disagreeing measurements (pilot TotalCPU residual vs AFTER `prep_work_s`); take the upper bracket rounded down, and let the pilot settle it. |
| `PREFETCH` | **8** | Must be ≥ `PREP_WORKERS` (the script refuses otherwise) and is also the concurrency; one slot of slack keeps all seven busy across a slow read. |
| `TAIL_WORKERS` | **4** | Measured tail work is 3.04 cores (`27.772 + 134.626` worker-s over a 53.49 s wall); 3 is below demand, 4 gives ~30 % headroom. Requires `--no-index`, which the sbatch already passes. |
| `--mem` | **32G** | `23.601597 GiB × 1.3 = 30.68` rounded up; known to be an over-estimate at 7/4 workers — re-measure in the pilot, do not shrink it on a guess. |
| `--time` | **00:12:00** | 60 shots × 2.67 s/shot ≈ 160 s + ~10 s startup, ×1.3 = 221 s — but `--timeout 240` lets one hung shot alone burn 4 min, so the wall must cover startup + one timeout + the rest. Wall over-request is not gated by jobstats; an under-request loses the chunk. |
| GPU / batch / AMP | **1 A100, 96, on** | Unchanged and measured: 139.81 tiles/s, GPU 100 %, peak alloc 18.212 GiB. |
| Array | **`--array 0-7%2`, `N_CHUNKS=8`** | As briefed; unchanged. |

I am explicitly *not* tuning any of these to make a gate pass: 12 CPUs is what
the measured 8.50 core-seconds-per-wall-second needs, and 32G is the measured
MaxRSS with the standard margin. If the second pilot shows real memory at
7/4 is 12 GiB, request 16G then — because that is what it measures, not
because 16G scores better.

### F3 — One shot's timeout can poison another shot's blocks (CPU path)

`src/labelmaker/events/masks.py:651-680`, `src/labelmaker/events/driver.py:838-840, 981`

`forward_context=lambda tokens: guard([token[0] for token in tokens])` and
`guard` takes `min(remaining(state) for state in states)` and raises
`StageTimeout` for the *whole batch* if any one shot has expired. In the
non-reference path the batch spans blocks from several shots, and
`masks.py:679-680` then does `if error is not None: state.error = error` for
**every** span in the batch. So shot *i* running out of time marks shot *i+1*'s
block as failed, and the alarm is armed with shot *i*'s deadline while shot
*i+1*'s forward runs.

Production is CUDA, where `reference.consume` calls
`forward_context([state.token])` with a single token and then sets
`offset = len(host)` so the multi-token loop never executes — so this is
`--device cpu` only today. But the CPU path is what the identity tests and
`process_shot` comparisons run on, and `test_pooled_timeout_is_an_error_and_next_shot_runs`
(`test_l14perf_identity.py:296`) uses `tile_batch=1`, which puts one block per
batch and so never reaches the shared-batch case.

**Fix.** Attribute the failure per span: when the exception is a
`StageTimeout`, only mark the states whose own `remaining(state) <= 0`; leave
the rest of the batch to be retried or completed. Add a regression with two
shots, `tile_batch=4` and a model that sleeps past shot *i*'s deadline.

### F4 — `size` never resets after an OOM halving in the pooled non-reference path

`src/labelmaker/events/masks.py:666-667`

`size = max(1, size // 2)` is a generator-lifetime variable: once halved it
stays halved for every remaining block, across shots. The oracle (`infer` /
`_infer_compact`) starts each block at `batch`. `_ReferenceBatches` gets this
right (`self.size = self.batch` when a new block starts,
`masks.py:~505`), so it is the CPU/non-preserving path that drifts. Harmless
for output only because the report claims CPU batching is exact; it is still a
silent, unbounded throughput loss after one transient. Reset `size` when a
block boundary is crossed, as the reference path does.

### F5 — Identity is conditional on OOM not occurring, and OOM demonstrably occurs

The task established the right result — "cuDNN AMP rounding changes when a
tile moves to another batch shape (and can cross the stored threshold)" — and
then enforced oracle forward boundaries for exactly that reason. But the OOM
halving path *changes the forward batch shape by construction*, and both
profiles record it happening: "One forward attempted too large a V100S
allocation and used the existing halving retry" (BEFORE) and "The AFTER run
used the existing halving fallback for V100S OOM attempts". The oracle and the
pooled run need not OOM at the same points, because their allocator pressure
differs.

So the true contract is *byte-identical when no OOM halving occurs, or when it
occurs identically*. That is still a strong and useful contract — the A100
pilot at 18.212 GiB peak on a 40 GB card almost certainly never halved — but
the report and `docs/LABELMAKER.md` both state it unconditionally. State the
condition; it is a limit, not a defect, and hiding it is what would make it
one. (No code change required.)

### F6 — The pooled scheduler's pool-restart paths are untested; the tests that cover restarts now test a non-default path

`tests/labelmaker/test_tokeye_masks.py` contains **zero** occurrences of
`pooled` (verified by grep). Every prep-pool-death, tail-pool-restart,
unpicklable-payload, task-timeout and worker-crash test in that ~1,300-line
file calls `driver.run_shots(...)` without `pooled=True`, so all of them
exercise the legacy scheduler — which since `main`'s `pooled=not
args.probs_on_host` (`driver.py:1600`) is **no longer what the CLI runs**.
Several would not transfer even if flipped: they monkeypatch
`pipeline.infer_block`, which `_run_pooled` never calls.

Specifically uncovered in `_run_pooled`: the `epoch` / `future.l14_epoch`
re-submission after a prep-pool restart (`driver.py:855-903`); the tail-pool
restart and payload re-queue in `flush()` (`driver.py:915-925`); the
`tail_workers > 1 and index` refusal (`driver.py:1140-1141`); and F1's branch.

The report's claim that "additional regressions cover a timed-out shot
followed by a successful shot, OOM in transfer and forward, later-block
recovery, and multiple ordered tail workers" is true as far as it goes, but it
does not mention that the pre-existing isolation suite no longer covers the
default path. Either parametrise the existing tests over `pooled` in
`_totals`, or add pooled equivalents of the three restart tests.

### F7 — A tail-pool restart can reorder `rows` out of shot order

`src/labelmaker/events/driver.py:915-925`

```python
for other, old, payload in list(pending):
    if old is not None and (not old.done() or old.cancelled()
                            or isinstance(old.exception(), BrokenExecutor)):
        pending.remove((other, old, payload))
        pending.append((other, tails.submit(payload), payload))
```

Entries are moved to the **end** of `pending`. With more than one tail worker
a mixed state (an early shot still running, a later shot already done) is
reachable, and `[B(done), C(running), D(done)]` becomes `[B, D, C]` — after
which `flush()` appends rows out of shot order, contradicting the module
docstring's "Results retain plan and shot order" and the run record's implied
ordering. (The short-circuit on `not old.done()` correctly avoids blocking in
`old.exception()`; that part is right.) Fix by rebuilding `pending` in place
— resubmit the future for the affected entries without changing their
position.

### F8 — "CUDA implies preserve_batches" is enforced but not pinned by a runnable test

`src/labelmaker/events/masks.py:~565` (`if preserve_batches is None: preserve_batches = cuda`)

The constraint *is* enforced by code, not merely described — good — and
`test_full_transfers_preserve_reference_forward_boundaries`
(`test_l14perf_identity.py:329`) pins the *mechanism* (`copies == [3,3,3]`,
`model.batches == [2,3,2,2]`). What no CI-runnable test pins is the
**default**: on a CPU-only runner every call resolves to `preserve_batches =
False`, so a future edit flipping the default to `False` unconditionally — the
single change that would silently reintroduce the AMP mismatch this task
discovered — passes the whole suite. Factor the default into a named helper
(`_preserves_reference_batches(device)`) and assert
`_preserves_reference_batches(torch.device("cuda")) is True` without needing a
card.

### F9 — `tokeye_masks.sbatch`'s README header and SBATCH defaults still describe L12

`scripts/labelmaker/tokeye_masks.sbatch:1-34`

The header still reads as L12's measured basis — "Completed cold pilot
2931065_0 … 48.97 tiles/s … GPU 6.9% … sacct MaxRSS 6750340K = 6.438 GiB;
x1.30 = 8.37 GiB -> request 9G … 161/20*60*1.30 = 627.9 s -> time 00:11:00" —
and the directives are still `--cpus-per-task=8 --mem=9G --time=00:11:00`,
with `--output=` pointing at the **production** `runs/slurm/`. The driver's
schedule and memory profile changed materially underneath it (the pending tail
backlog now holds each block's raw log-power until description), and the L14
pilot measured **23.6 GiB** MaxRSS. A bare `sbatch scripts/labelmaker/tokeye_masks.sbatch`
now submits L12's sizing to the new driver against the production root. The
header is the only in-band record of "the measured basis"; leaving it stale is
a correctness defect in the same class as a wrong comment.

Related and cheap: `tests/labelmaker/test_tokeye_sbatch.py:28` still asserts
`cpus == workers + 2` while the script now enforces `PREP_WORKERS +
TAIL_WORKERS + 1` (line 55-57). They agree only because `TAIL_WORKERS`
defaults to 1. Change the assertion to `cpus == workers + tail + 1` and parse
`TAIL_WORKERS` — otherwise a default change passes the test and aborts at
submission with exit 2.

### F10 — Substantial operating knowledge was deleted rather than moved

Two places:

* `src/labelmaker/events/driver.py:1-33`. The module docstring lost ~100 lines
  of *why*: the `events_index.parquet` lost-update analysis that motivates
  `--no-index`, the `text/logs_subset.jsonl` pre-pass argument (including why
  `readonly` changes what is *written*, not just read), the `SIGALRM`
  isolation semantics, the memory arithmetic (131 MB per `(512, T)` array;
  `--prefetch 4` ≈ 1.05 GB), and the measured "~3-6 % more CPU per channel"
  note on the job unit. The replacement is accurate but is a summary; in a
  codebase whose docstrings are its design record (see the top of
  `masks.py`, or `PrepPool`'s), this is a real loss.
* `docs/LABELMAKER.md:473-548`. The runnable pilot workflow is gone — the
  `tokeye_text_subset.sh` pre-pass, the array submission, the
  `--dependency=afterok` rebuild+gate via `tokeye_masks_afterok.sbatch`, and
  the note that **compute nodes have no `/usr/local/bin/jobstats`** so the
  pre-pass stages the site client into `runs/slurm/jobstats-client/`. All
  three scripts still exist in `scripts/labelmaker/`. What replaced it is a
  record of *this* pilot's already-executed gate command. An operator reading
  the section can no longer run the job.

Everything else in the docs section I checked is accurate to the code:
`PREP_WORKERS + TAIL_WORKERS + 1`, the `ROOT`-over-`LABELMAKER_ROOT`
precedence, `RUNTIME_ROOT`/`PHASE3_PYTHON`/`UNET` independence, the
resolved-root print, and the compact/sparse-probability description. Restore
the workflow block and keep the new paragraphs alongside it.

### F11 — `dict(locals())` as the option carrier

`src/labelmaker/events/driver.py:1142-1145`

```python
options = dict(locals())
options.pop('shots')
return _run_pooled(shots, **options)
```

At that point `locals()` also contains `bad` (the validation list), which is
passed into `_run_pooled(**opts)` and silently absorbed. It works, but it
makes the parameter surface of `_run_pooled` invisible and will pick up any
local a future edit introduces above this line. Build the dict explicitly, or
promote it to a small frozen dataclass — `_run_pooled` already reads it as
`opts["..."]` throughout, so the change is mechanical.

### F12 — Bare `assert` in a production hot path

`src/labelmaker/events/masks.py` — `_ReferenceBatches.consume` has
`assert state is self.state` and `assert last and not available`. These encode
the invariant that makes the forward-boundary reconstruction correct, and they
vanish under `python -O`. Raise a real exception (or at minimum note why an
`assert` is acceptable here, as the rest of this codebase tends to).

### F13 — Minor report inaccuracies

* The ideate suite is quoted as **132.69 s** in the "Pre-profile implementation
  gate" paragraph and **126.47 s** in the final block;
  `runs/l14perf/evidence/suite-ideate-final.log` says `1252 passed in
  126.47s`. Different runs, but the reader cannot tell.
* "The requested CPU checks did not expose CUDA AMP batch-shape rounding" —
  worth adding that the same class of risk applies to the **CPU** cross-block
  forward pooling that remains enabled (`preserve_batches = cuda`). oneDNN
  convolution reduction order can also depend on batch shape once
  `OMP_NUM_THREADS > 1`, and every identity run was done with
  `OMP_NUM_THREADS=1`. The CPU-exactness claim is empirical, single-threaded
  and single-checkpoint. Either say so, or set `preserve_batches=True`
  unconditionally — the measured win came from pooled *transfers* and device
  compaction (D2H 2.454 GB → 0.108 GB), not from cross-block forwards.

---

## Answers to the brief's questions

### A. Output identity — **substantiated, with one stated condition (F5)**

* **What the tests actually assert.** `test_output_identity`
  (`test_l14perf_identity.py:17-90`) compares `masks_file(...).read_bytes()`
  against the oracle's — actual NPZ **file bytes**, not array-by-array — and
  `pd.testing.assert_frame_equal(..., check_exact=True)` on both the events and
  sources frames after dropping only `run_id`/`written_at`, with an explicit
  `{"intervals", "min_gap_s"} <= set(actual.columns)` guard so the C3fix
  columns cannot silently vanish. It runs 12 cases: `{compact, pooled, tail,
  prep} × {(0,1), (1,4), (2,2)}`, both passes, 14 blocks. This is the contract
  the RESUME addendum demanded, unweakened.
* **Hermetic?** Yes. Synthetic `synth_shot` fixture, `PaintedNet`, `tmp_path`
  roots, `FAKE_SHA`; no data root, no checkpoint, no network. Runtime 146 s for
  the whole subset I ran. The real-shot test is correctly quarantined behind
  `L14PERF_REAL_ROOT` and additionally asserts the root is under
  `runs/l14perf` before writing (`test_l14perf_real_identity.py:47`) — a good
  guard I'd keep.
* **Would a wrong compaction order fail?** Threshold and packbits order: yes,
  pinned directly. `test_device_overlap_and_compaction_are_exact` seeds three
  rows at `nextafter(thr, 0)`, `thr`, `nextafter(thr, 1)` and compares
  `result.coh_packed` to `masks.pack(expected >= thr)`, `row_lit` to
  `coh.mean(axis=1).astype(np.float32)` and `col_act` to
  `tra.mean(axis=0)` — so a `>` for `>=`, a little-endian bit order, or a
  float32 row-fraction accumulator all fail loudly.
  Float64-vs-float32 *accumulation* is a different matter: with `TILE=512` and
  `STRIDE=448`, no column is covered by more than two tiles, and for a
  two-element sum `round_f32(a+b)/2 == round_f32((a+b)/2)` because the halving
  is an exact power-of-two scaling — so float32 accumulation would be
  indistinguishable here and no test could catch it. That is not a gap: the
  float64 `DeviceStitch.total` is belt-and-braces and matches the oracle's
  documented intent. Worth saying in the report, because "the test proves the
  float64 order" is not quite what happened.
  I traced the rest by hand and it is right: `total.add_(pred[i].float())`
  promotes float32→float64 exactly as NumPy's `+=` into a float64 array does,
  in the same ascending-tile order; `(total/count).float()` matches
  `.astype(np.float32)`; `lit.reshape(-1, 8)` is safe because `N_BINS = 512`
  makes `512 * n_cols` always a multiple of 8; `torch.sum` on `uint8` promotes
  to int64 so the `(bits << shifts).sum(dim=1)` cannot wrap.
* **Is the CUDA forward-group constraint enforced or only described?**
  Enforced in code (`preserve_batches = cuda` when `None`, and
  `_ReferenceBatches` reconstructs the oracle's groups from pooled transfers),
  and the mechanism is pinned by
  `test_full_transfers_preserve_reference_forward_boundaries`. The **default**
  is not pinned — see **F8**.
* **Is the sparse-probability ruling faithful?** Yes, and I checked
  `tracks.descriptors` rather than taking the report's word. It reads `prob`
  only at `rows, cols` (`tracks.py:388`, `tracks.py:396`), which come from
  `np.concatenate([c.rows for c in group])` over `tracks.components(coh_mask)`;
  `components` (`tracks.py:145-178`) labels the mask with `ndimage.label` and
  a 3×3 structure and returns `np.nonzero(blob)` — it never dilates, closes or
  otherwise grows beyond lit pixels. So `f_centroid_khz` (probability-weighted,
  `p @ f_pix / p_sum`), `mean_prob` (`p.mean()`) and `conf`
  (`np.percentile(p, CONF_PCT)`) are all exactly reconstructible from the lit
  pixels alone. `bandwidth`, `duty`, the power-weighted per-column centroid and
  the chirp use `raw_logpow` and geometry, not `prob`. Dropping transient
  floats is likewise sound: `describe_block` no longer calls
  `transients.column_activity(probs[1])` but uses `compact.col_act`, and
  `ACTIVITY_THR = PROB_THRESHOLD` (`transients.py:42`) makes those identical
  computations. One latent coupling worth a one-line comment or an assert at
  the substitution site (`pipeline.py:822-823`): if `ACTIVITY_THR` ever
  diverges from `PROB_THRESHOLD`, the legacy path follows it and the compact
  path silently does not.

### B. Correctness risks in the driver

* **Pinned memory / copy stream / `record_stream`** — correct. `copied` is
  allocated on `copy_stream` and `record_stream(compute)`'d after
  `compute.wait_event(ready)`, which is the documented ownership hand-off;
  `collect()` does the same for `stitch.total`/`stitch.count` on
  `result_stream`; the `finally` synchronises both auxiliary streams on early
  close. Pinned-host lifetime is safe via PyTorch's caching host allocator
  (which records the copy event on the source block for a pinned
  `non_blocking` H2D), and the `finally` closes the residual case.
* **OOM halving on both transfer and forward** — forward is correct on both
  paths; transfer is correct only on the non-reference path. See **F1** and
  **F4**.
* **Timeout of one shot not poisoning the pool** — correct on CUDA, broken on
  CPU. See **F3**.
* **Tail-pool restart preserving ordered payloads** — the payloads *are*
  preserved (the re-queue resubmits the retained `FinishJob`, which carries
  `runs` and `descriptions`, so no GPU work is redone), but the **order** is
  not guaranteed. See **F7**. Untested either way — see **F6**.
* **`--no-index` with >1 tail worker** — **refused, not silently racy**:
  `driver.py:1140-1141` raises `ValueError("multiple tail workers require
  index=False / --no-index")` before any work starts, and the sbatch always
  passes `--no-index`. The design is sound: at `tail_workers=1` the single
  worker serialises index writes, so allowing `index=True` there is right. Two
  nits: the failure surfaces as an unhandled traceback rather than an argparse
  error, and there is no test (**F6**).
* **CPU-path behaviour change** — yes, and it matters more than the report
  says. `--device cpu` under the new default takes `preserve_batches=False`,
  i.e. genuine cross-block forward pooling with batch shapes the oracle never
  used. Identity there is an empirical, single-threaded, single-checkpoint
  result (**F13**), and the CPU path is simultaneously where F3 and F4 live.
  Given the measured win came from transfers and device compaction rather than
  from cross-block forwards, I'd default `preserve_batches=True` everywhere
  and keep the CPU pooling behind an opt-in.
* **Does `--probs-on-host` really restore the legacy schedule?** Yes. It sets
  `pooled=False`, which returns control to the pre-existing `run_shots` body,
  untouched by this diff except for the `prep_seconds` field on
  `PreparedBlock`; `pipeline.infer_block` → `masks.infer` with `compact=False`
  is the original code; `describe_block`'s compact branch is skipped because
  `probs` is not a `CompactMask`. It is a genuine escape hatch.

### C. Jobstats honesty — **clean**

I read `runs/slurm/2931999_0.jobstats.txt` and `2931999_0.sacct.txt` directly.

| Claim | File says | Verdict |
|---|---|---|
| CPU 12.8 % | `stellar-m01g5: 00:06:30/00:50:50 (efficiency=12.8%)` | exact |
| CPU-mem 32.9 % | `22.4GB/68GB` (bar shows 33 %) | exact |
| GPU 100 % | `stellar-m01g5 (GPU 1): 100%` | exact |
| GPU-mem 96.6 % | `38.6GB/40GB (96.6%)` | exact |
| MaxRSS 24,748,068K | `2931999_0.0 … 24748068K` | exact (23.6016 GiB) |
| TotalCPU 454.406 s | `07:34.406` | exact |
| 61 s wall, COMPLETED, 0:0 | `00:01:01 | 0:0` | exact |
| tiles/s 139.81, wall 53.49 | run JSON `tiles_per_s: 139.81`, `elapsed_s: 53.49` | exact |
| SHA-256 × 4 | re-hashed jobstats.txt, sacct.txt, completion-evidence.json, both profile.json | all four match |

No threshold, sampling interval or allocation was manipulated: the pilot kept
its original 50 CPUs / 68G, the gate ran with `--pilot` exactly as prescribed,
and the report states plainly that the **exit 0 comes from the exemption, not
from passing** — "The full jobstats verdict remains FAIL (exempt): CPU 12.8%
and CPU-memory 32.9% are below 70%". That is the correct description. The
report also volunteers three things it did not have to: that the jobstats
summary bars (12/33/100/97) differ from the detailed values it tabulates, that
sampled CPU time (390 s) and sacct TotalCPU (454.406 s) disagree, and that
"100 %" over a 61-second run does not establish continuous utilisation on a
long one. The CPU proposal is derived from the *larger* of the two CPU-time
measurements. This section is a model of what the rule asks for.

### D. Production sizing — see **F2** and the recommended table above.

### E. Scope & rules — **clean**

* Files changed vs 67f47e3: `docs/LABELMAKER.md`,
  `scripts/labelmaker/tokeye_masks.sbatch`,
  `src/labelmaker/events/{driver,masks,pipeline}.py`,
  `tests/labelmaker/{test_l14perf_identity,test_l14perf_real_identity,test_tokeye_sbatch}.py`
  (+1,298/−264), plus `.superpowers/sdd/{profile_l14perf.py, brief, report}`.
  All permitted. `git diff --name-only … -- docs/superpowers/` is **empty**.
* Data-root writes: `find … -newermt '2026-09-14 17:30'` outside `l14perf/`
  returns exactly `runs/slurm/{2931999_0.jobstats.txt, 2931999_0.sacct.txt,
  jobstats.json}` — the three prescribed gate captures, nothing else. The
  production `masks/`, `events/`, `features/`, `labels/` trees and
  `events_index.parquet` were last touched at 13:09–13:10, before this task's
  17:59 resume; `runs/l14perf/` contains everything else. Nothing under
  `/scratch/gpfs/EKOLEMEN/nc1514/ideate/` moved.
* No bare `pixi run` anywhere in the report's commands — every invocation
  carries `--frozen --no-install --manifest-path
  /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e <env>`, and no `.pixi`
  exists in the worktree.
* **Commits vs suites**: the last commit touching `src`/`scripts`/`tests` is
  `c707b26` at **18:44:23**; `evidence/suite-labelmaker-complete.log`
  (`1621 passed, 3 skipped in 214.74s`) and `suite-ideate.log` are timestamped
  **18:44**, and `ruff-final.log` / `suite-ideate-final.log` at **19:10** sit
  just before `fa63f7a` at **19:10:23**. The pattern is consistent with
  "capture logs, then commit", not with committing under a running suite. The
  three later commits touch only `.superpowers/**` and `docs/LABELMAKER.md`,
  so the "final labelmaker run covered all source changes" claim holds — and
  my own re-run at HEAD confirms it independently.
* Docs accurate to the code except **F9** and **F10**.

### F. Tests/suites — **re-verified independently**

Report claims labelmaker 1,621 passed / 3 skipped, ideate 1,252 passed, ruff
clean. The evidence logs say exactly that
(`suite-labelmaker-complete.log`: `1621 passed, 3 skipped in 214.74s`;
`suite-ideate-final.log`: `1252 passed in 126.47s`, with a companion
`.exit` file containing `0`; `ruff-final.log`: `All checks passed!`).

I re-ran the cheap subset myself at HEAD. `tests/labelmaker/test_events_driver.py`
does not exist, so I dropped it per the instruction and substituted
`tests/labelmaker/test_tokeye_masks.py`, which is where the driver's own tests
actually live (that substitution is what surfaced **F6**). Both my runs are
green. I did **not** run the opt-in real-identity test.

---

## Commands run

| # | Command (from `/scratch/gpfs/nc1514/FusionAIHub-L14perf`) | Exit |
|---:|---|---:|
| 1 | `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker python -m pytest tests/labelmaker/test_l14perf_identity.py tests/labelmaker/test_tokeye_sbatch.py tests/labelmaker/test_events_masks.py tests/labelmaker/test_tokeye_masks.py -q -W error -p no:cacheprovider` → **191 passed in 146.60s** | **0** |
| 2 | `/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/ruff check --no-cache src/labelmaker scripts/labelmaker tests/labelmaker` → `All checks passed!` | **0** |
| 3 | `git log --oneline 67f47e3..recommender-L14perf` / `git diff --stat` / `git diff --name-only` / `git log --format='%h %cI %s'` / `git show --stat` per commit | 0 |
| 4 | `git diff 67f47e3..recommender-L14perf -- src scripts tests docs/LABELMAKER.md` (and per-file) | 0 |
| 5 | `git diff --name-only 67f47e3..recommender-L14perf -- docs/superpowers/` → empty | 0 |
| 6 | `cat /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/2931999_0.jobstats.txt` and `…sacct.txt` | 0 |
| 7 | `sha256sum` on `2931999_0.jobstats.txt`, `2931999_0.sacct.txt`, `pilot-l14perf/completion-evidence.json`, `profile-{before,after}/profile.json` → all 5 match the report | 0 |
| 8 | `python3 -c` reads of `profile-{before,after}/profile.json` totals and the three `runs/events/*.json` totals | 0 |
| 9 | `find /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs -maxdepth 2 -newermt '2026-09-14 17:30' -not -path '*/l14perf/*'` → only the 3 gate captures | 0 |
| 10 | `find …/{events,features,masks} -newermt '2026-09-14 17:00'` → empty (no production writes) | 0 |
| 11 | `tail`/`cat` of `runs/l14perf/evidence/{suite-labelmaker-complete,suite-ideate-final,ruff-final,identity-final,cuda-unit-final}.log` and `.exit` | 0 |
| 12 | Source reads: `masks.py`, `driver.py`, `pipeline.py`, `tracks.py`, `transients.py`, `run.py`, the three new test files, `test_tokeye_masks.py`, `tokeye_masks.sbatch` | 0 |

No job was submitted, no branch checked out, no file outside this review
written, no `pixi install`, no bare `pixi run`, nothing created under
`/scratch/gpfs/nc1514` other than this file.
