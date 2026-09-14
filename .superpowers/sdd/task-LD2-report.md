# Task L-D2 — report

Worktree `/scratch/gpfs/nc1514/FusionAIHub-Lfix`, branch `recommender-LD2` from `recommender` @ 131f092.
Brief `.superpowers/sdd/task-LD2-brief.md`, plus the coordinator's mid-task addition (the iteration-0
re-critic's defect 2, "Missing Ip flat-top becomes successful QH coverage").

**Finish outcome:** the fresh 500-shot smoke reproduces **271 / 60 / 50**, but **neither
full suite is green**. Labelmaker: **1 failed, 1442 passed, 2 skipped**; ideate:
**1 failed, 929 passed**. No implementation, threshold, test, or configuration changes were
made in this finish pass. The failures and the unverified real-QH criterion are recorded below.
The only repository changes in this pass are the reviewed documentation and this report.

## The number the task turns on

```
$ python -m labelmaker.run events --rules-only \
      --shot-file $LABELMAKER_ROOT/recommender_v1.txt --root /tmp/ld2/rules500
events --rules-only: 500 ok
shots with a q-min band: qmin_elevated=60, qmin_high=50, qmin_hybrid=271
events by source: qmin_rule=465
```

**hybrid 271 / elevated 60 / high 50 — exact in the completed finish run at `167a1b1`.**
500 of 500 shots `ok`, zero shots skipped or errored, and `shots_skipping: {}`: no missing
features file, `ip`, or `qmin` in this list. There are 465 event rows over 337 shots — 335 hybrid,
73 elevated, 57 high bands; a shot can hold several bands, and 163 hold none. The stage writes
**500 event files** (163 empty), **500 source files** with 1,500 rows, and the events index.

External process wall time: **86.33 s (1m26.33s)**; user CPU **26.43 s**, system CPU **3.40 s**.
The run JSON's **80.37 s** is the sum of per-shot timings, not the process wall time.
This fresh timing supersedes the previous agent's 2m50s figure. Exact commands and independently
counted Parquet/source tables appear in "500-shot rules-only smoke" below.

## Per deliverable

**1. Features-store step in the events stage.** `pipeline.features_block(shot, paths)` opens
`paths.features_file(shot)` through `features/store.read_feature` — never a raw h5 path — and
returns canonical `ip` and `qmin` as 1-D traces plus a per-quantity miss dict.
`pipeline.rules_block` is the guarded chain around it (file → `ip` → flat-top → `qmin` → rule) and
is shared by `finish_shot` and `rules_shot`. Absent file → `skipped["features"]`; absent quantity →
`skipped["ip"]` / `skipped["qmin"]`. Sources rows: `("features", "<quantity>")` — `ok` with that
quantity's OWN finite span when it was read, `skipped` with the reason on the same key when it was
not, so one line of the table answers "what did we have" either way.

Reasons are **path-free**, which was not a style choice: `events/driver.py` and `process_shot` write
one shot's rows under different roots and `test_tokeye_masks.py` pins them equal, so a reason
carrying `<root>/features/...` made the two schedules disagree (found by that test).

**2. Ip flat-top (task L-Ip).** `heuristics.ip_flattop(t, y)` — longest contiguous run of
|Ip| > 0.9·max|Ip|, as (first sample, last sample), magnitude not sign, ties to the earlier run,
`UNKNOWN` when nothing is finite.

*The corpus has no `ip` group at all* — checked on a real shot file (`185786_processed.h5`: 32
groups, `beam_voltage … vib`, no `ip`) — so the features store is not the fallback, it is the only
source in the current implementation. Same for `qmin`. `_actuator_features` does not include
an `ip` mapping; `finish_shot` assigns `features["ip"] = store_features["ip"]`. There is no
implemented corpus-current preference or fallback to claim here.

**Real QH claims are NOT demonstrated by the available runs.** The earlier draft's statement
that `qh_proxy` now claims was unsupported. Shot 195896 successfully evaluated the proxy but
produced zero QH events. See the supplemental full-stage audit below; synthetic QH tests exercise
the wiring, but they do not establish the brief's requirement of a real-shot QH claim.

**3. q-min regime rule.** `heuristics.qmin_regimes(qmin_t, qmin_y, flattop, *, shot)` → exclusive
interval events, `qmin_hybrid` (0.95 < q ≤ 1.5) / `qmin_elevated` (1.5 < q ≤ 2) / `qmin_high`
(q > 2), inside the flat-top, ≥ 500 ms, `evidence_kind="heuristic"`, `source="qmin_rule"`,
`confidence=NaN`, `diag="qmin"`, `attrs = {qmin_min, qmin_max, efit: "efit01", thresholds}`. The
open-topped band writes `hi: null` rather than an infinity, which the strict-JSON `attrs` column has
no word for. Module constants `FLATTOP_FRAC`, `QMIN_HYBRID`, `QMIN_ELEVATED`, `QMIN_HIGH`,
`QMIN_MIN_MS`, `QMIN_BANDS`, `QMIN_EFIT`, each with its one-line justification.

Coverage of a `qmin_rule` row is the flat-top **met with** the finite q-min span
(`qmin_rule_coverage`), not the whole record: outside the flat-top the rule declines to look, and
declaring the ramp covered would turn an abstention into an observed absence. The single coverage
span does not encode internal dropouts; those split event bands. Zero events means no band met
the duration requirement, not that q-min stayed below 0.95 throughout.

Two decisions worth naming, both measured, both in the docstring:

* **The duration test is `>=`, edge to edge on the sample times.** 500 ms is exactly 25 intervals between 26 samples
  at EFIT01's 20 ms cadence, so real bands land on the boundary constantly: `>` instead of `>=` gives
  **268 / 60 / 48**. The `>=` side is what reproduces the proposal's numbers.
* **The same knife-edge costs one shot.** Shot 190509's elevated band spans 25 sample intervals but
  `t[-1] - t[0]` computes to `0.4999999999999999` in float64 and is dropped. Thirteen other bands
  across the 500 land on 0.5 exactly (or `0.5000000000000001`) and are kept. Adding a tolerance
  would make the counts 271/**61**/50 and break the pin, so the bare comparison stands and the cost
  is recorded rather than discovered later.

Three ids added to `events/lexicons.yaml` (and `lexicon.PHENOMENON_IDS`, now 15) and to
`configs/ideate/phenomena.yaml` with `events: [{source: qmin_rule, phenomenon: <band_id>}]`,
`coverage_sources: [qmin_rule]`,
`diags: [qmin]`, `requires_group: []`. Aliases: `hybrid` / `hybrid scenario` / `hybrid discharge` /
`hybrid shot`; `elevated q-min` / `elevated qmin`; `high q-min` / `high qmin`. **Two candidates left
out, with the reason in the file and a test pinning the omission**: bare `qmin` names the quantity
and cannot say which band (putting it on all three would count one mention three times), and
`reversed shear` is a claim about dq/drho that a bodily-lifted monotonic profile satisfies without
and a weakly reversed one violates with. `prior` measured through `text_weak.weak_labels` over the
500 shots' own logbooks: 0.004 / 0.000 / 0.000.

**4. Rule model folder `d3d_qmin_regime_rule` — NOT BUILT, and the reason is measured.** The brief
allows skipping it if the rule-folder runner cannot express an interval rule cheaply. It cannot,
for a reason that would have cost correctness rather than effort:

* the runner hands `predict` the RAW record only for inputs whose `features/namespace.py` kind is
  `"waveform"`. `qmin` and `ip` are `kind="scalar"`, so `InputSpec.build` resamples them onto the
  fixed 25 ms grid (`ns.GRID_S`, 240 rows, 0–5.975 s) with the archive's 50 ms boxcar for
  fdp-resolved fields (`sample_by_resolver`, `ARCHIVE_WINDOW_S`);
* **MEASURED**: running the *same* `qmin_regimes` on that grid over the 500 gives
  **293 / 71 / 54** and disagrees with the events path's band set on **45 of 500 shots** — and it
  truncates every shot longer than 5.975 s. That is a second, different definition of a label that
  already has one, published under the same three phenomenon ids;
* changing `qmin`'s namespace kind to `"waveform"` to avoid that would change how it is stored
  (`store._is_waveform` → float32) and how `d3d_tearing_time_to_event_dsm` reads it;
* separately, `registry.verify_artifacts` refuses any slug whose card records no sha256 — it is the
  only guard between the pipeline and unverified weights and fails closed by design — so a folder
  with `artifacts=()` cannot run through `run.py infer` without a carve-out in that guard.

The events are the primary product and they are intact. If the folder is wanted later, the honest
route is a rule-runner that reads records at their native rate, not a `ModelAdapter`.

**5. `run.py events --rules-only`.** `rules_stage` → `pipeline.rules_shot` per shot: `rules_block`
plus L-D1's `databases_block`, writing `events/<shot>_events.parquet`,
`events/<shot>_sources.parquet` and the index through the same writers as the full stage — no
corpus read, no U-Net, no masks (a test monkeypatches `unet.load_unet` to fail if it is touched).
Refuses to be combined with `--databases-only` (which would ingest the tables twice) and belongs to
the events stage alone. Summary counts SHOTS per band, because that is the number the rule is pinned
to. Run details above; the 500 shots wrote 500 events files (337 nonempty) and an `events_index.parquet` whose
`(source, phenomenon)` groups are exactly 271 / 60 / 50.

**6. Tests.** See the counts below. Synthetic q-min traces cover: three regimes in order inside the
flat-top and a fourth excursion outside it that yields nothing; a 400 ms band dropped and a 500 ms
band kept; a q-min exactly on a threshold falling in the band below, and 0.95 itself in no band;
NaN samples splitting a band and leaving the coverage; exclusivity; the `attrs`/`confidence`
contract and a row the events table accepts. The gate is pinned on the REAL distribution through
`tests/labelmaker/data/qmin_regimes_recommender_v1.json` (19 KB, 500 per-shot rows, written by
`scripts/labelmaker/qmin_regime_census.py`): the ungated variant fires on 497/500 (> 99 %), the
gated counts are 271/60/50, the totals are checked against the file's own per-shot rows so it cannot
be hand-edited, and the thresholds it was measured with are checked against the module's so a moved
threshold fails rather than leaving a fossil. Pipeline tests cover the flat-top off the store, the
`nbi_counter` and `qh_proxy` skips not returning, the sources rows for ok/skipped, and `--rules-only`
end-to-end on `tmp_path`.

**7. `docs/LABELMAKER.md`.** New "Rule labels from the features store" subsection: what the flat-top
is and why it is a gate and not a measurement, the bins and the ≥ 500 ms rule, the 497/500 ungated
figure, the NaN-confidence rule, EFIT01 as today's source and how a better one would substitute, the
coverage rule, and the `--rules-only` invocation (including that `--root` moves the read end too).
`qmin_rule` added to the source table.

## The coordinator's addition: defect 2, "missing Ip flat-top becomes successful QH coverage"

Two changes in `pipeline.finish_shot`, both general rather than QH-specific:

1. **A step that cannot be evaluated writes no `ran` row at all**, so its OWN source row is
   `skipped` with the reason and NaN coverage. `_qh_coverage` returns a reason when there is no mask
   block, no ELM clock, no measured `pinj_total`, no flat-top, or when the four spans have no common
   interval; `_counter_coverage` does the same for `tinj_total`/`pinj_total`/`ip`.
2. **A step that CAN be evaluated declares the intersection of every input it needed.** `qh_proxy`:
   the track blocks' hull ∩ the ELM-clock block's span ∩ the `pinj_total` span ∩ the flat-top.
   `nbi_counter`: torque ∩ power ∩ current — replacing the torque's own span, which
   `feature_spans` had put there.

`nbi_counter`'s row moves from `("nbi_counter", "", -1, "")` to `("actuator", "tinj_total", -1, "")`,
which is the key its rows are ALREADY counted under: `heuristics._actuator_event` stamps
`diag="tinj_total"` on `nbi_counter` rows and on no others. So the skipped row and the successful
row are one line of the table rather than two names for the same question, and the successful one
carries the right `n_events`.

### Shot 198658, before and after

Before (quoted from the re-critic's report, `pipeline_198658` / `pipeline_audit`):

```text
qh_flattop  skipped  NaN..NaN  0 events
  reason: ip is not a corpus group ... no flat-top interval ... proxy claims nothing
qh_proxy    ok       -0.09715200215578079..4.097149848937988  0 events
```

After — a real run of this branch over the real corpus file (7 blocks, 555 events, 839 s, CPU, wide
pass; 198658 has no features file, which is exactly the case the defect is about):

```text
  source            diag  status  t_cov0_s  t_cov1_s  n_events  reason
qh_proxy                 skipped       NaN       NaN         0  cannot evaluate the QH proxy: there is no Ip flat-top
qh_flattop               skipped       NaN       NaN         0  no Ip flat-top: `ip` is not a corpus group and the features store served none, ...
actuator   tinj_total    skipped       NaN       NaN         0  cannot evaluate counter-injection: no canonical `ip`: ...
features                 skipped       NaN       NaN         0  FileNotFoundError: no features file for this shot; the `features` stage has not run on it
qmin_rule        qmin    skipped       NaN       NaN         0  no canonical `qmin` in the features store, so there are no q-min regimes to claim
actuator   pinj_total         ok  0.000000 13.099900         1
actuator          gas         ok -10.00000 94.857498         2
actuator          rmp         ok -1.062856 10.201044         1
```

There is no longer any finite-coverage `ok` row for a phenomenon nobody could compute, so ideate
cannot read `coverage_state: observed` off this shot's QH.

Regression tests, in `tests/labelmaker/test_events_pipeline.py`:
`test_a_missing_flattop_makes_the_qh_source_itself_skipped`,
`test_with_a_flattop_the_qh_source_covers_the_intersection_of_its_inputs` (a stored current held
only over 0.2–0.6 s shrinks the row below the magnetics reference span the old code declared),
`test_counter_injection_is_skipped_or_covers_all_three_of_its_inputs`, and
`test_inputs_that_never_overlap_are_no_coverage_at_all`.

## 500-shot rules-only smoke

This is the **fresh completed run** at `167a1b1`, distinct from the previous implementation
runs. Its root is `/tmp/ld2/rules500`; its `features` entry is a symlink to
`/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/features`. The store API opens files with mode `r`.
All outputs are under `/tmp/ld2`; no corpus files, masks, U-Net, network fetches, or SLURM jobs
are involved in this smoke. The old completed root was preserved at
`/tmp/ld2/finish/rules500-before-finish` before creating the fresh root.

The flags were checked with `python -m labelmaker.run events --help`. Exact successful command
and layout (the three scratch cache directories had already been created):

```bash
cd /scratch/gpfs/nc1514/FusionAIHub-Lfix
mkdir /tmp/ld2/rules500
ln -s /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/features /tmp/ld2/rules500/features
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/src"
export TMPDIR=/tmp/ld2/finish/tmp
export XDG_CACHE_HOME=/tmp/ld2/finish/cache
export MPLCONFIGDIR=/tmp/ld2/finish/matplotlib
unset LABELMAKER_LABEL_TABLES
/usr/bin/time -f 'wall_seconds=%e\nuser_seconds=%U\nsystem_seconds=%S\nexit_status=%x' \
  -o /tmp/ld2/finish/rules500.time \
  /scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/python -u \
  -m labelmaker.run events --rules-only \
  --shot-file /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/recommender_v1.txt \
  --root /tmp/ld2/rules500 --run-id ld2-finish-rules500 \
  > /tmp/ld2/finish/rules500.log 2>&1
```

| Shot/run result | Measured value |
|---|---:|
| Selected / completed `ok` | 500 / 500 |
| Shots skipped / errored | 0 / 0 |
| Shots with any skipped step | 0 |
| Missing features file / `ip` / `qmin` | 0 / 0 / 0 |
| Shots with any q-min event / none | 337 / 163 |
| Event rows / event files / source files | 465 / 500 / 500 |
| Source rows | 1,500 |
| Process wall / user CPU / system CPU | 86.33 / 26.43 / 3.40 s |
| Sum of per-shot timings | 80.37 s |
| Process exit status | 0 |

| Phenomenon | SHOTS | Event rows |
|---|---:|---:|
| `qmin_hybrid` | **271** | 335 |
| `qmin_elevated` | **60** | 73 |
| `qmin_high` | **50** | 57 |

These shot counts agree independently between the run JSON, event Parquets and
`events_index.parquet`. All events have `source=qmin_rule` and NaN confidence. The 500 source
files all record `git_sha=167a1b1`. The real-distribution ungated **497/500 (99.4%)** assertion
is also covered by the committed census fixture's tests in the full suite; no threshold changed.

Per-source status table below counts **rows**, with an extra absent-row column to avoid treating
an unrequested step as a successful or skipped evaluation. The full key also contains
`channel=-1` and empty `pass_name` for the three recorded smoke rows.

| Source | Diagnostic | ok | skipped | error | Shots without this row |
|---|---|---:|---:|---:|---:|
| `features` | `ip` | 500 | 0 | 0 | 0 |
| `features` | `qmin` | 500 | 0 | 0 | 0 |
| `qmin_rule` | `qmin` | 500 | 0 | 0 | 0 |
| `features` (whole-file failure row) | empty | 0 | 0 | 0 | 500 |
| `actuator` (`nbi_counter`) | `tinj_total` | 0 | 0 | 0 | 500 |
| `qh_proxy` | empty | 0 | 0 | 0 | 500 |

There are no additional source rows, including no database source rows. `features` has 1,000
successful quantity rows across 500 shots, rather than an additional umbrella success row.
The Ip flat-top is computed internally without a standalone successful `ip_flattop` row.

**Scope limitation:** `rules_shot` calls `rules_block` and `databases_block`; it never calls the
actuator or QH branches in `finish_shot`. Consequently `nbi_counter` is a recorded skip on **0/500**
smoke shots, but is also evaluated on **0/500**. The absence of a skip cannot establish that it
ran. The real full-stage evidence for successful counter-injection is recorded separately below.
The smoke's `qh_proxy` claim count is **0**, because this mode does not evaluate that source.

### Three fresh smoke source samples

First, middle and last shots from the input list; coverage values are rounded to six decimals.
All nine rows have empty reasons and status `ok`.

| Shot | Source | Diagnostic | Coverage seconds | Events |
|---:|---|---|---|---:|
| 185786 | `features` | `ip` | 0.000000–5.975000 | 0 |
| 185786 | `features` | `qmin` | 0.100000–4.500000 | 0 |
| 185786 | `qmin_rule` | `qmin` | 0.550000–2.100000 | 1 |
| 195919 | `features` | `ip` | -4.133400–19.878600 | 0 |
| 195919 | `features` | `qmin` | 0.160000–6.700000 | 0 |
| 195919 | `qmin_rule` | `qmin` | 1.273600–6.560600 | 0 |
| 204925 | `features` | `ip` | -4.115509–19.896491 | 0 |
| 204925 | `features` | `qmin` | 0.100000–6.660000 | 0 |
| 204925 | `qmin_rule` | `qmin` | 0.718491–6.089491 | 1 |

The 500-shot smoke contains no skipped rows, so its skipped-row check is vacuous. To verify
critic defect 2 with missing inputs, the audit also reads three earlier full-stage source files
and directly inspects the current dependency helpers as follows.

### Supplemental full-stage source audit and the QH limitation

These are **previous agent outputs at `0da1e13`, freshly inspected, not new full-stage runs**:
`/tmp/ld2/qh/events/{194626,195896}_sources.parquet` and
`/tmp/ld2/198658/events/198658_sources.parquet`. Current executable source equals `0da1e13`.
For the requested sources, the three-shot aggregate is:

| Source | Diagnostic | ok | skipped | error |
|---|---|---:|---:|---:|
| `features` | empty | 0 | 1 | 0 |
| `features` | `ip` | 2 | 0 | 0 |
| `features` | `qmin` | 2 | 0 | 0 |
| `qmin_rule` | `qmin` | 2 | 1 | 0 |
| `actuator` (`nbi_counter`) | `tinj_total` | 1 | 2 | 0 |
| `qh_proxy` | empty | 1 | 2 | 0 |
| `qh_flattop` | empty | 0 | 1 | 0 |

- **195896:** `actuator/tinj_total` is `ok`, coverage **0.000000–13.099900 s**, and
  **16 `nbi_counter` events** were verified in the event Parquet. This establishes that
  counter-injection is no longer invariably skipped in the full stage. `qh_proxy` is `ok` over
  **1.342599–2.679599 s**, with **zero events**; its successful row has `pass_name="zoom"`.
- **194626:** the counter-injection row is skipped because both torque and NBI power are absent;
  QH is skipped because `pinj_total` was not measured. Both have NaN coverage and zero events.
  Feature `ip` and `qmin` rows are `ok`; the q-min rule is `ok` with zero events.
- **198658:** its missing feature file produces skipped `features`, `qmin_rule`, `qh_flattop`,
  `qh_proxy`, and `actuator/tinj_total` rows. Their coverage is NaN and event counts are zero;
  the exact example reasons appear in the defect-2 section above.

**All 26 skipped rows across these three complete source files** (including other sources)
were checked for NaN coverage, nonempty reasons without absolute data/output paths, and the
absence of an additional `ok` row with the same shot/source/diagnostic. No violation was found.

The fresh helper audit also calls `rules_block` for real missing-feature shot 198658 through
the scratch symlink root: its `ran` dictionary is **empty**, and `features` and `qmin_rule`
are skipped. Seeding `ran[("actuator", "tinj_total", -1, "")]` and calling
`_counter_coverage` without current removes that key and records the skip. `_qh_coverage`
without a flat-top returns NaN coverage and `there is no Ip flat-top`; `finish_shot` only
inserts its QH `ran` row on the success branch. The full-suite defect-2 regression tests passed
(the schema-vocabulary test is the only labelmaker failure).

The other two saved QH attempts, 194549 and 194614, have error results rather than successful
source rows: `t1_s=6.143663945399567` exceeded `t_cov1_s=6.143147945404053`. The complete saved
attempts 194626 and 195896 produced zero QH events. Thus the original brief's **at least one
real-shot QH claim remains unverified**, and the earlier report's contrary claim was removed.
These pre-existing full-stage failures were not adjusted or rerun in this finish task.

Audit command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src" \
  TMPDIR=/tmp/ld2/finish/tmp XDG_CACHE_HOME=/tmp/ld2/finish/cache \
  MPLCONFIGDIR=/tmp/ld2/finish/matplotlib \
  /scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/python \
  /tmp/ld2/finish/audit_smoke.py > /tmp/ld2/finish/smoke-audit.txt 2>&1
```

The audit passed. Its aggregate data are in `/tmp/ld2/finish/smoke-audit.json`, the complete
source sample rows in `smoke-audit.txt`, and the fresh run JSON is
`/tmp/ld2/rules500/runs/events/ld2-finish-rules500.json`.

## Test counts

| suite | before | after |
|---|---|---|
| `tests/labelmaker` | 1400 passed, 2 skipped | **1442 passed, 2 skipped, 1 failed** (200.20 s pytest; 203.88 s wall) |
| `tests/ideate` | 929 passed (+ the `.mcp.json` cwd worktree artifact) | **929 passed, 1 failed** (285.21 s pytest; 289.80 s wall) |

Both full suites ran with `-W error`, with no exclusions and no code changes or reruns after
their failures. Both returned exit status 1. The labelmaker run's only failing test is
`tests/labelmaker/test_events_schema.py::test_vocabularies_are_the_documented_ones` (line 46):
its expected `KNOWN_SOURCES` tuple omits `qmin_rule`, which `events/schema.py` includes.
The ideate failure is
`tests/ideate/test_mcp.py::test_the_project_mcp_config_points_at_this_server` (line 891):
`.mcp.json` records `/scratch/gpfs/nc1514/FusionAIHub`, while the test expects this worktree.
Neither was changed, per the finish brief's stop rule. Labelmaker also printed an XRootD
finalizer `FutureWarning` about `torch.distributed.reduce_op` after pytest's summary; it is
preserved in the log rather than suppressed.

`ruff check src/labelmaker tests/labelmaker scripts/labelmaker` returned exit status 1 with
exactly the two known findings in `tests/labelmaker/data/make_tearing_dsm_golden.py`:
`SIM115` at line 25 and `UP031` at line 37. That file has no diff from base `131f092`.
There were no other Ruff findings.

### Identity checks and execution environment

After applying the existing labelmaker environment activation from `pyproject.toml:107`,
**4 passed in 26.17 s**, with `-W error`:

- `test_the_driver_writes_exactly_what_process_shot_writes`: all three worker/prefetch
  combinations `(0,1)`, `(1,4)`, `(2,2)`; this compares summary rows (including
  `by_phenomenon`), masks, events and source rows across different roots.
- `test_a_block_whose_prep_fails_in_a_worker_is_a_skip_and_not_a_hang`: one case.

The initial direct-interpreter identity attempt omitted Pixi's configured `LD_LIBRARY_PATH`
activation and yielded **1 passed, 3 failed in 12.83 s**. Spawned SciPy imports selected the
system `libstdc++` and failed on missing `GLIBCXX_3.4.26`. The successful rerun used the
repository's existing environment setting, with no source or test modification. Both attempts
are retained in `/tmp/ld2/finish/identity.log` and `identity-activated.log`.

To avoid Pixi installing or modifying files in the main checkout, these runs used the existing
interpreters named in the brief directly. `PYTHONPATH` selected this worktree's source; the
labelmaker `lib` path reproduces its declared activation. Test output, caches and temporary
files were directed to `/tmp/ld2/finish`. MiniLM's existing cached model was copied into a
scratch HF cache so the offline test could read the weights without writing the original cache.
The shared Git SHA remained `167a1b1` throughout both suites.

Exact full-suite and Ruff commands, from this worktree:

```bash
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/src"
export TMPDIR=/tmp/ld2/finish/tmp
export XDG_CACHE_HOME=/tmp/ld2/finish/cache
export MPLCONFIGDIR=/tmp/ld2/finish/matplotlib

# Labelmaker, in its own shell:
export LD_LIBRARY_PATH="/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin:$PATH"
unset LABELMAKER_FDP
/usr/bin/time -f 'wall_seconds=%e\nexit_status=%x' -o /tmp/ld2/finish/labelmaker.time \
  /scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/python -m pytest \
  tests/labelmaker -q -W error --basetemp=/tmp/ld2/finish/labelmaker-tmp \
  -o cache_dir=/tmp/ld2/finish/labelmaker-cache > /tmp/ld2/finish/labelmaker.log 2>&1

# Ideate, in a separate shell with the common exports above:
mkdir -p /tmp/ld2/finish/huggingface/hub
if test -d "$HOME/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2"; then
  cp -aL "$HOME/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2" \
    /tmp/ld2/finish/huggingface/hub/
fi
export HF_HOME=/tmp/ld2/finish/huggingface
export HF_HUB_OFFLINE=1
export IDEATE_DATA_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/ideate
export LABELMAKER_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker
export IDEATE_CORPUS=/scratch/gpfs/EKOLEMEN/foundation_model
export PATH="/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/ideate-cpu/bin:$PATH"
/usr/bin/time -f 'wall_seconds=%e\nexit_status=%x' -o /tmp/ld2/finish/ideate.time \
  /scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/ideate-cpu/bin/python -m pytest \
  tests/ideate -q -W error --basetemp=/tmp/ld2/finish/ideate-tmp \
  -o cache_dir=/tmp/ld2/finish/ideate-cache > /tmp/ld2/finish/ideate.log 2>&1

RUFF_CACHE_DIR=/tmp/ld2/finish/ruff-cache \
  /scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/ruff \
  check src/labelmaker tests/labelmaker scripts/labelmaker > /tmp/ld2/finish/ruff.log 2>&1
```

## Commits

On base `131f092`, in order:

1. `afc84a2` — `labelmaker: L-D2 - the Ip flat-top and the q-min regime rule`
2. `0da1e13` — `labelmaker: L-D2 - the features store in the events stage, and a missing prerequisite that is no longer an evaluated zero`
3. `167a1b1` — `labelmaker: L-D2 - document the q-min regime rule, the Ip flat-top and the features store in the events stage`
4. This report commit — `labelmaker: L-D2 - report with the 500-shot rules-only smoke and final suite counts`

All four use `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
The report commit identifies itself by subject to avoid embedding its own changing hash.

## Deviations from the brief

1. **Deliverable 4 (the `d3d_qmin_regime_rule` folder) is skipped**, under the brief's own escape
   clause, with the measured reason above (293/71/54 and 45 shots' disagreement through the runner's
   25 ms grid; plus the `verify_artifacts` carve-out it would need).
2. **The bare `qmin` alias is left out** of all three lexicon entries, where the brief listed it as a
   candidate. Reason above; a test pins the omission so it is a decision and not an oversight.
   "reversed shear" is likewise out, which the brief anticipated.
3. **`qmin_regimes` computes its own coverage** rather than taking a `t_cov` argument, so the
   signature is exactly the brief's `(qmin_t, qmin_y, flattop, *, shot)`. `qmin_rule_coverage` is
   exported so the pipeline declares the same span the rows carry.
4. **The 500-shot smoke's `--root` needs its `features/` pointed at the real store.** `--root` moves
   both ends of the stage, so a scratch root has no features to read; the run used
   `ln -s $LABELMAKER_ROOT/features /tmp/ld2/rules500/features` and wrote everything else under
   /tmp. Nothing in `rules_shot` writes to `features/`, so the store stayed read-only. Documented in
   `docs/LABELMAKER.md` beside the invocation.
5. **`ShotResult` gained `by_phenomenon`** (and `as_row()` a key), because `--rules-only`'s summary
   has to count shots per BAND and `by_source` cannot: one source writes three phenomena. Both
   schedules fill it identically, so `test_tokeye_masks`' driver/sequential row equality still holds.
6. **The events-stage source table gained two keys** beside the q-min ones: `features` (the store
   read) and the relocated `nbi_counter` row. The second is a behaviour change to an existing row,
   forced by the coordinator's addition; the existing tests that named the old key were updated
   rather than deleted.

7. **Both full suites failed; no fixes were made after those results.** Labelmaker's expected
   vocabulary lacks `qmin_rule`; ideate retains the known `.mcp.json` worktree-cwd failure.
   Ruff has exactly its two documented pre-existing findings. This finish is a measured report,
   not a claim that the branch passes its acceptance gate.
8. **The rules-only smoke cannot certify `nbi_counter` or QH execution.** Their source rows are
   absent on all 500 shots. A separately audited prior full-stage shot (195896) has 16
   counter-injection events, but the real-QH claim criterion is not demonstrated. Two saved
   full-stage QH attempts also failed the event-coverage bounds check, as detailed above.
9. **Execution setup deviations are retained, not hidden.** A first smoke launch accidentally
   set `LABELMAKER_LABEL_TABLES` to this worktree's nonexistent `src/data/labels`. It was
   interrupted before completion (exit 130), and its partial outputs/log/timing were preserved
   under `/tmp/ld2/finish/rules500-interrupted-table-path*`. The completed 86.33 s smoke used a
   fresh root and the correct repository default `data/labels`. No thresholds changed.
   The first identity attempt lacked the existing Pixi library-path activation; after reproducing
   that configured environment, all four cases passed. Direct interpreter invocation and scratch
   caches avoided running Pixi against the main checkout. No dependencies were installed.
10. **Documentation corrections:** the store opens separately per quantity; 500 ms spans 25
    sample intervals; the ungated census checks `qmin > 0.95` without requiring one exclusive
    band; zero events does not imply q-min stayed below 0.95; changing equilibrium requires
    updating provenance as well as the resolver; source rows and the rules-only scope are now
    explicit. The fresh run wrote 500 event files, not the earlier report's claimed 337.
