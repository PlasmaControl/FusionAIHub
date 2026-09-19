# Group A labels assessment — 2026-09-13

This assesses the four CSV rows as one labels project, against their **Proposed Method and Notes**, using the workstream's `done` criteria. All four are **in_progress**. Existing models and this branch's observed-ELM implementation are useful pieces; none establishes a completed, validated, indexed observed family on all 500 shots today. Forecasts are not observations. No production data, labels, events, or index was changed by this assessment.

Scope sources: [CSV](../../../data/labels/Recommender%20System%20-%20Discrete%20Labels.csv), [workstream](2026-09-13-labels-workstream.md), [labelmaker](../../LABELMAKER.md), and the local model cards linked below. The implementation started from `b265f40` on `recommender-LA`. The final implementation commits and verification results are recorded in [task report](../../../.superpowers/sdd/task-LA-report.md).

## Read-only census

The population is the 500 distinct shots in `/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/recommender_v1.txt`, from 185786 through 204925. Its SHA-256 is `a58c9c8983bab1c71d5ab74eb4949f86b0a7d06df7b0f1ff5a39132aee6ac2e2`. The census reads the existing ideate parquet tables, opens every pool shot's label and feature HDF5 file with mode `r`, and inventories every production event file. It does not resolve features or run a model.

```bash
cd /scratch/gpfs/nc1514/FusionAIHub-build
CUDA_VISIBLE_DEVICES='' PYTHONPATH=$PWD/src /scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/ideate-cpu/bin/python scripts/labelmaker/assess_labels_a.py census --out /tmp/task-LA/census
```

The complete machine-readable output is `/tmp/task-LA/census/census.json`. Inputs are `/scratch/gpfs/EKOLEMEN/nc1514/ideate/db/{labels_wide,events,event_sources,corpus_coverage}.parquet` and `/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/{labels,features,events}`. These results are the **2026-09-13 snapshot, measured at commit `b265f40`**, not a claim about future production joins.

| CSV row | `labels_wide` rows / shots | Event rows: detector / heuristic / forecast / other | Shots with observed events | Forecast-only event shots | Status and decisive gap |
|---|---:|---:|---:|---:|---|
| Tearing Mode | 13,800 / 500 | 0 / 0 / 1,016 / 0 | 0 | 69 | **in_progress** — forecasting exists; observed onset and n=1/2 confirmation do not |
| AE Mode | 1,000 / 500 | 0 / 0 / 0 / 0 | 0 | 0 | **in_progress** — activity model exists; weak human agreement and no published observed AE family |
| ELM | 2,000 / 500 | 0 / 0 / 21 / 0 | 0 | 7 | **in_progress** — new D-alpha family tested in `/tmp`; production has only forecasts and no manual validation |
| Sawtooth | 0 / 0 | 0 / 0 / 0 / 0 | 0 | 0 | **in_progress** — ECE rule exercised on ten shots; false candidates and missing radius/ground truth prevent acceptance |
| Total | 16,800 / 500 unique | 0 / 0 / 1,037 / 0 | 0 | 76 unique | No observed-source coverage rows on the 500 |

`labels_wide` has **no `evidence_kind` column**. Its rows summarize `(shot, model, output label)` arrays, not individual observations or valid timestamps. All 1,037 indexed event rows have `source=label_forecast`, `evidence_kind=forecast`. There are no detector, heuristic, human, text, database, or model event rows in this population. `event_sources.parquet` is empty. A zero observed count therefore means unprocessed/unknown, not a negative label.

The raw label files contain AE, ELM DSM, tearing CNN, and original tearing DSM groups on 500/500 shots; the continued tearing DSM group occurs on 140/500. The model row counts are respectively 1,000, 2,000, 1,000, 10,000, and 2,800. Much of this is invalid padding or auxiliary outputs:

| Representative output | Summary rows | Shots with ≥1 valid sample | Valid time samples | Shots whose maximum valid value exceeds 0.5 |
|---|---:|---:|---:|---:|
| Tearing CNN `tm_prob` | 500 | 163 | 15,045 | 67 |
| Tearing DSM `tm_risk_1s` | 500 | 151 | 17,179 | 0 |
| Continued tearing DSM `tm_risk_1s` | 140 | 139 | 15,518 | 0 |
| AE `ae_active` | 500 | 203 | 48,720 | 170 |
| ELM DSM `elm_risk_20ms` | 500 | 10 | 1,691 | 0 |

The four ELM forecast horizons each have 10 valid shots. The 0.5 column is descriptive, **not** the forecast join threshold: the DSM operating threshold is 0.2, which explains the seven ELM forecast-event shots. Auxiliary beta, frequency, mixture, and survival parameters must not be counted as mode detections.

At that snapshot there were **exactly three production labelmaker event files**: `185946_events.parquet`, `185953_events.parquet`, and `198658_events.parquet`. **None is a recommender_v1 shot.** They contain older TokEye tracks/transients, ECE sawtooth candidates and rules, including old `tokeye_transient/elm` labels and magnetics-derived `elm_clock/elm_free`. There were zero production `*_sources.parquet` files. These three files cannot support a claim of observed coverage on the 500. L-A keeps them read-only; query compatibility and rerun migration are covered by regression tests.

**Current inventory as of the read-only review (2026-09-14):** production `events/` holds **18 event files and 15 source files**, including **15 `recommender_v1` shots** from L12's pilot products. The joined `events.parquet` is unchanged (SHA-256 `7a46f795e2e05ba6617a92e4b5c5c7767edd3a4fec6d919086904616fe46195d`) with **zero observed rows**. Per-shot products are distinct from the joined database; the newer files do not change the dated joined-table census above. See [review check 2](../../../.superpowers/sdd/task-LA-review.md).

Input table SHA-256 values, for snapshot identification:

| Table | SHA-256 |
|---|---|
| `labels_wide.parquet` | `15609115e0873c25307d69859dc90d4046611b7c8bcde1ecf61155b5547c77bd` |
| `events.parquet` | `7a46f795e2e05ba6617a92e4b5c5c7767edd3a4fec6d919086904616fe46195d` |
| `event_sources.parquet` | `32d4604024290790dc87470fbffbb81c3722b6ddb199d7c0e507e549e1a1a540` |
| `corpus_coverage.parquet` | `7979710d0e8673a9507686d8049d872abf6ccc7e9fbb01eb63fa18e6ec82067d` |

## Available diagnostic inputs

| Corpus group | Present on the 500 | Nominal channel count | Candidate channels, zero based |
|---|---:|---:|---|
| `mhr` | 500 | 8 | 0, 4; existing magnetic TokEye reference |
| `mirnov` | 500 | 29 | 0, 8 for existing fallback; geometry audit needed for mode-number fitting |
| `ece` | 500 | 48 | 8, 20, 40 for TokEye; all 48 for inversion |
| `filterscopes` | 500 | 104 | 0–7 D-alpha; select a usable finite channel |
| `co2` | 204 | 4 | 0, 2 for corroborating mode tracks; AE model uses all four |
| `bes` | 256 | 64 | 26, 28 where present; seven pool shots have no BES metadata row |

Presence is cached group availability, not proof that a particular channel contains finite samples or has a calibrated spatial mapping. In particular, a present filterscopes group can have an all-NaN channel 0. The pipeline falls back through 0–7, requiring at least two adjacent finite samples.

The brief's global Mirnov 98.6% and MHR 86% are historical corpus-wide availability figures, not the 500-shot denominators above. The current cached table has Mirnov 16,444/16,604 entries present and MHR 13,744/16,607; differing incomplete metadata denominators preclude treating these as a new exhaustive whole-corpus census. Both groups are present on all 500 selected shots.

## Tearing Mode — in_progress

The CSV proposes **“Survival Model, detection model”** and has no Notes. The [CNN card](../../../src/labelmaker/models/d3d_tearing_onset_cnn1d/README.md), [DSM card](../../../src/labelmaker/models/d3d_tearing_time_to_event_dsm/README.md), and [continued DSM card](../../../src/labelmaker/models/d3d_tearing_time_to_event_dsm_continued/README.md) describe available model implementations. Despite the detection-model wording, the CNN's `tm_prob` targets tearing presence **25 ms ahead**; DSM risks target future onset over 250/500/1,000 ms. They are forecasts, including the CNN, and cannot timestamp a retrospective observed onset. The registry still exposes `tm_prob` as a label association; that needs an explicit forecast-semantics follow-up.

The proposed forecasting pieces exist but do not satisfy the workstream's observed/source-coverage/search acceptance. An **observed tearing-onset event is in scope**: the row describes the physical phenomenon, and a forecast alone cannot answer when it happened. Existing TokEye coherent-mode intervals are detector evidence for a mode, not proof of tearing or of toroidal mode number.

The concrete follow-up is a TokEye track in a 0.5–30 kHz tearing band, with stored onset, duration, occupancy/duty, mean/peak mask probability, centroid/range/slope/bandwidth, pickup flag, and corroborating magnetic phase descriptors. Initial development cuts are duration ≥50 ms, duty ≥0.5, confidence ≥0.6, bandwidth ≤30 kHz, and no pickup flag. These are **proposed starting thresholds, not validated operating points**.

Confirm |n|=1 or 2 using calibrated, toroidally separated magnetic probes: at least four valid probes, pairwise coherence ≥0.7, a wrapped phase-versus-toroidal-angle fit with circular RMS ≤π/6, and a consistent winning n over ≥20 ms. Store n, fit residual, coherence, probe names, polarity/delay corrections and geometry version. Missing geometry means n unknown and no confirmed tearing claim. Do not infer n from a frequency band or a poloidal array.

The corpus's 29-channel `mirnov` and eight-channel `mhr` groups supply signals, but their positional and phase conventions must be audited. The read-only peer source `/scratch/gpfs/nc1514/fdp/scripts/omnimode.py` lists a 31-probe MPI **poloidal** ring at approximately one toroidal angle (322°), with geometry in the omnimode metadata CSV. That ring alone cannot determine toroidal n; neither its channel count nor ordering can simply be substituted for the 29 corpus channels. Resolve the actual toroidally separated probe identities before the proposed fit. Validation must compare onset and n against independent expert labels, with shot-separated development and held-out positives, non-tearing low-frequency modes, receiver pickup, and negative shots. Published confirmed onset points should use `evidence_kind=detector`, with the heuristic cuts and limitations documented rather than represented as calibrated probabilities.

## AE Mode — in_progress

The CSV proposes **“Tokeye + AzaLenny”**, with no Notes, and explicitly asks to improve the existing method with TokEye. The [AE model card](../../../src/labelmaker/models/d3d_ae_activity_seldnet/README.md) reports AUROC **0.9908** on teacher/human-agreement frames, **0.9857** against TokEye mask activity, but only **0.6209 against human annotation**. Human precision is 0.2584 and recall 0.8804; the headline 0.99 is not human-level classification quality. The frequency head does not beat a constant baseline (17.63 versus 16.79 kHz MAE). `ae_active` is a 25 ms mean activity probability, not a validated observed AE interval; its CO2 input observes only the instrument band up to 250 kHz, with the trained band starting near 80 kHz.

The existing AE producer is a neural model: its stored label arrays have no event `evidence_kind`, and there is no AE event adapter to assign one. They must not be retroactively counted as observed detector events. There are valid activity samples on 203 pool shots and no observed AE event rows. Neither a high teacher score nor the 170 shots with a sample above 0.5 completes the row.

Define the observed AE candidate from TokEye coherent tracks using a proposed 40–250 kHz band, duration ≥20 ms, duty ≥0.5, confidence ≥0.5, bandwidth ≤100 kHz, and pickup exclusion. Store all track descriptors and harmonic/coincident-track associations. Harmonics should be reported, not required: an isolated fundamental can be an AE, and a low-frequency EHO harmonic comb must not be relabeled as AE. Require corroborating magnetic (`mhr` 0/4 or verified `mirnov` channels) and ECE 8/20/40 or CO2 0/2 tracks where usable; proposed agreement is ≥20 ms overlap and frequency within max(2 kHz, 5%). Missing corroboration is an explicitly weaker candidate, not a confirmed negative. Use `evidence_kind=detector`; tune these cuts on development data and freeze them before human evaluation.

The aemodes human annotations are the required validation reference, not the TokEye teacher. The card's 180 annotated shots (170659–178879, 120 training / 60 held out) have **zero FAITH corpus files**. Their original signal cache must be used read-only to reproduce that held-out evaluation; one cannot join those annotations to a recommender shot and call it validation. This is a domain-transfer test, not in-corpus ground truth. Add a blinded human-labelled in-corpus subset, and retain the existing held-out split. AzaLenny provenance/interface and any different intended model need recording before claiming the CSV's named method is fully satisfied.

## ELM — in_progress; observed point family implemented

The CSV proposes **“Semin and Jalal?”**; its classification field mentions a BES-only detector and **David Smith's manual label database**. Notes are empty. Those external implementations and the manual database have not been supplied or validated here. The [existing DSM](../../../src/labelmaker/models/d3d_elm_time_to_event_dsm/README.md) supplies 5/10/20/50 ms forecasts, not observed ELM times.

This branch adds an independent D-alpha clock over filterscopes channels 0–7. The first channel with a contiguous finite run is normalized over its finite range to [0,1], smoothed over 0.64 ms, and passed to the existing peak picker with prominence 0.03 and minimum distance 3 ms. Min-max scaling is essential: real corpus D-alpha amplitudes are around 10^15, whereas the former mask trace was bounded by one. These starting rules and the first-usable-channel policy still need physical validation, saturation checks and a plasma gate.

Every peak produces a POINT event (`t0_s == t1_s`) with `source=elm_clock`, `phenomenon=elm`, `evidence_kind=heuristic`, `confidence=NaN`, `diag=filterscopes`, and exactly `{prominence, width_ms, channel, rate_hz_local}` in attrs. Prominence is on the normalized trace, width is the smoothed peak's half-prominence width, and local rate uses the existing centred 100 ms clock. `elm_free` intervals derive from **those same peaks**. Both use the chosen filterscope's first/last finite sample as source coverage. NaN gaps are not smoothed across and do not receive quiet intervals; the existing single-span source contract still stores an outer hull, so it does not encode every interior hole.

TokEye retains `source=tokeye_transient`, detector evidence, checkpoint provenance and burst attributes, but its phenomenon is now **transient**, registered in both lexicon and ideate. It never supplies an ELM count or ELM coverage. Legacy stored `tokeye_transient/elm` rows are normalized to transient in the read-only MCP response; registry ELM matching excludes that source. Legacy magnetics `elm_clock` rows cannot prove D-alpha coverage. A rerun replaces obsolete shot-level index families while preserving other shots. A rejected track conversion is isolated to its diagnostic block and recorded as skipped with unknown coverage; it cannot prevent the independent D-alpha clock or valid blocks from publishing.

Both windows and ideate now select `elm_clock/elm`. Window reduction formulas are unchanged and tests keep the same expected numbers for the same point train. **The brief's parenthetical premise was not true at `b265f40`: windows previously selected TokEye transients, and the clock was mask-derived.** L-A implements the required D-alpha behavior; actual window values can consequently change when its peak train differs.

Synthetic tests establish N known pulses at peak times within one sample, physical-amplitude invariance, finite source coverage, NaN-gap behavior, zero ELM contribution from transients, quiet-source recording, independent operation without mask coverage, fallback channels, legacy migration, and observed `ideate phenomenon ELM` / `get_events` results. Real CPU outputs on three pool shots are reported below. This establishes the software path, not precision or recall against manual ELMs. The whole row remains in_progress because production has no observed rows/source coverage and manual validation is outstanding; the named manual-database dependency is blocked_external within its follow-up.

## Sawtooth — in_progress; ten-shot validation check

The CSV proposes **“Tokeye”**, notes existing OMFIT/Hiro ECE approaches, and says **“Typical sawtooth 0.3-0.6 inversion layer. If sawtooth trigger ELM, inversion not visible.”** The available implementation is the transparent `ece_sawtooth` heuristic in `events/heuristics.py`, not a TokEye sawtooth classifier. It forms 1 ms ECE envelopes, proposes coherent fractional drops (≥2% over ≥2 channels, ≥10 ms separation), and checks a contiguous dropping region with an adjacent rising region using before/after crash profiles. It emits heuristic candidates with the dropping block's channel bounds. Testing that its own sign rule fired is not independent proof of a sawtooth.

```bash
cd /scratch/gpfs/nc1514/FusionAIHub-build
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=$PWD/src /scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/python scripts/labelmaker/assess_labels_a.py sawtooth --out /tmp/task-LA/sawtooth-labelmaker
```

The ten shots are the evenly spaced indices of the sorted 500-shot pool (`linspace(0,499,10,dtype=int)`). All computation is on CPU. Artifacts are one event parquet and one signal/crash-profile PNG per shot, plus `sawtooth.json`, under `/tmp/task-LA/sawtooth-labelmaker`. The run uses Python 3.11.16, NumPy 1.26.4, SciPy 1.17.1. An exploratory NumPy 2.4.6 run under `/tmp/task-LA/sawtooth` yielded 670 instead of 672 candidates; threshold-sensitive numeric differences are another reason to record the runtime, not evidence of physical disagreement.

| Shot | Crash candidates | Median period (ms) | Modal channel boundary (events) | Below 100 kA | Ip unknown | Radius |
|---|---:|---:|---|---:|---:|---|
| 185786 | 59 | 16.0 | 46/47 (11) | 8 | 3 | unknown |
| 187168 | 55 | 41.5 | 39/40 (15) | 0 | 1 | unknown |
| 189890 | 64 | 77.0 | 31/32 (9) | 8 | 0 | unknown |
| 191213 | 163 | 21.0 | 39/40 (35) | 43 | 0 | unknown |
| 195037 | 57 | 67.0 | 31/32 (12) | 0 | 0 | unknown |
| 196488 | 61 | 77.0 | 39/40 (10) | 0 | 0 | unknown |
| 200133 | 35 | 84.5 | 39/40 (6) | 4 | 0 | unknown |
| 201718 | 65 | 45.5 | 13/14 (24) | 0 | 0 | unknown |
| 203660 | 42 | 55.0 | 31/32 (4) | 3 | 0 | unknown |
| 204925 | 71 | 78.0 | 46/47 (20) | 1 | 0 | unknown |

Total: **672 candidates**, including **67 below 100 kA** and **4 with unknown Ip**.


Each ECE record's finite coverage is approximately [-0.050000001, 6.143147945] s. The period is the median spacing of **all candidates**, not an independently validated physical sawtooth period. The channel column reports the most frequent high-index dropping-block boundary, zero based, as `(stop-1)/stop`; it is only one edge of a half-open dropping block. The full distribution and each event's low and high bounds are in the artifacts. It is not a unique inversion layer and is not normalized radius.

A signal-level visual check revealed candidates before the pulse and in the low-signal tail, particularly on 191213. The table's independent sanity check interpolates existing features-store Ip (units A) at each candidate and counts |Ip| <100 kA; unknown Ip is reported separately. This is a diagnostic check, **not a tuned production gate or a new ground-truth label**. Candidates in that range cannot justify confident claims of plasma sawteeth. The existing relative-drop and sign-reversal tests can accept noise when absolute signal is small, and currently have no plasma-current gate. This validation therefore identifies a failure requiring follow-up, rather than approving all 672 candidates.

### Radius mapping and the ELM-triggered limitation

All 500 features-store ECE groups contain only `xdata` and `ydata`; no ECE frequency, geometry, radius or rho mapping is stored. The 33-point `te_zipfit` rho grid does not map the 48 ECE channels. **No radius is derivable from those features alone**, and every radius in this check is recorded as unknown; channel/48 is not a valid substitute for radius.

There is a reusable fdp-backed route in peer source, inspected read-only without running its writers:

- `/scratch/gpfs/nc1514/fdp/scripts/elmcycle.py` records the ECEVS01…48 signal naming and TECEF amplitude calibration; amplitude calibration is not radial mapping.
- `/scratch/gpfs/nc1514/fdp/scripts/omnimode.py` defines an ECEGEOM request for `\ECE::TOP.SETUP.{FREQ,ECEZH,ECEPHI,ECETHETA,FLTRWID}` in the electrons tree, plus EFIT equilibrium inputs.
- `/scratch/gpfs/nc1514/omnimode/src/omnimode/modefit/ece_fwd.py` loads raw ECEGEOM or a per-shot geometry file and maps channel frequency to second-harmonic resonance position and equilibrium coordinates. Its approximate relation is f = 2 × 27.99 GHz/T × |B_T(R,Z)|, with B_T from the equilibrium; finite optical depth and relativistic shifts require care.

This is **available mapping code**, not a validated per-shot mapping in the current feature store. Reusing it requires checking ECEVS channel ordering, GHz/Hz conversion, equilibrium time alignment, ray geometry, which side of the magnetic axis each boundary lies on, and the definition of normalized radius. The CSV's 0.3–0.6 must be evaluated only after that coordinate definition is explicit. Neither this check nor an assumed monotonic channel index establishes that interval.

If a sawtooth triggers an ELM, the adjacent positive ECE heat pulse can be masked by the coincident edge transient. The present inversion rule can miss that crash; absence of a visible inversion is not evidence of absence of a sawtooth. Validation must include manually adjudicated coupled events using independent core crash timing plus ELM observations, report this subset's recall separately, and allow an explicit unknown inversion radius. The new D-alpha clock alone is not independent sawtooth ground truth.

## Real CPU events-stage check

```bash
cd /scratch/gpfs/nc1514/FusionAIHub-build
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=$PWD/src /scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/python -u -m labelmaker.run events --shots 185786 191213 204925 --root /tmp/task-LA/real-events --unet /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/models/tokeye/big_tf_unet_251210.pt --device cpu --passes wide --tile-batch 4 --timeout 3600 --run-id task-LA-cpu-final
```

This runs the actual pinned TokEye network, not a test double, and reads the actual corpus. The first command succeeded on 185786 and 191213 but failed on 204925 with an existing track-coverage violation (A9 below); that failed shot is not counted as a successful ELM demonstration. A supplementary identical command with `--shots 187168 --run-id task-LA-cpu-supplement` reproduced the same failure. After the test-first block-isolation fix, a rerun with `--shots 187168 --run-id task-LA-cpu-isolated` supplies the third successful pool shot, with log `/tmp/task-LA/real-events-isolated.log`. Failed track blocks remain skipped, not quiet; their masks are retained for diagnosis. The manifest, masks, event/source parquets, run log and index are under `/tmp/task-LA/real-events` and `/tmp/task-LA/real-events-final.log`. Production remains unchanged. Expected unrelated unavailable diagnostics are recorded as skips rather than silently interpreted as negatives.

| Shot | ELM points (`elm_clock`, heuristic) | ELM-free intervals | Transient points (`tokeye_transient`, detector) | Filterscope channel | Clock coverage (s) | Recorded skips |
|---|---:|---:|---:|---:|---|---:|
| 185786 | 1388 | 0 | 2 | 1 | [-0.049880, 7.949920] | 9 |
| 191213 | 1057 | 0 | 12 | 0 | [-0.049880, 6.949920] | 11 |
| 187168 | 488 | 3 | 548 | 1 | [-0.049880, 7.949920] | 11 |

Total: **2,933 ELM points**, **562 generic transient points**, and **3 ELM-free intervals** across three successful shots. All have five actual CPU mask blocks. On 187168, ECE track blocks 8 and 20 are explicitly skipped with unknown coverage after conversion errors; its D-alpha clock and remaining blocks complete.

A fresh CPU recomputation of each saved filterscope trace exactly matches the stored ELM times and verifies point semantics, attrs, evidence kind and coverage. Evidence: `/tmp/task-LA/verify_real.py`, `/tmp/task-LA/real-events-verification.json` and `/tmp/task-LA/real-events-verification.log`. This is a serialization/current-code check, not independent manual classification.


The counts are software-validation outputs of the initial peak rule, not manually verified ELM truth. Peak thresholds, plasma gating and channel selection still require the acceptance study below.

## Follow-up tasks and exit criteria

Each task is sized for one agent and has a bounded output. Thresholds below are proposed acceptance targets to freeze before opening the held-out results. Do not tune on the acceptance set. Curated label data belongs under `data/labels/`; production artifacts remain outside git. No colleague messages were sent.

**A1 — Correct tearing forecast semantics.** Remove present-observation implications from the CNN `tm_prob` registry path, preserve its 25 ms horizon, and retain DSM horizons. Exit: synthetic tests prove that each CNN/DSM label can appear only in the forecast field, including CLI/MCP responses and mixed observed/forecast shots; source coverage and horizon semantics are documented; both warning-strict suites pass. Record revised per-model valid-shot and forecast counts on the 500 without counting them as observed. This is a prerequisite to a truthful completed tearing row, not the observed-onset task itself.

**A2 — Resolve magnetic geometry for n confirmation.** Audit corpus `mhr`/`mirnov` channel names, toroidal positions, polarity, timing and transfer phase against the fdp inputs on 20 pool shots; identify at least four usable toroidally separated probes. Exit: a versioned mapping with provenance and units, synthetic n=1/2 and wrong-n/noise phase-fit tests, measured finite-input availability on all 500, and a report that explicitly marks any unavailable geometry/probes. If the available array is only poloidal, return a named missing diagnostic dependency and `blocked_external` for n confirmation; do not invent n. This task provides the mapping needed by A3, not a completed tearing family.

**A3 — Publish and validate observed tearing onset.** Implement the track and n rule specified above using A2's mapping; emit `phenomenon=tearing` POINT onsets under a distinct documented detector source, retaining full track and phase provenance. Validate on at least 30 shot-separated development and 30 held-out shots including ≥100 expert onsets plus negative, pickup and non-tearing examples; target held-out precision ≥0.90, recall ≥0.80, and median onset error ≤10 ms with matched-event tolerance ±20 ms. Exit: report confusion counts, timing errors and bootstrap uncertainty by shot, failures and n-unknown cases; satisfy the full integration checklist below and record the all-500 source/observed/forecast-only counts. A failed quality target leaves the row in_progress, regardless of event volume.

**A4 — Publish and validate observed AE intervals.** Implement the specified track, corroboration and harmonic descriptors, with `evidence_kind=detector`; document how it relates to AzaLenny. Use the original aemodes held-out 60-shot human set (never teacher labels as truth) and a new blinded ≥30-shot corpus set, each containing negatives. Exit: report interval precision/recall at IoU ≥0.3 and frame-level AUROC against humans, instrument-band and per-diagnostic results; exceed the current 0.6209 human AUROC baseline on the original split and target precision ≥0.75 / recall ≥0.80 on the independent corpus set, with shot-level confidence intervals. Publish matched/failed examples and track provenance. Satisfy the integration checklist and report all-500 availability/observed counts; if original signal/annotation access is missing, name that dependency rather than training/evaluating on the same frames.

**A5 — Validate ECE sawtooth timing and inversion radius.** Reuse the verified fdp/geometry route to map both dropping-block boundaries to an explicitly defined normalized radius; add current/signal-quality gating and tests for pre-pulse, post-pulse and missing-geometry cases. Keep this ten-shot set as development evidence. Assemble ≥30 separate held-out shots with ≥200 independently annotated crashes, ≥10 quiet controls, and ≥30 sawtooth/ELM coupled examples (overlap of these subsets is allowed and must be listed). Exit: precision ≥0.90, recall ≥0.80 within ±3 ms, per-shot period error reported, and separate coupled-event recall/unknown-inversion statistics. Report the fraction of independently verified isolated events in rho 0.3–0.6 and deviations; do not filter to that band to manufacture agreement. Target radius agreement within 0.05 in the declared coordinate where independently mapped references exist; otherwise radius stays unknown. Satisfy the integration checklist and all-500 ledger counts before marking the row done. A purely ECE rule also needs its relation to the CSV's proposed TokEye method explicitly documented.

**A6 — Independently validate the ELM point rule.** Compare filterscopes 0–7, saturation, finite gaps and plasma gating on ≥20 development and ≥30 held-out pool shots, with ≥200 manually timed ELMs and ≥10 quiet/noisy controls. Include BES and non-BES shots and coupled sawtooth/ELM cases. Exit: precision ≥0.90, recall ≥0.90 within ±2 ms, median time error ≤1 ms, and no false ELM-free claim across unobserved intervals; report shot-level uncertainty and failed cases. Freeze normalization/channel/threshold policy and reproduce the same peak train in events, free intervals and windows. Resolve the intended Semin/Jalal implementations and compare their availability and timing rather than claiming their method was ported. Satisfy the integration checklist and all-500 ledger counts. The present three-shot run is development evidence, not this held-out acceptance study.

**A7 — Manual ELM database intake (blocked_external).** Dependency: David Smith's manual ELM database plus permission/provenance, shot identifiers, time units and the examined intervals; Semin/Jalal detector artifacts are a separate named comparison dependency. The user/controller obtains these; this task sends no message. Exit after receipt: put source labels under `data/labels/`, validate units and duplicate handling, census corpus/pool overlap, and ingest with `evidence_kind=database` and unknown coverage unless an examined-range contract is supplied. Test that unlisted shots are never negatives, database rows never become diagnostic window counts, and ideate exposes the database field. Report the pool count, including zero. Keep manual annotations distinct from independently generated heuristic observations.

**A8 — Publish validated families and join the 500-shot index.** After the relevant family acceptance task passes, run that source over the exact frozen pool, record successful/quiet/skipped source rows with real diagnostic coverage, and join events and source tables into ideate using the existing driver. Exit: all 500 have a recorded processing outcome, every success/skip is explained, event counts agree between per-shot products and joined tables, forecast-only and observed counts are separate, and the controller's ledger records the final counts/commit/commands. Query at least three positive and three quiet/skipped examples per family through both CLI and MCP. This task must be separately authorized for production writes; L-A only writes temporary validation products.

**A9 — Recover valid rows inside rejected track blocks.** Real CPU attempts on 204925 and 187168 failed at existing ECE track endpoints beyond the 6.143147945404053 s coverage, by 0.772 and 0.516 ms respectively. The existing `clip_to_coverage` intentionally refuses wholly outside tracks; a synthetic point track reproduces that refusal (`/tmp/task-LA/reproduce_track_padding.py`). L-A now isolates conversion failures by diagnostic block, records the block as skipped with unknown coverage, and publishes the unaffected blocks and independent ELM/sawtooth sources; a regression test guards this. The remaining task is finer-grained recovery of valid tracks in a failed block. Exit: a test-first fix discards or explicitly records unobserved support without moving it into a plausible observation, preserves original track identities/co-occurrence references, and reports rejected-row counts/reasons. Reprocess the saved masks where available and rerun 204925 on CPU successfully, retain coverage invariants, pass both warning-strict suites, and record the pool's rejected-block/row counts before A8. Do not convert a rejected block into a quiet negative.

**Integration checklist for a row to be `done` (A3/A4/A5/A6 + A8):** a source actually produces the row's labels/events on recommender_v1; evidence_kind and forecast horizon are correct; source coverage is written even for quiet successful runs; missing input remains unknown; the phenomenon is in `lexicons.yaml` and `phenomena.yaml`; ideate phenomenon search and `get_events` expose the correct evidence class; a regression test guards each of these; both exact suites pass with `-W error`; lint is clean for the changed components; and the shot-level 500-shot count and quality result are recorded in the ledger. Partial outputs, teacher agreement, an available implementation, or external table presence alone do not satisfy this checklist.
