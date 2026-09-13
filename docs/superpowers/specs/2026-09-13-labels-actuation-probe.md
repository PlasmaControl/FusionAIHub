# Actuation probe — L-C-probe

Measured on the login node `stellar-vis2.princeton.edu`, 2026-09-13, on
`recommender-LCprobe` (starting revision `b265f4012798278ac28a5915f4a174ad4ea70a09`).
This is a measurement and definition proposal. **No labels, events, canonical
features, registries, or data stores were implemented or updated.**

The most consequential results are:

- **LH exists:** `\RF::LH_POWER` returned a measured kW waveform on 203505
  (2025-05-28). The default **LH = lower-hybrid current drive** is supported by
  the RF tree's `LHCD` branch and the operations literature. It must not be
  marked `no_data_in_corpus`.
- **Helicon exists:** the RF tree explicitly describes TWA as
  `Traveling Wave Antenna, aka Helicon`. Six TWA power candidates return on
  200009 and 201068. Their data-unit field is a blank space: a watts conversion
  remains an inference, not a confirmed calibration.
- **EFC inputs already exist:** `i_coil` includes the six C-coils as well as
  twelve I-coils. Three separate n=1 current-amplitude signals in the
  `operations` tree are readable on all 20 probe shots.
- **Current Saturation is not yet an operational definition.** `ecoil` is
  measured in A, and F-coil currents are also available. Neither a flat current
  trace nor a large absolute current proves that an applicable current limit
  was reached. The E-coil reading is plausible, but remains unconfirmed as the
  user's intended meaning; voltage/control-channel calibration is still needed.
- **Pellet remains unresolved as delivered material:** lithium-granule event
  nodes exist but are empty in the sample; additional PCS candidates are
  measured below. A sequence command is insufficient evidence of pellet arrival.

## Scope, sample, and access

Requirements: [task family/spec](2026-09-13-labels-workstream.md), the Actuation
rows in `data/labels/Recommender System - Discrete Labels.csv`, and the task
brief `.superpowers/sdd/task-LCprobe-brief.md`. The bounded task covers NBI,
ECH, Gas, RMP, Helicon, Pellet, EFC, Current Saturation, and LH.

The 500-shot YAML actually contains **2021–2025**, with year counts
64 / 130 / 94 / 116 / 96; it contains no 2020 shot. For the remote probe, sort
each year's selected shots numerically and take ranks
`round(i * (n_year - 1) / 3)`, `i = 0, 1, 2, 3` (Python's `round`). This is a
deterministic four-shot spread per represented year, not a random prevalence
sample. All waveform requests below address these same 20 shots, one point and
one shot per call. No SLURM, bulk-fetch API, or forked fetch pool was used.

| Year | Shot (run date) |
| --- | --- |
| 2021 | 185786 (04-12), 186257 (04-30), 186892 (06-10), 187308 (06-24) |
| 2022 | 188734 (04-15), 189836 (06-13), 190730 (07-19), 193354 (12-07) |
| 2023 | 193542 (01-04), 195122 (04-10), 196009 (05-31), 196639 (06-29) |
| 2024 | 198351 (05-03), 200009 (08-12), 201068 (09-26), 202180 (12-04) |
| 2025 | 202726 (04-09), 203505 (05-28), 204114 (06-26), 204925 (08-08) |

`scripts/labelmaker/fdp_probe.py` calls the existing
`resolve_fdp._fetch_ptdata` / `_fetch_mds` helpers, and checks the resulting
one-dimensional time axis with `_scalar_axes`. It preserves the original
shape, dtype, units, dimension shape/range, time step, finite counts, and
untruncated exception class/message. The positive control is canonical PTDATA
`ip`, readable on 20/20. A returned signal is not necessarily an active
actuator; a missing record is never an off measurement.

The shared environment was invoked with its required wrapper, adding the
read-only-environment flags `--frozen --no-install` and importing this
worktree's `src` explicitly:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
PYTHONPATH=/scratch/gpfs/nc1514/FusionAIHub-Ifix/src \
XDG_CACHE_HOME=/tmp/LCprobe/cache \
pixi run --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml \
  --frozen --no-install -e labelmaker fdp run python \
  /scratch/gpfs/nc1514/FusionAIHub-Ifix/scripts/labelmaker/fdp_probe.py \
  --shots /tmp/LCprobe/sample.json \
  --candidates /tmp/LCprobe/candidates.json \
  --output /tmp/LCprobe/probe.jsonl
```

The second pass uses the same command with `candidates-index.json` and
`probe-index.jsonl`. Output files are created exclusively, so a rerun needs a
new output filename. Discovery and summaries also live under `/tmp/LCprobe`.
Corpus and feature HDF5 files were opened read-only. An initial ordinary Pixi
invocation began dependency validation and was stopped; all actual probe and
census execution uses the existing environment without installation or lock
updates. No environment repair, host service change, or credentials change was
attempted.

## How candidate names were found

1. Existing configuration supplies bare C/I-coil PTDATA names in
   `configs/ideate/actuators.yaml`; `scripts/data_fetching_omega/config_atlas.yaml`
   supplies the I-coil toroidal-harmonic paths. Canonical `ip` comes from
   `src/labelmaker/features/namespace.py`.
2. Read-only MDSplus tree metadata for shot 204925: `rf` (1,478 nodes),
   `pellet` (74), `neutrals` (407), `operations` (194), and `d3d` (245,927,
   metadata only). This finds `RF::TOP.TWA`, `RF::TOP.LHCD`, the lithium-granule
   nodes, and C/I-coil harmonics. `neutrals::TOP.PELLETS` has no child records in
   this listing. Standalone trees named `lhcd`, `helicon`, `pcs`, and
   `engineering` fail with E4 below; their failure does not establish that the
   subsystem is absent from other trees.
3. Native `PtDataReader.list_pointnames` failed for the attempted extensions.
   Instead, read the **single-shot metadata JSON**, through FDP's origin, for
   200009, 203505, and 204925. The wrapper selects
   `pelican://osg-htc.org:443/fdp-d3d/archives/index/json`, pattern
   `json_indexes_*`; the newest listed directory is
   `json_indexes_2026-06-23_15:20:51`. The three JSON files are 602,614, 673,308,
   and 576,211 bytes. Their `pointname_ext` maps reveal the second-pass names,
   including `ECOILFWDCL`/`ECOILREVCL` (`.CRI`) and PCS pellet/Helicon candidates.
   These are directory metadata reads, not waveform downloads.
4. Brief-inspired spellings such as `helicon`, `pellet`, `pifire`, `epso`,
   `iecoil`, and `lhpower` were also tried explicitly. They are search
   hypotheses, not confirmed aliases. No wildcard was passed as a signal name.
   Table B distinguishes these from the index-discovered points.

The PTDATA fallback failure is material: an unindexed point can fall through
to a `PTSERVER` route whose service lookup fails **even inside `fdp run`**.
The success of `ip`, `ecoil`, and the other indexed points demonstrates partial
access. E1 must be reported as **unreachable**, not “point does not exist.”

Exact error dictionary used in the point tables:

| Code | Exact exception |
| --- | --- |
| E1 | `PtDataError: Error opening network connection: getservbyname failed for task 'PTSERVER' (also tried 'ptserver'): not in /etc/services` |
| E2 | `TreeNNF: %TREE-W-NNF, Node Not Found` |
| E3 | `TreeNODATA: %TREE-E-NODATA, No data available for this node` |
| E4 | `TreeFOPENR: %TREE-E-FOPENR, Error opening file read-only.` |
| E5 | `TdiINVCLADSC: %TDI-E-INVCLADSC, Storage class not valid, must be scalar or array` |

Directory-listing errors were exactly
`PtDataError: Shot.extension not found: Shot 204925.PLA` (default source),
and the same error with `204925.PCA`, `204925.PCS`, `204925.D3D`;
also `{185786,204925}.{HEL,LPI,EPS,MAG,PC0}`, each attempted separately.
The later successful JSON-index reads supersede the inability to enumerate
names, but do not repair the E1 point-fetch route.

## Interpretation and proposed definitions

Every future rule in this section has `evidence_kind="heuristic"`. All nine
rows remain **`not_started` for the requested family of on/off plus regime
labels**. Existing on/off heuristics alone do not meet the workstream's full
registry, source-coverage, ideate, test, and 500-shot-count contract. The probe
does not justify `no_data_in_corpus` for any of these nine rows. In particular,
0/20 is not proof of absence across the 500 or across every shot in 2020–2025.

Use the existing Schmitt convention for continuous quantities: enter at
`T_on`, exit below `T_on / 2`, retain state between the two thresholds. Existing
continuous-actuator settings are a minimum 20 ms duration and bridging gaps
strictly shorter than 20 ms. Assign mutually exclusive fixed physical regimes
to measured intervals; do not recompute boundaries from population tertiles.
Coverage is the finite measured time support of the specific input(s), with
unknown outside it; do not extend RF pulses to an assumed whole-shot clock.
Do not bridge missing measurement coverage or infer negatives from absent
shot records. The nominal low-band entry bounds below include the retained
hysteresis tail in the low regime while the actuator remains on. The PCS
clocks below contain gaps up to about four seconds despite much shorter median
sample steps; segment those gaps before resampling or interval extraction.

| Row | Status | Proposed rule and concrete next step |
| --- | --- | --- |
| NBI | `not_started` | Total injected power; on/off 0.5/0.25 MW, low 0.5–3, medium 3–7, high ≥7 MW. Add regimes and complete the workstream's source/registry/ideate tests and 500-shot ledger; keep `tinj_total` for the separate counter-current qualifier. |
| ECH | `not_started` | Total ECH power; on/off 0.1/0.05 MW. Use low 0.1–1, medium 1–2, high ≥2 MW. First replace the event path's plain channel sum with the canonical NaN-safe reduction described below; preserve the W-to-MW conversion and source provenance. Then complete regimes and the downstream contract. |
| Gas | `not_started` | Separate summed calibrated flow, summed valve command, and the existing `gasa` command heuristic. Use the census's flow proposal only with its stored-unit qualification. Preserve existing `gasa` 0.5/0.25 V on/off as a command proxy; never call its voltage “total gas.” |
| RMP | `not_started` | `max_coil(abs(I))` in kA; on/off 0.5/0.25 kA, low 0.5–1.5, medium 1.5–3, high ≥3 kA. This marks I-coil actuation, not proof of resonant-field coupling or ELM suppression. Complete regimes and coverage integration. |
| Helicon | `not_started` | Use an identified delivered/coupled TWA power, not its PCS enable or klystron output indiscriminately. After confirming units and the meanings of TWAPWR/TWAPWRC/TWAPWRO, provisional on/off 0.02/0.01 MW and regime edges 0.2/0.5 MW distinguish commissioning-scale from higher-power operation. The immediate step is an RF calibration/loader check on 200009 and 201068; no conversion from the blank unit field is approved here. |
| Pellet | `not_started` | Identify material, injector, and measured arrival/fire event separately from a requested PCS sequence. Schmitt pulse extraction must use confirmed logic levels and native pulse duration; the 20 ms continuous-actuator filter may erase pellet triggers. Regimes should be fixed cadence and/or delivered-mass bands after units and injector limits are known; numeric bins are deliberately unset. Next: resolve the PCS channel definitions against injector timing/arrival data, and assess fuel pellets versus lithium granules versus shattered pellets separately. |
| EFC | `not_started` | Candidate n=1 current proxy: `max(A1_C,A1_IU,A1_IL)` in kA, provisional on/off 0.1/0.05 kA, edges 0.5/1 kA. Confirm the n=1 amplitudes from the corpus channels against the measured OPERATIONS nodes and carry a mode/intent qualification before naming it EFC. Amplitude alone does not establish correction of the intrinsic error field. |
| Current Saturation | `not_started` | If ECOILFWDCL/REVCL are documented limit-state monitors, use their verified active polarity and Schmitt voltage levels, gated to the appropriate control phase. Otherwise, once the effective signed limit is known, use normalized E- or F-coil current: enter ≥0.98, exit <0.95; provisional headroom bands <0.8, 0.8–0.95, 0.95–0.98, ≥0.98. This uses tight hysteresis rather than the factor-two power convention. Next: establish monitor/control semantics, validate a known limit encounter, and confirm the user's E-coil versus F-coil meaning. |
| LH | `not_started` | `RF::LH_POWER`, converted from recorded kW to MW; provisional on/off 0.01/0.005 MW, low 0.01–0.05, medium 0.05–0.15, high ≥0.15 MW. Check the RF loader's definition (net coupled power versus forward power) and `LH_INTOD3` routing before labeling injection; then add canonical source, coverage, registry, ideate tests and a measured 500-shot count. |

### EFC and the existing corpus

The producer's `i_coil` order is C19F, C79F, C139F, C199F, C259F, C319F,
IU30F, IU90F, IU150F, IU210F, IU270F, IU330F,
IL30F, IL90F, IL150F, IL210F, IL270F, IL330. The distinct `rmp` group contains
only the twelve I-coils. Thus a new `c_coil` corpus group is **not inherently
required** to form an n=1 current proxy: all three six-coil rings are already
represented in `i_coil`, which is an IGNITE actuator input.

For a six-coil ring with verified physical ordering and sign convention, the
candidate Fourier amplitude is
`A1 = (2/6) * abs(sum_j I_j * exp(-i * phi_j))`, with angles C=19+60j degrees
or I=30+60j degrees. Evaluate upper, lower, and external rings separately;
do not sum signed currents into cancellation, and do not equate this current
amplitude to a calibrated resonant magnetic-field amplitude. The existing
`actuators.yaml` documents historical `*F` naming/alias pitfalls, including
C19F/C79F duplication in an older era, so neither channel names alone nor a
blind Fourier transform certifies EFC. The corpus census supplies presence and
shape, not a new point-by-point calibration of all 18 channels. In the
500-shot census the first 17 channels are finite on all 493 group-valid
shots, but bare `IL330` is finite on only 254. The `rmp` alias `IL330F` is
finite on 246 with nearly complementary year coverage. Reconcile those
pointnames and their calibration before a complete lower-ring transform;
missing channels must not be silently treated as zero current.

The directly measured alternatives are `\OPERATIONS::CN1IAMP`,
`\OPERATIONS::IUN1IAMP`, and `\OPERATIONS::ILN1IAMP`, each with a millisecond
clock and `Amps` units. These give a concrete reference for the next validation
step. n=1 applied fields are also used for locked-mode phase/rotation control;
therefore their presence does not uniquely identify error-field correction.
[DIII-D control experiment](https://arxiv.org/abs/1801.05012).

### Verdict: LH = lower-hybrid current drive

**Accept this reading for the Actuation row; data exist within the selected
years.** `\RF::TOP.LHCD:LH_POWER` (tag `\RF::LH_POWER`) is positive on shot
203505, with peak 223.95677 kW, and `LH_PHASE` is 90 degrees. The sibling
`LH_INTOD3` is a scalar record `1` on that shot, found by a read-only metadata
read. Its waveform fetch fails with E5 because it has no time dimension:
that is not a missing scalar. Its routing semantics still need loader
confirmation before it is used as an injection gate.

DIII-D's 2025 commissioning plan explicitly identifies a 4.6 GHz HFS LHCD
system, and a November 2025 experimental abstract reports initial coupled
power and nonthermal-electron observations. These support the interpretation
but do not certify every shot or the calibration of our particular record.
[Facility commissioning plan](https://d3dfusion.org/2025-26-01/),
[first experimental results](https://meetings-archive.aps.org/dpp/2025/no05/2/).
The alternative L→H confinement transition already belongs to the separate
`dalpha_lh` regime work. Helicon is the distinct 476 MHz traveling-wave
system, despite both being RF current-drive concepts.
[General Atomics helicon description](https://www.ga.com/diii-d-scientists-at-ga-develop-new-system-to-improve-production-of-fusion-energy).

### Verdict: Current Saturation = ohmic coil at its current limit

**Physically plausible, but not confirmed as either the intended label or a
measurable limit condition by this probe.** PTDATA `ecoil` is a genuine
current record on all 20 shots; the largest native absolute sample is
132.72734 kA. Reaching a current maximum, a flat plasma-current waveform, and
exhausting usable inductive flux are different statements. A plateau can be
commanded, and loss of flux headroom depends on the discharge trajectory.

DIII-D's engineering description makes the OH/TF force constraint depend on
the product of their currents. Its historical static ±87 kA OH limit must
not be promoted to a universal threshold for these later discharges.
[OH-coil force-limit protection](https://web.gat.com/pubs-ext/MISCONF97/A22705.pdf).
The measured forward/reverse limit-related voltages and PCS bounds below
require control documentation and a physical calibration; their names alone
do not provide an A-valued limit.

The alternative **F-coil/power-supply saturation** is credible: F1A, F7B, and
F9B are all measured, and DIII-D's 2023 negative-triangularity campaign
reported F7B/F9B supply and patch-panel constraints affecting shape control.
[Campaign paper](https://doi.org/10.1088/1361-6587/ad6f40).
Distinguish current limiting from voltage-command clipping, and include the
applicable power-supply/patch configuration. Neither interpretation gets a
label until a known saturation case validates that distinction.

## Corpus census and fixed physical bins

### Scope and population

This is a read-only census of the 500 unique shots in
`configs/ideate/shot_lists/recommender_v1.yaml`. All 500 corpus files opened.
The selected shots span 2021-04-12 through 2025-08-08: 64 in 2021, 130 in
2022, 94 in 2023, 116 in 2024, and 96 in 2025. The manifest contains no 2020
shot. `run_id`'s leading date agrees with the manifest year on every shot.

Three populations are reported and must not be conflated:

1. **Corpus time samples** are exact 1 ms bin means within every valid record,
   pooled across shots. This population weights a shot by record duration.
   `p33` and `p67` are the two population tertiles, calculated by
   `numpy.percentile(method="linear")`.
2. **Feature-store time samples retain their stored cadence:** archive-resolved
   groups remain at 25 ms while corpus-resolved groups are at 1 ms. Their pooled
   population is therefore mixed-cadence and weights shots by record duration,
   cadence, and resolver; it is reported separately from the corpus population.
3. **Shot summaries** contain one 1 ms-bin peak per valid corpus shot, or
   one peak at stored cadence per valid feature-store shot. Every shot has
   equal weight. RMP additionally has the requested
   **native per-shot/per-coil absolute maximum** population (one value per
   finite shot-coil).

Corpus aggregation first computes a finite-sample mean in each channel's
1 ms bin, then sums finite channels for total power/flow or takes the maximum
absolute finite coil current for the RMP set level. An all-missing bin remains
NaN. “Total” therefore means the sum of available channels; it is not a claim
that every installed source is measured.

“Active” uses the existing Schmitt-trigger on threshold before hysteresis:
NBI 0.5 MW, ECH 0.1 MW, RMP 0.5 kA, and current gas feature `gasa` 0.5 V.
For the alternative calibrated `gas_flow` total, 10 stored units is shown only
as a provisional activity boundary; no governed threshold exists.

### Stored schema, units, and missingness

The corpus HDF5 groups contain only `xdata` and `(channels, time) ydata`; the
files and groups have **no attributes**, so units are not discoverable from
the stored files. Unit statements below come from the corpus producer config
and labelmaker namespace:

- `pinj`: 8 beams in W; sum finite channels and divide by `1e6` for MW.
- `ech_power`: 12 gyrotrons in W; sum finite channels and divide by `1e6` for
  MW. The canonical feature store writes `ech_power_total` in W.
- `gas_flow`: 11 calibrated `...:FLOW` channels. The registry documents the
  source leaves as Torr.L/s, but the corpus itself stores no unit metadata;
  tables therefore call these “stored flow units” and preserve the raw scale.
- `gas_raw`: 11 raw valve commands in V. The current labelmaker actuator is
  only channel 0, PTDATA `gasa`, rather than a total.
- `rmp`: 12 I-coil currents stored on the A scale; divide by `1000` for kA.
  The actuator level is the maximum absolute current across finite coils.
- `i_coil`: 18 currents on the A scale (six C-coils, then the same 12
  I-coils); divide by `1000` for kA.

The corpus missing-signal sentinel is `ydata.shape[-1] < 2`. Every sentinel
seen here is `(C, 1)` and all-NaN; it is missing, not an actuator value of
zero. A missing group is separately recorded as `KeyError`.

| group | valid / 500 | missing | stored shape/rate | finite shot-channels |
|---|---:|---|---|---:|
| `pinj` | 465 | 35 `(8,1)` sentinels | `(8,131002)`, 10 kHz, 0–13.1001 s | 3720 / 3720 |
| `ech_power` | 499 | group absent on 200009 | `(12,101002)` or `(12,102501)`, 10 kHz, start −0.10 or −0.25 s, end 10 s | 5167 / 5988 |
| `gas_flow` | 499 | `(11,1)` sentinel on 203444 | `(11,216001/262145/1048577)`, 10 kHz, start −14.607 or −10 s, end 11.60–94.858 s | 2946 / 5489 |
| `gas_raw` | 499 | `(11,1)` sentinel on 203444 | same clocks/shapes as `gas_flow` | 4926 / 5489 |
| `rmp` | 499 | group absent on 200009 | `(12,112641)`, 10 kHz, start −1.149…−0.906 s, end 10.115–10.358 s | 5735 / 5988 |
| `i_coil` | 493 | group absent on 7 2021 shots | `(18,563200)` or `(18,1200639)`, 50 kHz, start −4.170…−0.906 s, end 10.115–20.086 s | 8635 / 8874 |

The seven missing `i_coil` shots are 186055, 186194, 186196, 186223,
186861, 186867, and 186904.

The canonical feature store has `pinj_total` on 473/500 shots (171 archive,
302 corpus; 27 recorded misses), `ech_power_total` on 499/500 (144 archive,
355 corpus; shot 200009 missing), and `gas` on 499/500. Every `gas` group is
corpus-resolved with units `V`, locator `gas_raw#0`, and a 1 ms clock; shapes
are `(1,21601)` on 215 shots, `(1,26215)` on 64, and `(1,104858)` on 220.
Shot 203444 records the only `gas` miss as `corpus:SignalAbsent`. Units are
present on every stored NBI/ECH group: `kW` and `W`, respectively. Archive
groups are `(1,240)` at 25 ms; corpus groups are `(1,13101)` for NBI and
`(1,10101)` or `(1,10251)` for ECH at 1 ms. Corpus-resolved feature peaks
agree with this census to sub-watt precision after unit conversion (maximum
absolute differences below 0.5 W NBI and 0.12 W ECH).

An exhaustive key enumeration over the 500 feature files found no stored
`gas_total`, `gas_flow`, `gas_flow_total`, `gas_raw`, `rmp`, `rmp_total`,
`i_coil`, or `icoil` group, and none of those names appears in a file's
recorded-missing map. The only feature keys containing `gas`, `rmp`, or
`coil` are `gas` (499 files) and unrelated B-coil feature `pcbcoil` (500).
Thus calibrated total gas and RMP/i-coil values in this report come only from
the raw corpus census.

Stored `gas` shot peaks agree with the corpus `gasa` calculation within
`4.77e-7 V` (499 comparisons). Over all 499 stored peaks, p33/p50/p67 are
3.801/4.470/5.433 V, p95 10.020 V, p99 10.069 V, max 10.306 V. The 495
active stored peaks have p33/p50/p67 3.845/4.474/5.437 V, matching the
active corpus `gasa` table to its printed precision. No second total-flow
or RMP feature-store distribution exists under the audited keys.

### Time-sample distributions

Corpus values are 1 ms bin means; feature-store values mix unchanged 25 ms
archive records with 1 ms corpus records. “All” includes off/background time.
“Active” is conditioned on the threshold above. The `p33` and `p67` columns
are the tertiles.

| quantity / population | n | p33 | p50 | p67 | p75 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| corpus NBI all (MW) | 6,091,500 | 0 | 0 | 0 | 2.281 | 8.811 | 12.019 | 16.424 |
| corpus NBI active (MW) | 1,834,513 | 3.568 | 5.039 | 6.628 | 7.587 | 11.401 | 13.476 | 16.424 |
| feature-store NBI all (MW) | 3,995,173 | 0 | 0 | 0 | 2.127 | 8.259 | 11.555 | 15.762 |
| feature-store NBI active (MW) | 1,183,623 | 3.096 | 4.672 | 6.264 | 7.230 | 10.910 | 13.094 | 15.762 |
| corpus ECH all (MW) | 5,093,949 | 0 | 0 | 0 | 0.00029 | 1.713 | 2.115 | 2.488 |
| corpus ECH active (MW) | 657,277 | 1.020 | 1.558 | 1.761 | 1.946 | 2.137 | 2.266 | 2.488 |
| feature-store ECH all (MW) | 3,637,239 | 0 | 0 | 0 | 0.00034 | 1.679 | 2.106 | 2.633 |
| feature-store ECH active (MW) | 452,928 | 1.027 | 1.572 | 1.741 | 1.786 | 2.137 | 2.293 | 2.633 |
| calibrated gas total, all (stored flow units) | 29,390,520 | 0.0365 | 0.0711 | 0.149 | 0.280 | 27.074 | 148.680 | 1690.544 |
| `gasa`, all (V) | 29,390,520 | 0.00069 | 0.00551 | 0.0114 | 0.0143 | 0.112 | 3.526 | 10.306 |
| `gasa`, active (V) | 1,205,750 | 1.634 | 2.126 | 2.977 | 3.475 | 6.267 | 9.955 | 10.306 |
| RMP set level, all (kA) | 5,620,736 | 0.00990 | 0.0129 | 0.0190 | 0.0288 | 2.013 | 4.809 | 6.872 |
| RMP set level, active (kA) | 938,006 | 1.222 | 1.510 | 1.935 | 2.147 | 5.153 | 6.346 | 6.872 |

The all-time ECH populations contain negative digitizer background
(1,018,489/5,093,949 corpus bins and 783,558/3,637,239 feature-store bins).
Those values do not affect the active distribution, but the namespace note
that negative ECH is clamped to zero is not implemented by the current
corpus resolver.

### Equal-weight shot-peak distributions

All percentile and maximum columns below describe the active-shot subset.
`n valid` remains the availability denominator; it includes inactive shots.

| quantity | n valid | n active | active p33 | p50 | p67 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| corpus NBI peak (MW) | 465 | 465 | 6.948 | 8.401 | 10.015 | 14.460 | 15.478 | 16.424 |
| feature-store NBI peak (MW) | 473 | 465 | 6.316 | 7.917 | 9.609 | 13.430 | 15.234 | 15.762 |
| corpus ECH peak (MW) | 499 | 248 | 1.118 | 1.661 | 1.853 | 2.217 | 2.381 | 2.488 |
| feature-store ECH peak (MW) | 499 | 246 | 1.159 | 1.678 | 1.871 | 2.332 | 2.529 | 2.633 |
| calibrated gas total peak (stored flow units) | 499 | 499 at provisional ≥10 | 70.806 | 89.379 | 111.927 | 235.517 | 443.507 | 1690.544 |
| current `gasa` peak (V) | 499 | 495 | 3.845 | 4.474 | 5.437 | 10.021 | 10.069 | 10.306 |
| RMP set peak (kA) | 499 | 287 | 1.540 | 2.032 | 2.719 | 5.797 | 6.560 | 6.872 |

For completeness, summed `gas_raw` command-voltage shot peaks over all 499
valid records have p33/p50/p67 = 4.139/5.410/7.035 V, p95 = 14.623 V,
p99 = 20.628 V, and max = 62.676 V. This sum is descriptive only; offsets
and different valve responses make it unsuitable as a physical total-gas
label. It must not be confused with the single-channel `gasa` rows above.

The gas total has a long tail and all 499 valid shot peaks exceed the
provisional 10-unit boundary. Because only 53.7% of possible `gas_flow`
shot-channels contain any finite sample, this finite-channel sum is a mixed
population: its magnitude can change when a valve channel is absent. It is
not safe to treat its tertiles as a calibrated machine-wide physical total
until channel expectations and missing-channel handling are fixed.

### RMP native per-shot/per-coil absolute maxima

There are 5,735 finite shot-coil maxima out of 5,988 possible. Pooled across
all coils, the all-value tertiles are 0.0255/1.391 kA (median 0.380); among
the 2,809 active values (≥0.5 kA), tertiles are 1.417/2.624 kA (median
1.690, p95 5.530, max 6.878). Values below are per coil over all finite
shot maxima, so the inactive mode remains visible.

| coil | shots | p33 | p50 | p67 | p95 | max (kA) |
|---|---:|---:|---:|---:|---:|---:|
| IU30F | 499 | 0.0297 | 0.311 | 1.306 | 4.769 | 6.806 |
| IU90F | 499 | 0.0293 | 1.007 | 1.514 | 4.886 | 6.788 |
| IU150F | 499 | 0.0279 | 1.004 | 1.467 | 4.822 | 6.790 |
| IU210F | 499 | 0.0262 | 0.237 | 1.231 | 4.716 | 6.516 |
| IU270F | 499 | 0.0264 | 0.967 | 1.496 | 4.800 | 6.782 |
| IU330F | 499 | 0.0243 | 0.645 | 1.443 | 4.765 | 6.878 |
| IL30F | 499 | 0.0243 | 0.876 | 1.467 | 4.799 | 6.806 |
| IL90F | 499 | 0.0257 | 0.283 | 1.316 | 4.882 | 6.526 |
| IL150F | 499 | 0.0236 | 0.967 | 1.484 | 4.833 | 6.662 |
| IL210F | 499 | 0.0202 | 0.030 | 1.204 | 4.580 | 6.778 |
| IL270F | 499 | 0.0251 | 0.288 | 1.282 | 4.772 | 6.610 |
| IL330F | 246 | 0.0280 | 0.958 | 1.510 | 4.432 | 6.390 |

The 2021–2025 population does not show the repository's warned-about
~`1e5` raw-current scale: the largest stored value is 6,878 A. RMP program
mix changes by year, however: the shot-level median set peak is 0.018 kA in
2021, 0.083 in 2022, 1.614 in 2023, 1.111 in 2024, and 1.448 in 2025.

### `i_coil` coverage relevant to EFC

The producer channel order is:

`C19F, C79F, C139F, C199F, C259F, C319F, IU30F, IU90F, IU150F,
IU210F, IU270F, IU330F, IL30F, IL90F, IL150F, IL210F, IL270F, IL330`.

The first six channels are C-coils despite the group name. All first 17
channels have finite data on every group-valid shot: 493/500 overall
(57/64 in 2021, then 130/130, 94/94, 116/116, 96/96). Their stored units are
not attributed in HDF5; producer/actuator configs identify amperes, and the
largest native absolute value in each of these channels ranges from about
5.58 to 6.88 kA after conversion (the range of channel maxima, not a
lower bound on individual shot peaks).

Channel 18 is a naming/era trap. `i_coil` calls it bare `IL330` and it is
finite on only 254 shots: 0 in 2021, 76 in 2022, 4 in 2023, 78 in 2024, and
96 in 2025. Conversely, `rmp` calls its twelfth channel `IL330F` and it is
finite on 246 shots: 64 in 2021, 54 in 2022, 90 in 2023, 38 in 2024, and 0
in 2025. These nearly complementary windows are evidence of the documented
pointname-era mismatch, not absence of the physical coil. Any EFC/n=1
derivation should use the six C-coil channels and explicitly reconcile the
bare/`F` I-coil naming rather than treating channel 18 NaN as coil-off.

### Proposed fixed physical bins

These are fixed rounded thresholds, not fitted tertiles:

- **NBI:** off `<0.5`, low `0.5–<3`, medium `3–<7`, high `≥7 MW` (the
  user's fixed choice). Shot-peak counts among 465 valid corpus records are
  0/43/114/308. Active-time tertiles 3.57/6.63 MW support the two regime
  boundaries without defining them.
- **ECH:** off `<0.1`, low `0.1–<1`, medium `1–<2`, high `≥2 MW`.
  Active-time tertiles are 1.02/1.76 MW and active-shot tertiles are
  1.12/1.85 MW, so 1 and 2 MW are stable rounded physical boundaries.
  Shot-peak counts among 499 valid records are 251/59/131/58.
- **Gas, current implemented `gasa` semantics:** off `<0.5`, low
  `0.5–<1.5`, medium `1.5–<3`, high `≥3 V`. Active-time tertiles are
  1.63/2.98 V, supporting rounded 1.5 and 3 V boundaries. Shot-peak counts
  are 4/6/78/411; the peak distribution is top-heavy because brief puffs
  reach full command voltage, so interval-level mean/max should retain the
  regime evidence.
- **Gas, if the requested label is instead calibrated total flow:** a
  provisional set is off `<10`, low `10–<50`, medium `50–<100`, high
  `≥100 stored flow units` (source config indicates Torr.L/s). Shot-peak
  counts are 0/101/198/200 and shot tertiles 70.8/111.9 motivate 50/100.
  Do not promote these to canonical physical bins until the 46.3%
  shot-channel sparsity and units metadata are resolved.
- **RMP:** off `<0.5`, low `0.5–<1.5`, medium `1.5–<3`, high `≥3 kA` using
  the maximum absolute coil current at each time. Active time tertiles are
  1.22/1.94 kA, while active shot-coil maximum tertiles are 1.42/2.62 kA;
  1.5 and 3 kA are rounded boundaries that serve both views. Shot-level set
  peak counts among 499 records are 212/86/110/91.

### Data-path findings and follow-up

1. **The ECH event input reduction produces no finite total (high
   confidence/high impact).** Across all 50,935,141 native timestamps in 499 valid
   `ech_power` records, at least one of the 12 channels is NaN. The event
   pipeline uses plain `y.sum(axis=0)`, producing an all-NaN total; the
   canonical corpus resolver correctly uses `nansum` and returns NaN only
   when every channel is NaN. A later implementation task should make the
   event path use the canonical resolver or duplicate its nan-safe rule.
   This is a data-and-code finding, not a measured event-generation count;
   no event pipeline was run. This census implements no labels or code fix.
2. **Gas definition must be explicit (high confidence/medium impact).** The
   current event is `gasa` raw command voltage. `gas_flow` is a different,
   calibrated 11-valve quantity with severe channel sparsity. Summing all
   `gas_raw` voltages is also unsuitable: 14,469,863 of 29,390,520 1 ms bins
   are negative due to channel offsets, and the channels are commands for
   distinct valves.
3. **RMP and `i_coil` missing channels are not actuator-off values (high
   confidence/medium impact).** The IL330/IL330F complementary coverage is
   a namespace-era issue. Nan-safe per-coil reduction is required.

### Reproducibility and artifacts

All census invocations used the existing shared environment with
`--frozen --no-install`; Python bytecode and caches were redirected/disabled.
No SLURM, network, labels, commits, or writes to corpus/feature roots were
performed.

```bash
cd /scratch/gpfs/nc1514/FusionAIHub-Ifix
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export XDG_CACHE_HOME=/tmp/LCprobe_ctx_cache PIP_CACHE_DIR=/tmp/LCprobe_pip_cache
export HDF5_USE_FILE_LOCKING=FALSE
pixi run --frozen --no-install \
  --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml \
  -e labelmaker python /tmp/LCprobe_actuator_census_20260913.py \
  --shot-list configs/ideate/shot_lists/recommender_v1.yaml \
  --corpus /scratch/gpfs/EKOLEMEN/foundation_model \
  --features /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/features \
  --output-prefix /tmp/LCprobe_actuator_census_20260913 \
  | tee /tmp/LCprobe_actuator_census_20260913_run.log
pixi run --frozen --no-install \
  --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml \
  -e labelmaker python /tmp/LCprobe_actuator_census_focus_20260913.py
pixi run --frozen --no-install \
  --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml \
  -e labelmaker python /tmp/LCprobe_feature_actuator_scope_20260913.py \
  --shot-list configs/ideate/shot_lists/recommender_v1.yaml \
  --features /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/features \
  --existing-shot-summary /tmp/LCprobe_actuator_census_20260913_shot_summary.csv \
  --output /tmp/LCprobe_feature_actuator_scope_20260913.json \
  | tee /tmp/LCprobe_feature_actuator_scope_20260913.log
```

Artifacts:

- `/tmp/LCprobe_actuator_census_20260913.py` — full read-only census script
- `/tmp/LCprobe_actuator_census_focus_20260913.py` — derived-table script
- `/tmp/LCprobe_actuator_census_20260913_results.json` — full machine-readable summary
- `/tmp/LCprobe_actuator_census_20260913_focused.json` — active, year, and bin views
- `/tmp/LCprobe_actuator_census_20260913_shot_summary.csv` — one row per shot
- `/tmp/LCprobe_actuator_census_20260913_coil_summary.csv` — one row per coil/group
- `/tmp/LCprobe_actuator_census_20260913_run.log` — progress and completion log

SHA-256 at delivery: full script
`d881c5a2845b57788a6f40470aeca0b88332755c3ae5cce96aea1cc72d23a782`;
results JSON
`7a900851c8d9229615b64b007a2b6135fe12b4659b6ded9f368bd43b251abfef`.

## Per-point probe measurements

The two passes made **1,240 serial point/shot requests** (47 initial + 15
index-discovered candidates, each on 20 shots): 548 returned records and 692
fetch errors. Availability below counts returned records, including constant
records; it is not the number of shots with actuation. Every returned waveform
has a `times` dimension of the same length with units `ms` and a finite,
strictly increasing clock. Every actual data length is listed; `Δt` is the
median step, not a promise of uniform sampling. The t0 and t1 ranges are the
minimum and maximum endpoints across readable shots, **not continuous common
coverage**. Full unrounded per-shot values and errors are retained in
`/tmp/LCprobe/point-shot-table.csv` and the two JSONL files.

- Helicon RF records: only 200009 and 201068. On 200009, TWAPWR has 984 samples,
  1999.526–2097.826 ms; TWAPWRC/TWAPWRO have 983 samples,
  1999.576–2097.776 ms. On 201068 these are 1876 and 1875 samples,
  5499.540–5687.040 and 5499.590–5686.990 ms, respectively. Median Δt ≈0.1 ms.
  TWAPWR peaks are 302024.59 and 899796.00 **in undeclared units**;
  TWAPWRC peaks are 212335.33 and 509077.69. No unit conversion was applied.
  D1CHELICON/DOCHELICON are zero on all 15 readable shots, including those
  with RF records; PCHELISTS is approximately 3.54–5.48 `raw`. None is a
  validated substitute for measured power.
- LH_POWER and LH_PHASE: only 203505, shape (81920,), times
  2514.500–2842.176 ms, median Δt 0.004 ms; power 0–223.95677 kW,
  phase constantly 90 Deg. LH_INTOD3 has a scalar record `1` on the same
  shot, despite its expected E5 from the waveform resolver. Metadata also
  contains LH_COORDS `[1,2,3]`; no coordinate semantics were inferred.
- Pellet's four LGI event/mass candidates have E3 on 20/20. The metadata says
  material `Lithium`, drop and impact times in ms, and hit mass in µg; these
  are descriptions, not units measured from a populated record. ONIPELON and
  ONSPELSQ are constantly zero on 20/20; PDTPELSIZE is zero on its four
  readable shots. PEIPELIN1 reaches 1 on six of eleven readable shots
  (186892, 188734, 189836, 198351, 202180, 204925); PESNPELL reaches 1 on
  its only readable shot, 204925. These PCS names/values do not certify
  delivery, material, or physical units. IPELI is a substring-discovered
  candidate only; its relationship to pellets is **unconfirmed**.
- ECOILFWDCL/REVCL return only on the four sampled 2025 shots, in V, at
  0.25 ms: 88,000 samples for the first and 118,000 for the remaining three.
  Maxima are about 4.55–4.56 V. Header ASCII is blank, so active polarity and
  engineering meaning remain unverified. IPTECOILMX/MN are constantly
  +2047/−2047 `vo` on 20/20; these are not measured A-valued limits.
  IPTECOIL and IPXOECOIL are zero on 20/20. The latter is a 7- or 8-point,
  approximately 1-second clock and is not a fast saturation monitor.
- PCS `TIMEPCS0` signals have median Δt 0.25 ms but gaps as large as
  3999.75 ms; the fast PCS clock has median Δt 0.05 ms but gaps up to
  3999.90 ms. A median-step-based event extractor must not join across these
  gaps. The full clock statistics are retained per point and shot.

### A. Resolver/configuration, tree-listed, and explicit search candidates

| Row | Point (tree-qualified MDS; otherwise PTDATA) | Returned /20 | Actual data shape(s); dtype | Raw data units | Median Δt (ms); t0 and t1 ranges (s) | Fetch errors (count) |
| --- | --- | ---: | --- | --- | --- | --- |
| control | `ip` | 20 | (30720,), (480256,); float64 | `'a'` | 0.05…0.5; t0 -4.158641…-1.007126; t1 14.2469…20.05284 | none |
| Helicon | `\RF::TWAPWR` | 2 | (984,), (1876,); float32 | `' '` | 0.09997559…0.1000977; t0 1.999526…5.49954; t1 2.097826…5.68704 | E2 (5), E3 (13) |
| Helicon | `\RF::TWAPWRC` | 2 | (983,), (1875,); float64 | `' '` | 0.09997559…0.1000977; t0 1.999576…5.49959; t1 2.097776…5.68699 | E3 (18) |
| Helicon | `\RF::TWAPWRO` | 2 | (983,), (1875,); float32 | `' '` | 0.09997559…0.1000977; t0 1.999576…5.49959; t1 2.097776…5.68699 | E2 (5), E3 (13) |
| Helicon | `\RF::TWAPWRKL` | 2 | (984,), (1876,); float32 | `' '` | 0.09997559…0.1000977; t0 1.999526…5.49954; t1 2.097826…5.68704 | E2 (5), E3 (13) |
| Helicon | `\RF::TWAPWR15F` | 2 | (984,), (1876,); float64 | `' '` | 0.09999847…0.09999847; t0 1.999526…5.49954; t1 2.097826…5.68704 | E2 (5), E3 (13) |
| Helicon | `\RF::TWAPWR15R` | 2 | (984,), (1876,); float32 | `' '` | 0.09997559…0.1000977; t0 1.999526…5.49954; t1 2.097826…5.68704 | E2 (5), E3 (13) |
| Helicon | `twapwr` | 0 | — | `—` | — | E1 (20) |
| Helicon | `twapwrc` | 0 | — | `—` | — | E1 (20) |
| Helicon | `helicon` | 0 | — | `—` | — | E1 (20) |
| Helicon | `heliconp` | 0 | — | `—` | — | E1 (20) |
| LH | `\RF::LH_POWER` | 1 | (81920,); float64 | `'kW'` | 0.004; t0 2.5145; t1 2.842176 | E2 (13), E3 (6) |
| LH | `\RF::LH_PHASE` | 1 | (81920,); float64 | `'Deg'` | 0.004; t0 2.5145; t1 2.842176 | E2 (13), E3 (6) |
| LH | `\RF::LH_INTOD3` | 0 | — | `—` | — | E2 (13), E3 (6), E5 (1) |
| LH | `lhpower` | 0 | — | `—` | — | E1 (20) |
| LH | `lh_power` | 0 | — | `—` | — | E1 (20) |
| LH | `lhcd` | 0 | — | `—` | — | E1 (20) |
| Pellet | `\PELLET::LGIHI_T` | 0 | — | `—` | — | E3 (20) |
| Pellet | `\PELLET::LGIDR_T` | 0 | — | `—` | — | E3 (20) |
| Pellet | `\PELLET::LGIAB_TMAX` | 0 | — | `—` | — | E3 (20) |
| Pellet | `\PELLET::LGIHI_MASS` | 0 | — | `—` | — | E3 (20) |
| Pellet | `pellet` | 0 | — | `—` | — | E1 (20) |
| Pellet | `pellet1` | 0 | — | `—` | — | E1 (20) |
| Pellet | `pifire` | 0 | — | `—` | — | E1 (20) |
| Pellet | `pifire1` | 0 | — | `—` | — | E1 (20) |
| Pellet | `pigas` | 0 | — | `—` | — | E1 (20) |
| EFC | `c19` | 20 | (50176,), (480256,); float64 | `'a'` | 0.05…0.5; t0 -9.304602…-3.959906; t1 15.7829…20.05284 | none |
| EFC | `c79` | 20 | (50176,), (480256,); float64 | `'a'` | 0.05…0.5; t0 -9.304602…-3.959906; t1 15.7829…20.05284 | none |
| EFC | `c139` | 20 | (50176,), (480256,); float64 | `'a'` | 0.05…0.5; t0 -9.304602…-3.959906; t1 15.7829…20.05284 | none |
| EFC | `c199` | 20 | (50176,), (480256,); float64 | `'a'` | 0.05…0.5; t0 -9.304602…-3.959906; t1 15.7829…20.05284 | none |
| EFC | `c259` | 20 | (50176,), (480256,); float64 | `'a'` | 0.05…0.5; t0 -9.304602…-3.959906; t1 15.7829…20.05284 | none |
| EFC | `c319` | 20 | (50176,), (480256,); float64 | `'a'` | 0.05…0.5; t0 -9.304602…-3.959906; t1 15.7829…20.05284 | none |
| EFC | `iu30` | 20 | (51200,), (480256,); float64 | `'a'` | 0.05…0.5; t0 -10…-3.959906; t1 15.5995…20.05284 | none |
| EFC | `il30` | 20 | (51200,), (480256,); float64 | `'a'` | 0.05…0.5; t0 -10…-3.959906; t1 15.5995…20.05284 | none |
| EFC | `\OPERATIONS::CN1IAMP` | 20 | (7000,), (7500,), (8000,); float32 | `'Amps'` | 1; t0 0; t1 6.999…7.999 | none |
| EFC | `\OPERATIONS::IUN1IAMP` | 20 | (7000,), (7500,), (8000,); float32 | `'Amps'` | 1; t0 0; t1 6.999…7.999 | none |
| EFC | `\OPERATIONS::ILN1IAMP` | 20 | (7000,), (7500,), (8000,); float32 | `'Amps'` | 1; t0 0; t1 6.999…7.999 | none |
| Current Saturation | `ecoil` | 20 | (50176,), (480256,); float64 | `'a'` | 0.05…0.5; t0 -9.304602…-3.959906; t1 15.7829…20.05284 | none |
| Current Saturation | `epso` | 0 | — | `—` | — | E1 (20) |
| Current Saturation | `iecoil` | 0 | — | `—` | — | E1 (20) |
| Current Saturation | `e1a` | 0 | — | `—` | — | E1 (20) |
| Current Saturation | `e1b` | 0 | — | `—` | — | E1 (20) |
| Current Saturation | `pcecoil` | 0 | — | `—` | — | E1 (20) |
| Current Saturation | `pceilim` | 0 | — | `—` | — | E1 (20) |
| Current Saturation | `f1a` | 20 | (50176,), (480256,); float64 | `'a'` | 0.05…0.5; t0 -9.304602…-3.959906; t1 15.7829…20.05284 | none |
| Current Saturation | `f7b` | 20 | (50176,), (480256,); float64 | `'a'` | 0.05…0.5; t0 -9.304602…-3.959906; t1 15.7829…20.05284 | none |
| Current Saturation | `f9b` | 20 | (50176,), (480256,); float64 | `'a'` | 0.05…0.5; t0 -9.304602…-3.959906; t1 15.7829…20.05284 | none |

### B. Candidates discovered in the single-shot PTDATA index metadata

| Row | Point (tree-qualified MDS; otherwise PTDATA) | Returned /20 | Actual data shape(s); dtype | Raw data units | Median Δt (ms); t0 and t1 ranges (s) | Fetch errors (count) |
| --- | --- | ---: | --- | --- | --- | --- |
| Current Saturation | `ECOILFWDCL` | 4 | (88000,), (118000,); float64 | `'v'` | 0.25…0.25; t0 -19…-10; t1 10.49975…11.99975 | E1 (16) |
| Current Saturation | `ECOILREVCL` | 4 | (88000,), (118000,); float64 | `'v'` | 0.25…0.25; t0 -19…-10; t1 10.49975…11.99975 | E1 (16) |
| Current Saturation | `IPTECOILMX` | 20 | (40203,), (40204,), (40205,), (40206,), (40257,), (40435,), (42204,), (42205,), (44203,), (44204,), (46930,); float64 | `'vo'` | 0.25; t0 -8.99975; t1 6.99975…7.99975 | none |
| Current Saturation | `IPTECOILMN` | 20 | (40203,), (40204,), (40205,), (40206,), (40257,), (40435,), (42204,), (42205,), (44203,), (44204,), (46930,); float64 | `'vo'` | 0.25; t0 -8.99975; t1 6.99975…7.99975 | none |
| Current Saturation | `IPTECOIL` | 20 | (40203,), (40204,), (40205,), (40206,), (40257,), (40435,), (42204,), (42205,), (44203,), (44204,), (46930,); float64 | `'vo'` | 0.25; t0 -8.99975; t1 6.99975…7.99975 | none |
| Current Saturation | `IPXOECOIL` | 20 | (7,), (8,); float64 | `'vo'` | 1000; t0 0.4; t1 6.4…7.4 | none |
| Pellet | `ONIPELON` | 20 | (40203,), (40204,), (40205,), (40206,), (40257,), (40435,), (42204,), (42205,), (44203,), (44204,), (46930,); float64 | `'vo'` | 0.25; t0 -8.99975; t1 6.99975…7.99975 | none |
| Pellet | `IPELI` | 20 | (40203,), (40204,), (40205,), (40206,), (40257,), (40435,), (42204,), (42205,), (44203,), (44204,), (46930,); float64 | `'vo'` | 0.25; t0 -8.99975; t1 6.99975…7.99975 | none |
| Pellet | `ONSPELSQ` | 20 | (40203,), (40204,), (40205,), (40206,), (40257,), (40435,), (42204,), (42205,), (44203,), (44204,), (46930,); float64 | `'vo'` | 0.25; t0 -8.99975; t1 6.99975…7.99975 | none |
| Pellet | `PDTPELSIZE` | 4 | (142495,), (142499,), (142501,), (142554,); float64 | `'vo'` | 0.05; t0 -8.9999…-5; t1 6.99995 | E1 (16) |
| Pellet | `PEIPELIN1` | 11 | (19599,), (40203,), (40204,), (40205,), (40257,), (42205,), (44203,), (44204,), (46930,); float64 | `'vo'` | 0.25; t0 -8.99975…3.10025; t1 6.99975…7.99975 | E1 (9) |
| Pellet | `PESNPELL` | 1 | (40203,); float64 | `'vo'` | 0.25; t0 -8.99975; t1 6.99975 | E1 (19) |
| Helicon | `D1CHELICON` | 15 | (40203,), (40204,), (40205,), (40206,), (40257,), (40435,), (42204,), (42205,), (44203,), (44204,), (46930,); float64 | `'raw'` | 0.25; t0 -8.99975; t1 6.99975…7.99975 | E1 (5) |
| Helicon | `DOCHELICON` | 15 | (142495,), (142497,), (142498,), (142499,), (142500,), (142501,), (142554,), (144315,), (152499,), (152500,), (153962,), (162497,), (162499,); float64 | `'raw'` | 0.05; t0 -8.9999…-5; t1 6.99995…7.99995 | E1 (5) |
| Helicon | `PCHELISTS` | 20 | (142495,), (142497,), (142498,), (142499,), (142500,), (142501,), (142554,), (144315,), (152499,), (152500,), (153962,), (162496,), (162497,), (162499,); float64 | `'raw'` | 0.05; t0 -8.9999…-5; t1 6.99995…7.99995 | none |


## Reproducibility and verification

Retained local evidence (not committed data):

- `/tmp/LCprobe/sample.json`, `candidates.json`, `candidates-index.json`:
  exact sample and candidate manifests, including discovery provenance.
- `/tmp/LCprobe/probe.jsonl`, `probe-index.jsonl`, `point-shot-table.csv`:
  all 1,240 records, their raw units/shape/clock summaries, and exact errors.
- `/tmp/LCprobe/discover.py`, `discover.log`, `discovery-204925.json`,
  `details.py`, `details.json`, `ptdata-sources.json`, `read_index.py`,
  `index-{200009,203505,204925}.json`, `record-metadata.jsonl`, and
  `header-metadata.jsonl`: read-only namespace and record/header evidence.
- `/tmp/LCprobe_actuator_census_20260913.py`, `_results.json`,
  `_shot_summary.csv`, `_coil_summary.csv`, `_run.log`: full census script,
  aggregate/per-shot/per-coil results, and completion log; the companion
  `/tmp/LCprobe_actuator_census_focus_20260913.py` and
  `/tmp/LCprobe_actuator_census_20260913_focused.json` derive active/year/bin
  views from those retained summaries without rereading the corpus.
- `/tmp/LCprobe_feature_actuator_scope_20260913.{py,json,csv,log}`:
  read-only feature-store key/metadata audit and stored gas peak comparison.
- `/tmp/LCprobe/review.md`: independent spec and analysis/code review.
- `/tmp/LCprobe/SHA256SUMS`: integrity manifest for the retained evidence.

The `/tmp` evidence is session-local and may be removed by the host. The
committed tables, sample, error dictionary, interpretation, and definitions
above retain the findings independently of those scratch files. To repeat the
probe, recreate the two candidate manifests from tables A/B (MDS points carry
their displayed tree; all others are PTDATA) and the 20-shot sample above;
run the supplied script with a fresh output path. The census command is in
its section above.

Verification performed:

- Both probe processes exited 0; checked 62 candidates ×20 distinct selected
  shots =1,240 unique records, with 548 returns and 692 explicit fetch errors.
  Every returned waveform has matching data/time lengths, finite increasing
  times, and retained unit metadata. A scalar's waveform-fetch failure is
  separately explained rather than counted as a missing scalar.
- Census completed 500/500 unique manifest shots; output shot identities and
  aggregate counts are checked against the YAML. An independent review
  reconciled all 62 probe rows and all five census bin-count vectors against
  the retained machine outputs. No labels or events were run.
- `python -m ruff check --no-cache scripts/labelmaker/fdp_probe.py` and
  `python -m ruff format --check scripts/labelmaker/fdp_probe.py` pass in the
  shared labelmaker environment. The script compiles under the project's
  Python and was exercised by the complete live probe. The system `python3`
  is too old for the repository's `annotations` future import, so the compile
  check uses the project environment, just as the live execution does.
- No library code was added; per the brief, no new tests were required. No
  full model/label suite or GPU/real-data test suite was run for this
  documentation and probe-script task.
