# Physics-informed sawtooth labels and trained pickers

The labels combine Gude's multichannel redistribution test with the central ECE
crash criteria described by Muscatello, Heidbrink and Van Zeeland. The OuYang
HL-3 CNN/LSTM remains the external model baseline. This additive method does not
refresh production event stores. Independent expert spans, detector agreement
and synthetic checks are reported separately; model agreement with algorithmic
labels does not establish physical ground truth.

## Four-state detector

`labeler.sawtooth.physics.detect` applies a Gaussian derivative, locally adaptive
POSR significance, multichannel coincidence and a contiguous loss/gain profile,
following Gude et al., PPCF 59 (2017) 095009, DOI 10.1088/1361-6587/aa79d8. The
profile must lose temperature in the core and gain it outside the inversion
boundary. With verified ECE/equilibrium calibration, the signs must agree with
membership inside/outside the axis-connected q=1 surface; proximity alone is
insufficient. Inverted core crashes are excluded. Without that calibration,
the established ECE core group and ordered channel boundary are proxies.
DIII-D's array can cover both sides of the axis, but verified spatial pairing
is unavailable here. Channel order is not a calibrated radius, and Gude's
two-sided asymmetry test is not claimed.

Muscatello et al., PPCF 54 (2012) 025006,
DOI 10.1088/0741-3335/54/2/025006, defines amplitude as
`abs(Te_pre - Te_post) / Te_pre` on the central ECE channel. The detector requires
a minimum signed central loss and retains its relative amplitude in crash/train
attributes. The hottest valid channel in the specified core group is an
uncalibrated central proxy; it is not asserted to be the magnetic-axis channel.
Train membership additionally requires plausible adjacent periods, period
regularity and a stable inversion boundary. Slowly drifting boundaries can
form successive local trains, while abrupt shifts split them. Isolated profiles
are uncertain candidates rather than positive crash targets.

The four states are `present`, `absent`, `uncertain` and `unassessed`:

- `present`: an observable train passes the central drop, profile, period and
  inversion-stability guards without a known equilibrium conflict.
- `absent`: core ECE is observable and no accepted or uncertain train covers
  the bin.
- `uncertain`: an observable profile or train has unresolved evidence, including
  a q/ECE conflict or insufficient train support.
- `unassessed`: core ECE is missing or invalid, temperature is below the frozen
  floor, or the conservative density-cutoff proxy identifies invalid support.

Magnetics-only EFIT01 does not constrain central q reliably. Its q-min is a soft
conflict flag that can make a span uncertain; it never rejects a crash and never
turns a visible train into absence. A locally available MSE-constrained equilibrium
is preferred and its source is recorded. Missing equilibrium is unknown rather
than evidence of absence. Calibrated physical inversion radius and q=1 radius
stay null when verified channel geometry is unavailable.

D-alpha spikes reject an edge-only loss using independent core-motion evidence,
before the generic core gate; a real core crash coincident with an ELM remains
eligible. When locally cached signals exist, a neutron-rate drop or a Mirnov
burst corroborates Muscatello's crash signature. If usable auxiliary data exist
but neither signature corroborates the crash, its train becomes uncertain.
Missing auxiliary diagnostics stay unknown and do not supply negative evidence.
No precursor mode-number identification is claimed from an uncalibrated Mirnov
mean. SXR is not used as corroboration: selecting bright channels cannot establish
a spatial core-loss/edge-gain test. Pellet timing is unavailable.

## Observability and signal preparation

The corpus and existing label stores are read-only. ECE is filtered at its native
rate in bounded-memory overlapping chunks before decimation to the detector
rate. Invalid samples invalidate the associated FIR support. Nonuniform clocks
are rejected. The antialias comparison record quantifies candidates and timing
under the retired stride-first preparation and the corrected native-rate filter
on train shots only.

The per-bin `observable` mask requires valid core ECE above the temperature
floor. A conservative cutoff proxy uses the 90th percentile of local core Thomson
density and toroidal field with the second-harmonic X-mode cutoff relation:
`n_cut = 0.9 × 2 × (27.992e9 × abs(Bt) / 8.98)^2` in m⁻³. With density but no local
Bt, a frozen high-density guard of `8e19 m⁻³` applies. Without verified per-channel
ECE frequencies, this is a conservative exclusion rather than a calibrated
channel-specific cutoff calculation. Missing density makes cutoff status unknown;
the finite-ECE and low-temperature guards still apply. Reader choices and values
come from `outputs/labeler/sawtooth/fix/freeze.json` → `reader_guards`. Trains
split at all observability gaps. The `assessed` mask also excludes uncertainty.
Both masks are saved in signal caches, record spans, model targets and scoring.
Unobservable bins and predictions never enter assessed denominators.

Unavailable auxiliary inputs start as NaN. Each training fold estimates means
and scales from valid training support, then imputes missing values to zero in
normalized units. A missing physical signal is never initialized to physical zero.
Interpolated Ip retains the bandwidth of its cached source.

Cohort windows come from the fixed catalog. Outside-cohort population windows
use local Ip extent when available, otherwise ECE extent; these fallback windows
retain explicit unassessed and uncertain states. Exactly the same frozen rule
and q policy apply to cohort and population.

## Freeze and provenance

Every detector guard is frozen using nonexpert fixed-`train` shots only. Exact
shot lists, candidate summaries, adopted thresholds and selection rationale are
written by the calibration stage under
`outputs/labeler/sawtooth/fix/`. Synthetic noise calibration is separate and uses
no corpus shot.

Every guard was checked and frozen on these exact train shots:
185945, 186640, 189008, 190992, 192761, 193024, 193148, 195956, 195989, 196375,
199908, 201707, 203629, 203876, 203959 and 204225. The source for the list is
`outputs/labeler/sawtooth/fix/freeze.json` → `development_shots`; the per-guard
lists are `guards.<name>.shots`, and reader choices use `reader_guards.shots`.
Other than the central amplitude choice, values are literature/review priors
checked on this train sample rather than estimates of population-specific
optima.

| Guard | Frozen value | JSON field under `freeze.json` |
|---|---:|---|
| Gaussian derivative sigma | 0.25 ms | `guards.sigma_ms.value` |
| POSR frame | 10.3 ms | `guards.frame_ms.value` |
| POSR threshold | 6 | `guards.posr_threshold.value` |
| Coincidence window | 0.5 ms | `guards.coincidence_ms.value` |
| Minimum agreeing channels | 2 | `guards.minimum_channels.value` |
| Relative profile significance | 0.02 | `guards.significance.value` |
| Maximum net/absolute redistribution | 0.9 | `guards.maximum_net.value` |
| Minimum contiguous loss/gain block | 2 channels | `guards.minimum_block.value` |
| Neighboring block reach | 6 channels | `guards.pulse_reach.value` |
| Central relative loss | 0.05 | `guards.central_relative_drop.value` |
| Core temperature floor | 0.5 keV | `guards.te_floor_kev.value` |
| Adjacent period | 20–250 ms | `guards.minimum_period_ms.value`; `guards.maximum_period_ms.value` |
| Adjacent-period ratio | ≤2.5 | `guards.period_ratio.value` |
| Minimum train | 3 crashes | `guards.minimum_train.value` |
| Inversion-channel spread | ≤2 channels | `guards.inversion_spread_channels.value` |
| q-min conflict margin above one | 0.05 | `guards.qmin_margin.value` |
| Calibrated radius tolerance | 0.15 normalized rho | `guards.radius_tolerance.value` |
| Minimum plasma current | 0.3 MA | `guards.minimum_ip_ma.value` |
| Neutron-rate relative loss | 0.02 | `guards.neutron_relative_drop.value` |
| Mirnov burst threshold | 6 robust noise scales | `guards.mirnov_burst_z.value` |
| D-alpha burst threshold | 6 robust noise scales | `guards.dalpha_burst_z.value` |

The central-loss minimum was selected from 157 train candidate amplitudes using
`floor(100 × clip(q10, 0.05, 0.15)) / 100`, yielding 0.05. Source:
`outputs/labeler/sawtooth/fix/freeze.json` → `amplitude_selection`. This inclusive
minimum is not the typical amplitude of Muscatello's clean L-mode reference
discharges, and the train sample does not independently validate physical
central localization.

The earlier report's claim that the retired pilot contained only training shots
was incorrect: its sorted pilot also contained validation, test and expert shots.
That pilot is excluded from the corrected freeze. Its outputs are not reused.
Source: `outputs/labeler/sawtooth/fix/freeze.json` → `prior_pilot_correction`.
Expert and test results do not change any corrected threshold.

Large cohort/population label CSVs are untracked under
`data/events/sawtooth_oscillation/extend_saw_physics/` and copied under
`$LABELER_ROOT/round4/saw/`. The seven-column research interchange contains
`shot, category, t_start, t_end, crowd, confidence, attrs`; it is distinct from
the production interval schema. Times are milliseconds. Zero-duration crash
points use `crowd=False`, and half-open state spans use `crowd=True`. State spans
come only from canonical four-state support; raw diagnostic train metadata is
not exported as an overlapping truth span. Point states follow canonical support
when uncertainty or unassessed support overrides the candidate state. Compact
attributes retain `candidate_state` and `uncertainty_reasons` as well as the
exported `state`. Small JSON summaries are committed; waveforms, full
per-shot records, checkpoints, predictions and figures stay under the external
output root. Resume is valid only for the same frozen rule and source settings.

## Models and input parity

`saw-hl3` reimplements OuYang et al., PPCF 67 (2025) 105004,
DOI 10.1088/1361-6587/ae0786, from the supplied paper digest. It receives ECE
core-group and outer-group means, a Mirnov mean and Ip. The paper's core/edge SXR
pair is replaced by ECE proxies because verified SXR spatial pairing is missing.
The network combines residual convolutions, channel attention, a bidirectional
LSTM and sequence attention with three-regime window classification. Period
classes use a training-only median boundary rather than the HL-3 period boundary.
Its presence output gates a separately selected derivative picker for crash
timing; the published classifier does not itself predict point times.

`saw-ours` is a PhaseNet-style one-dimensional U-Net inspired by Zhu and Beroza,
GJI 216 (2019), arXiv:1803.03211. It receives all ECE channels and predicts a
crash-probability trace plus train presence. Crash-target centers must first
have assessed support; truncated Gaussian tails and the presence loss are also
masked to exclude uncertain/unassessed bins. This prevents an unknown crash
center from supplying a positive target in an adjoining assessed bin. The wider
context and second output are DIII-D adaptations.

The models do not have input parity: the baseline's compressed ECE/auxiliary
inputs and the picker's full ECE array supply different information. Results
therefore compare adapted systems rather than architecture alone. Published
HL-3 window accuracies and DIII-D crash F1 are different tasks and populations.

Both models run in the CUDA environment with early stopping on inner-fold
selection loss, without an epoch cap. Each outer training fold is divided into
fitting shots and independent selection shots, all from the fixed train split.
Fitting shots estimate normalization and the class boundary; selection shots
choose checkpoints and operating thresholds. Probability thresholds span
0.05–0.95, and the baseline
derivative threshold spans z=2–10. The fold JSONs retain complete grids, losses,
selected epochs, thresholds, shot lists, device and stopping reasons.

## Evaluation and figures

Out-of-fold predictions are scored on whole train-cohort shots never used to
fit or select that fold. Point matching is one-to-one at the recorded tolerance;
presence AUROC, AUPRC, precision, recall and F1 use identical assessed support.
Pooled out-of-fold estimates retain shot-bootstrap confidence intervals.
Fixed validation and blind test shots do not tune model thresholds or checkpoints.

The old `heuristics.sawtooth_events` rule is run read-only from the local corpus
on the full cohort and every expert shot. Old/new agreement states the count of
shots where both ran, and each rule is scored against known expert spans per shot
on the same known observable support. Primary label metrics score definite-present
retrieval; uncertain positives count as unresolved misses for recall/F1, without
exporting them as absent. Conditional assessed metrics and uncertainty coverage
remain secondary. Model expert scoring uses all known observable bins:
independent expert annotations resolve algorithmic uncertainty. Experts provide
spans rather than crash times,
so expert crash-timing precision/recall is unavailable. No bootstrap is reported
for the small expert set. With no positive predictions, precision is undefined.
Muscatello reference-shot availability and any physical period/amplitude/timing
check are recorded explicitly; missing files are not fetched.

The gallery retains the original random nonexpert train-shot sample. Every shot
has a whole-shot ECE, inversion-channel and EFIT01 q-min overview plus a short
crash-window trace and pre/post ECE channel profile. Present support is green,
uncertainty orange, and unassessed support gray. Crash markers show candidate
state; shading shows canonical bin state. Point crashes and sustained-presence
spans have different temporal support, including half-open train endpoints.
Every marker and shaded state
has a legend; legends and annotations sit outside the traces. Unavailable
geometry is stated outside the data area, and an empty inversion axis shows the
full channel range. Vector PDFs and 150-dpi PNGs are saved under
`$LABELER_ROOT/round4/saw/fix/gallery/`; `gallery.json` records figure paths,
window choices, states and PNG inspection.

## Reproduction

From this worktree, set:

```bash
export TMPDIR=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/claude-89242e53/r4/tmp/saw
export LABELER_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker
export LABELER_LABEL_TABLES=/scratch/gpfs/nc1514/FusionAIHub/data/events
export LABELER_NO_FETCH=1 PYTHONPATH=$PWD/src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

Use this prefix for calibration, local reading, labels, validation and figures:

```bash
pixi run --frozen --no-install \
  --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker python
```

The stage arguments are recorded by the calibration and evaluation scripts.
Use the corrected work directory for all label, validation and gallery stages:

```text
scripts/labeler/sawtooth_freeze.py freeze --work $LABELER_ROOT/round4/saw/fix
scripts/labeler/sawtooth_physics.py labels --work $LABELER_ROOT/round4/saw/fix --workers 8
scripts/labeler/sawtooth_physics.py labels --work $LABELER_ROOT/round4/saw/fix --population --workers 8
scripts/labeler/sawtooth_freeze.py impact --work $LABELER_ROOT/round4/saw/fix
scripts/labeler/sawtooth_fix_validation.py old --work $LABELER_ROOT/round4/saw/fix
scripts/labeler/sawtooth_fix_validation.py validate --work $LABELER_ROOT/round4/saw/fix
scripts/labeler/sawtooth_gallery.py --work $LABELER_ROOT/round4/saw/fix
scripts/labeler/sawtooth_results.py --work $LABELER_ROOT/round4/saw/fix
```

GPU model stages use the existing CUDA environment, not the CPU-only pixi torch:

```bash
CUDA_VISIBLE_DEVICES=1 \
  /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/envs/phase3/bin/python \
  scripts/labeler/sawtooth_benchmark.py train --work "$LABELER_ROOT/round4/saw/fix"
```

Use the same CUDA invocation for `evaluate`. No package installation, network
lookup or production-store write is needed. The results document is rendered
from JSON by `sawtooth_results.py`; the implementation report is written
separately and is not hard-coded into the renderer.
