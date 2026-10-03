# Unvalidated sawtooth research labels and trained pickers

The current method supplies four-state research labels and two adapted model
benchmarks. Neither crash times nor positive/negative spans have independent
blind expert validation. No model is recommended as latest or stable. Current
small records are under `outputs/labeler/sawtooth/fix2/`; large artifacts are under
`$LABELER_ROOT/round4/saw/fix2/`. The corpus and production stores are read-only.

## Detector and states

The detector combines Gude's Gaussian derivative, POSR significance,
multichannel coincidence and redistribution profile (PPCF 59 (2017) 095009) with
Muscatello's central relative-temperature loss (PPCF 54 (2012) 025006). A
contiguous loss block must have an adjacent gain block. The nearest qualifying
gain supplies the inversion boundary; a stronger remote gain cannot move it.
Period and inversion stability define trains. This is an unvalidated DIII-D
adaptation; calibrated two-sided symmetry and mode numbers are not established.

- `present`: observable support in a profile-passing stable train without a
  known conflict at the contributing crash.
- `absent`: a complete observable context extending ±1.5 maximum periods,
  no profile-passing candidate nearby, and a recorded core-ECE relaxation test
  with no periodic pattern or unresolved significant negative edge. Measured
  core edge noise must resolve the significance floor. This conservative
  negative policy does not prove absence of weak or atypical sawtooth.
- `uncertain`: observable support without definite train evidence or tested
  absence, including candidate-phase gaps, isolated/irregular core relaxations,
  excessive noise, incomplete context and q/ECE conflicts.
- `unassessed`: invalid/missing core ECE, low temperature or the conservative
  density-cutoff exclusion. Trains never cross these gaps.

`state_spans` defaults unresolved observable support to uncertainty. Every
profile-passing candidate protects surrounding support before the central-drop,
current, core-membership and equilibrium gates. Significant negative core edges
also protect support outside the positive train's period range. Each shot's
`absence_diagnostics` records the tests, edge times, period bounds, noise windows
and reason counts. Stable neighboring crashes retain continuous phase support;
other candidate gaps remain uncertain. State seconds and absent-in-hole seconds
before/after are recorded by `fix2/state_transition_audit.json`.

EFIT01 q-min is a soft per-crash conflict flag; it cannot reject visible crashes
or teach absence, and reasons are not unioned across a train. Cached
MSE-constrained equilibrium is preferred. Missing equilibrium supplies no
negative evidence. Cohort and population use the same frozen rule with different
equilibrium availability: the previous run had EFIT01 on 452/500 versus
1,213/13,650 shots. Current counts are in `cohort_labels.json` and
`population_labels.json` under `q_sources` and `processed_shots`.

## Channels and geometry

A coherent hot profile supplies a per-shot reference temperature. Channels
above 1.5 times that reference are masked, with a further per-crash core-relative
screen. Channels 40–47 are excluded from fallback core selection because their
spatial interpretation is unverified and reviewed examples show harmonic
overlap; their invalidity is determined by the temperature screen. The fallback
selects the hottest physical channel and neighbors, plus an adjacent outer
group. Exact masks/groups appear in `core_geometry` and the signal cache.
Detector, baseline and gallery now use these groups consistently.

Actual RF metadata require channel identities/order and a documented field
reference. No channel frequencies were found in the corpus, feature stores or
labelmaker raw store. The usable source is the same-shot archive
`/scratch/gpfs/nc1514/omnimode/data/raw/<shot>.h5:/ecegeom/FREQ`, originally read
from ELECTRONS `\ECE::TOP.SETUP.FREQ`. Stored units are blank; the archived
reader's documented contract is GHz. Corpus row `i` joins archived
`ECEVS{i+1:02d}`. The inventory records this contract and channel-order checks;
a universal channel-frequency table is not substituted for missing shots.
Nominal second-harmonic positions obey
`R_m = 2 × 27.992e9 × abs(Bt_T) × Rref_m / f_Hz`; a cached equilibrium
`F=R Bphi` supplies the field product directly where available. The EFIT axis
identifies the nearest core channels. These positions omit relativistic and
optical-depth corrections and remain unvalidated. Exact on-disk sources and
frequency/field/axis support are in per-shot `radius_geometry`; the local search
inventory is in `fix2/geometry_metadata_audit.json`. Archived grids vary by shot,
so unsupported same-shot frequencies are not silently borrowed. Such shots use
the hottest-channel proxy and retain null physical radii.

Inversion R and EFIT q=1 R are compared only where both can be derived on the
same equilibrium branch. Normalized flux radius is never relabeled as major
radius. Paired coordinates/counts appear in label summaries; missing positions
stay null. Nominal geometry does not independently validate the r_inv/q=1 relation.

## Auxiliary evidence and signal preparation

Neutron and Mirnov provide positive per-crash corroboration only. Neutron drops
require known NBI-on support (100 kW floor), the frozen relative drop and
`k=3` times the measured window noise. Noise is the quadrature of pre/post native
window MAD scales, without a square-root sample-count discount for correlated
measurements. Power, drop, noise, k and evidence status are recorded per crash.
Missing, undersampled, noisy or noncorroborating auxiliary data supply no evidence
and never cause uncertainty. `auxiliary_not_corroborated` is removed. Mirnov
requires a robust burst; no mode-number claim is made. D-alpha can reject an
edge-only event using core-motion evidence. SXR spatial pairing and pellet timing
remain unavailable.

Bounded native-rate FIR filtering precedes decimation to 10 kHz. Missing filter
support stays missing; nonuniform clocks are rejected. Local Thomson density/Bt
supply a conservative cutoff exclusion; missing density is unknown. Exact
guards, the original train-only development shot list, prior origins and
per-shot checks appear in `fix2/freeze.json`. The central-amplitude minimum is
a 5% prior floor clipped against train quantiles, not a data-derived optimum.
No test or expert shot tunes a detector/model threshold.
The 20 ms positive-train period floor is a prior and can leave faster trains
uncertain; per-channel negative evidence also protects those faster relaxations.

## Models and evaluation

`saw-hl3` reimplements OuYang's CNN/attention/bidirectional-LSTM classifier
(PPCF 67 (2025) 105004) from the supplied digest. Per-shot core/outer ECE means
replace its SXR pair; Mirnov mean and Ip supply the other inputs. A separately
selected derivative picker supplies point times. `saw-ours` is a PhaseNet-style
1-D U-Net with crash and presence heads and the screened multichannel ECE input.
Different input information prevents an architecture-only comparison.

Both models use observability-based input masking in normalization, training
and inference. Observable uncertain measurements stay in the input; assessment
masks affect loss and scoring only. Both receive 50% random / 50% crash-centered
sampling, balanced between positive period classes. Fitting shots determine
normalization and the period boundary. Fit/selection class diagnostics are
recorded. Missing values are imputed to zero after normalization.

Three whole-shot outer folds use fixed train; disjoint inner train shots
select checkpoints and thresholds. Fixed val, expert and test shots never fit
or select a model. CUDA training has no epoch cap and stops on inner-selection
loss patience. Fold JSONs record losses, best/completed epochs, device,
stopping conditions and the derivative-z grid, which extends beyond 10. The
results show crash F1 at ±1/±2 ms, assessed-bin metrics, and 1,000-replicate 95%
shot-bootstrap intervals. HL-3 confusion, class recalls, window accuracy with
CI, macro-F1 and majority baseline appear beside OuYang's published window
accuracies. All DIII-D metrics measure agreement with unvalidated targets.

The span annotations for three expert shots were drawn while viewing old
`ece_sawtooth` suggestions
(`review/source.json`) and are anchored. Shot 190637's span may contain
edge-originated relaxations. Per-shot span comparisons are exploratory; they
cannot validate independent crash timing or separate the rules. No small-set
CI or blind crash-time accuracy is reported. The queue has 15 nonexpert val
shots stratified by heating and predicted-period regime. Physical H/L regimes
remain unknown: Jalal Butt's cached table covers no fixed-val shot here. The
README specifies blind marking, observable negative spans and ambiguity.

## Exports, figure and reproduction

Untracked seven-column research CSVs contain millisecond crash points and
half-open canonical state spans. Present spans retain contributing train IDs,
period, inversion channel and nullable R/rho; per-contributor metadata preserve
mixed trains. Crash states follow canonical support. `fix2/paper_example.json`
records a 300 ms nonexpert train/val example, 3.25-inch width, fonts ≥7 pt,
vector PDF, 150-dpi PNG and visual inspection. Markers are candidate crashes.

Set the prescribed TMPDIR, LABELER_ROOT, LABELER_LABEL_TABLES, LABELER_NO_FETCH=1,
PYTHONPATH and single-thread BLAS environment from the implementer rules. Run
Python reader/label/figure stages through frozen/no-install pixi, with the main
checkout's manifest and labelmaker environment. Scripts live in `scripts/labeler`:

```text
sawtooth_freeze.py freeze
sawtooth_physics.py labels --workers 8
sawtooth_population.sbatch (four disjoint shards)
sawtooth_physics.py labels --population --records-only
sawtooth_fix_validation.py validate
sawtooth_fix2_artifacts.py audit
sawtooth_fix2_artifacts.py queue
sawtooth_fix2_artifacts.py figure
sawtooth_results.py
```

Default WORK/OUTPUT paths use `fix2`. Validation reuses immutable prior native
old-rule records because that rule/reader is unchanged: copy `fix/old_rule` into
`fix2/old_rule`, or run `sawtooth_fix_validation.py old` to regenerate.
GPU `sawtooth_benchmark.py train`, `evaluate` and `predict --prediction-shots
<queue IDs>` use the existing phase3 CUDA Python with CUDA_VISIBLE_DEVICES=1.
No package installation/fetch is needed. Run the prescribed tmp sweep after jobs.

## History appendix

The first implementation used a mixed pilot and fixed-val model selection.
The first fix removed leakage but retained gap-as-absence, train-wide auxiliary
uncertainty, fixed ECE proxies, unmatched sampling and label-dependent training
inputs. Its records remain under `fix/`; current results supersede them. Earlier
descriptions of these expert spans as independent were incorrect. Physical
ground truth and blind crash-time validation remain pending.
