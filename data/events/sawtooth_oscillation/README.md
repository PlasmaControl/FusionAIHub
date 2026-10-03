# Sawtooth Oscillation

## Description
Sawtooth are periodic relaxations of the plasma core when the central safety factor
falls below one: the m/n = 1/1 internal kink grows, magnetic reconnection
(Kadomtsev) flattens the core temperature and density inside the mixing radius, and
the profile then re-peaks over a few tens of ms - a slow ramp and a fast crash,
hence the name. The inversion radius r_inv (where the crash changes sign from a
drop inside to a rise outside) can be compared with the EFIT q = 1 surface;
that relation is unvalidated in the current catalog. Sawtooth crashes can seed
NTMs, expel fast ions and trigger ELMs, and
a crash-triggered ELM can hide the inversion in edge channels.

First observed on the ST tokamak with soft-X-ray diodes (von Goeler, Stodiek and
Sauthoff 1974).

Typically found via ECE radiometer channels (the 48-channel array here), soft
X-rays and core Thomson scattering as simultaneous drops in the core channels and
rises in the outer ones; the 1/1 precursor is visible on magnetics at 2-20 kHz.

## Data Provenance
### Dataset 1

**Dataset File(s)**:

**Author**:

**Description**: Rule-based; no curated table, so `raw/` is empty. The inventory notes OMFIT's ECE-based sawtooth tool and Hiro's detector as possible references.

**Publications**:

## Models
**stable**: none

**latest**: none (all current research candidates are unvalidated)

**all**:
- ece_sawtooth | 2026_09_12 (rule; omnimode inversion test, envelope-once port)
- saw_physics | 2026_10_03 (Gude multichannel inversion plus Muscatello central-drop/train criteria; q conflicts abstain)
- saw-hl3 | 2026_10_03 (OuYang HL-3 CNN + bidirectional LSTM external baseline; GPU fit, DIII-D input/timing adaptations)
- saw-ours | 2026_10_03 (PhaseNet-style multichannel ECE crash picker and train-presence head; GPU fit)

## Inputs
**ece_sawtooth**:
- `ECE` (48 ch)
- `SXR` (first lit fan of `SX90RM1F`, `SX90RP1F`, `SX90RM1S`, `SX90RP1S`)
- `Ip` (plasma start)

**sawtooth_frames (round three)**:
- `ECE Te, ch 20-23`, `ECE Te, ch 24-27`, `ECE Te, ch 28-31`, `ECE Te, ch 32-35`
- `SXR` (optional)

**saw-hl3**:
- Per-shot physical-channel-screened core and adjacent outer ECE means
  (uncalibrated proxies where actual RF channel frequencies are unavailable)
- Mirnov 0–1 mean, cached Ip in MA (optional; missing values use training means)

**saw-ours**:
- All 48 ECE channels, 100 ms context at 10 kHz

## Method
`ece_sawtooth` (`labeler.events.heuristics.sawtooth_events`), a port of the
omnimode `mrms.ece` inversion test with the envelope computed ONCE per shot
(1.55 s/shot instead of ~3.5 min): a 1 ms envelope over the 48 ECE channels;
candidate bins where >= 2 channels lose > 2% of their level in one bin, at least
10 ms apart; a candidate is a crash when the step profile across it (2 ms gap,
8 ms span) is a structured inversion - a contiguous block of dropping channels
next to rising ones. Each crash is a point event with `confidence` = the fraction of
finite channels that took part, and `attrs["inversion_channel_lo"]`,
`attrs["inversion_channel_stop"]` bound the dropping block (end-exclusive).
Crash-by-crash agreement with the omnimode reference is checked by
[`sawtooth_reference_check.py`](../../../scripts/labeler/sawtooth_reference_check.py)
and its committed reference record.
On shot 198658 it finds 45 sawtooth with a median period of 76 ms.

`labeler.sawtooth.physics.detect` combines Gude-style Gaussian edge filtering,
multichannel coincidence and a contiguous core-loss/outer-gain inversion profile
with Muscatello's central relative-temperature drop and plausible, stable trains.
Native-rate filtering precedes decimation. Valid core ECE defines observability;
missing ECE, low temperature and detected cutoff yield `unassessed`, and trains
split at observability gaps. Isolated profiles and q/ECE conflicts yield
`uncertain`. Magnetics-only EFIT01 q-min never rejects a crash or teaches absence.
Where local neutron-rate and Mirnov data exist, their drop/burst flags provide
optional corroboration. No SXR corroboration is claimed without verified
core/edge spatial pairing. Calibrated ECE psi and trusted equilibrium profiles
support a direction-aware radius test; unavailable mapping produces null radii.
Absence requires complete candidate-free context and a noise-resolved core
relaxation test. Stable significant negative core edges protect their entire
phase without period bounds; ambiguous observable support remains uncertain.
The untracked exports in `extend_saw_physics/` hold four-state spans and crash
points. They are additive research labels and do not replace production labels.

Both learned models use shot-grouped cross-validation on the fixed training
cohort, with inner training-split selection shots for checkpoint and threshold
choice. CUDA training stops on inner-selection loss patience.
All expert-reviewed shots and the blind test split are excluded from training
and tuning. `saw-hl3` receives four adapted inputs while `saw-ours` receives the
full ECE array, so their comparison includes input information as well as
architecture. Expert tables contain spans, so true expert crash recall/precision
cannot be measured; results are reported per shot without small-sample CIs.
The span annotations for three expert shots were drawn while viewing the old
`ece_sawtooth`
suggestions, so they are anchored rather than independent validation. The
190637 span may include edge-originated relaxations. No blind crash-time truth
is available; every detector/model accuracy claim is unvalidated, and no model
is recommended as latest or stable.
See [method, adaptations and reproduction](../../../docs/labeler/sawtooth_physics.md)
and [JSON-backed result tables](../../../docs/labeler/sawtooth_results.md).

## Blind crash-time annotation queue

`review/crash_time_queue.csv` lists 15 held-out nonexpert validation shots
stratified by recorded heating and predicted period regimes, with predicted crash
support; no test shot is included. Jalal Butt's cached confinement table covers
no fixed-validation shots here, so physical H/L regimes remain unknown. For
blind marking, show the owner only shot
and time window in a shuffled order, with native-rate core and outer ECE and
available auxiliary traces; hide the queue's `why` column, all model/detector
picks, suggestions and state shading. Mark each confidently identified
core-loss/adjacent-outer-gain crash time, its timing tolerance, and the observable
span; mark edge-originated or otherwise ambiguous relaxations separately and
explicitly mark observable crash-free spans. Preserve ambiguous/missing support
as unknown. Lock the annotations before revealing predictions, then score both
rules and both frozen models with one-to-one timing matches and 1,000 shot
bootstrap replicates; do not use these shots to retune this benchmark. The owner
is away and the queue is pending, so no blind expert results are reported.

## Alias
sawtooth, sawtooth, sawtooth oscillation, sawtooth crash, st crash, sawtooth-free

## Future Implementations

## Reference
- S. von Goeler, W. Stodiek and N. Sauthoff, "Studies of internal disruptions and
  m = 1 oscillations in tokamak discharges with soft-X-ray techniques", Phys. Rev.
  Lett. 33, 1201 (1974).
- I. T. Chapman, "Controlling sawtooth oscillations in tokamak plasmas", Plasma
  Phys. Control. Fusion 53, 013001 (2011).

## Contact
- **Nathaniel Chen**: nathaniel [at] princeton [dot] edu
