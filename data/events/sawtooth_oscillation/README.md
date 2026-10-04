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
- saw_physics | 2026_10_03 (Gude multichannel inversion plus Muscatello central-drop/train criteria; nominal geometry and bias-aware q evidence)
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
- EFIT-axis core ECE and low-field-side outer ECE at nominal geometric ρ=0.4–0.65
- Mirnov 0–1 mean and Ip in MA; missing values use fitting-shot means

**saw-ours**:
- The physical first-40-channel ECE array, 100 ms context at 10 kHz; unverified
  channels 40–47 and third-harmonic overlap samples are masked

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

## Physics-rule labels and validation

These are **physics-rule labels validated only by the checks described** in the
[current-state report](../../../docs/labeler/sawtooth_results.md). The rule uses
Gude-style POSR, multichannel coincidence, core loss / outer gain, central drop,
stable trains and nominal EFIT localization. It screens harmonic overlap and
uses EFIT01 bias-aware q-min conflicts and sustained high-q absence evidence.
POSR-qualified phase edges require ≥10 ms periods and a shuffled-time null that
repeats the same group search. Geometry is nominal, with no flux calibration.
Present/absent/uncertain/unassessed states remain distinct. Production labels
are not replaced. Old `ece_sawtooth` disagreement and its reader audit are in
the report; the retained legacy rule and current catalog detector differ.
Valid core ECE defines observability: missing ECE, low temperature and detected
cutoff yield `unassessed`, and trains split at observability gaps. Native-rate
antialiasing precedes decimation to 10 kHz. Where local neutron-rate and Mirnov
data exist, their drop/burst flags give optional corroboration; no SXR
corroboration is claimed without verified core/edge spatial pairing. Absence
requires complete candidate-free context with a noise-resolved core relaxation
test, or sustained EFIT01 q-min ≥ 1.5; ambiguous observable support remains
uncertain. The untracked exports in `extend_saw_physics/` hold four-state spans
and crash points; they are additive research labels.

Complete population label shards are at
`$LABELER_ROOT/round4/saw/fix3/labels/`. Verify with `sha256sum -c SHA256SUMS`
from that directory. The `SHA256SUMS` file has sha256
`3c70d44325cf98a0d4e9cb92efd3f6219e2c3fd13bf4f4bd7c57ee52d24168c3`; individual CSV hashes are in
`outputs/labeler/sawtooth/fix3/label_manifest.json`. Cohort shards are a separate
bundle at `$LABELER_ROOT/round4/saw/fix3/cohort_labels/`; `extend_saw_physics/`
is the untracked integration copy. Git does not carry the large label store.

Both learned models use three whole-shot TRAIN folds with inner-shot selection
of checkpoint, hyperparameters and thresholds; CUDA training stops on
inner-selection loss patience. The trivial derivative and always-present
baselines use the same folds. The three reviewed shots and the blind test split
are excluded from training and tuning. `saw-hl3` receives adapted inputs
(EFIT-axis core ECE, low-field-side outer ECE, Mirnov, Ip) while `saw-ours`
receives the first 40 ECE channels, so comparisons include input information as
well as architecture. The second held-out set is the 47 nonexpert
fixed-validation shots. Headline scores are **conditional agreement with the
physics rule on assessed bins** and include excluded-pick counts and paired
shot-bootstrap comparisons. HL-3 crash timing is **derivative picker gated by
HL-3**, an adapted baseline, rather than a learned crash head. Reviewed spans
were anchored to old suggestions and used in previous rule revisions; they
are exploratory and provide no independent crash-time precision/recall; the
190637 span may include edge-originated relaxations.

## Blind crash-time annotation queue

The prediction-free input pack is
`$LABELER_ROOT/round4/saw/fix3/annotation_pack/`. It contains sensor windows,
observable masks, nominal geometry and blank annotation targets. Random windows
are frozen before prediction access. Candidate-free, model-negative, uncertain
and disagreement cases supplement the primary probability sample. Keep the
private `selection_audit/` directory and every detector/model prediction hidden.
Mark crash times, timing tolerances, positive/negative observable spans and
ambiguity masks; lock annotations before revealing picks. Use preregistered
sampling weights and whole-shot bootstrap intervals. Approximately 97
independent positive events give a worst-case 95% recall half-width of 0.1;
shot clustering reduces the effective count. The owner is away: annotation is
pending and physical accuracy remains unvalidated. No model is recommended.

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
