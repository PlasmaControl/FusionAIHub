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
- saw_physics | 2026_10_04 (Gude multichannel inversion plus Muscatello central-drop/train criteria; nominal geometry, local-field cutoff and ECE-validity guards, ECE-tested absence with a separate q-prior state)
- saw-hl3-full | 2026_10_05 | AUROC: 0.870 [0.827, 0.908] | AUPRC: 0.949 [0.929, 0.966] | F1: 0.870 [0.835, 0.899] (OuYang HL-3 CNN + bidirectional LSTM on the paper's nine offline inputs, label-free SXR chords; presence out of fold on 2 ms bins, 390 shots, conditional agreement with the physics rule on assessed bins (not an independent truth); crash F1 ±2 ms 0.894 [0.874, 0.910] (derivative picker gated by HL-3); three-class window accuracy 0.715 [0.662, 0.769] against 0.522 [0.447, 0.589] for the majority class; always present AUROC 0.500, AUPRC 0.736, F1 0.848)
- saw-hl3-ece | 2026_10_04 | AUROC: 0.765 [0.697, 0.823] | AUPRC: 0.890 [0.853, 0.919] | F1: 0.843 [0.796, 0.881] (the same network on the earlier four-input ECE adaptation; presence out of fold on 2 ms bins, 390 shots, conditional agreement with the physics rule on assessed bins (not an independent truth); crash F1 ±2 ms 0.894 [0.872, 0.912] (derivative picker gated by HL-3); three-class window accuracy 0.644 [0.591, 0.697] against 0.522 [0.447, 0.589] for the majority class; always present AUROC 0.500, AUPRC 0.736, F1 0.848)
- saw-ours | 2026_10_04 (PhaseNet-style multichannel ECE crash picker and train-presence head; GPU fit)

## Inputs
**ece_sawtooth**:
- `ECE` (48 ch)
- `SXR` (first lit fan of `SX90RM1F`, `SX90RP1F`, `SX90RM1S`, `SX90RP1S`)
- `Ip` (plasma start)

**sawtooth_frames (round three)**:
- `ECE Te, ch 20-23`, `ECE Te, ch 24-27`, `ECE Te, ch 28-31`, `ECE Te, ch 32-35`
- `SXR` (optional)

**saw-hl3-full**:
- The paper's offline set at 10 kHz: Ip, line-integrated density (CO2 chord V2),
  Mirnov rows 0–1 mean, SXR core and edge chords, EFIT01 stored energy, ECE core
  electron temperature, total beam power and ECH power (nine channels; a missing
  input is a masked channel)
- SXR chords chosen per shot without labels, because no SX90 chord geometry is
  available: the structured chord with the highest 100 Hz–2 kHz variance is the
  core, and the most anti-correlated structured chord further from the fan centre
  is the edge (method counts and per-input shot coverage are in the report)

**saw-hl3-ece**:
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
uses EFIT01 bias-aware q-min conflicts. POSR-qualified phase edges require
≥10 ms periods and a shuffled-time null that repeats the same group search.
Geometry is nominal, with no flux calibration. Present / absent /
q_prior_ece_contradicted / q_prior_untested / uncertain / unassessed states
remain distinct. Production labels are not replaced. Old `ece_sawtooth` disagreement and its reader audit are in
the report; the retained legacy rule and current catalog detector differ.
Valid core ECE defines observability: missing ECE, low temperature and detected
cutoff yield `unassessed`, and trains split at observability gaps. Native-rate
antialiasing precedes decimation to 10 kHz. Where local neutron-rate and Mirnov
data exist, their drop/burst flags give optional corroboration; no SXR
corroboration is claimed without verified core/edge spatial pairing. **Absent**
means a tested absence: no POSR-periodic core edge on any valid ECE channel at
nominal ρ<0.5 over a complete ±375 ms context, noise-resolved on at least two
channels (an edge counts only when its relative change is at least 2%, so a
core relaxation train below 2% is called quiet). Sustained EFIT01 q-min ≥ 1.5 is
a prior, not a test: that time is `q_prior_ece_contradicted` where it lies in
the absence-test context of a periodic edge or profile candidate (the ECE shows
relaxation evidence there) and `q_prior_untested` elsewhere; both are exported
as `uncertain` with the state name as the reason and are never a benchmark
negative. A density-cutoff guard uses the local field at the axis resonance
where it is mapped (Bt at R0, then a fixed guard, only as fallbacks; the report
gives the records by branch) and an ECE validity test (adjacent-channel step
ratio above 2, or a near-axis channel below 0.6 of the profile maximum,
sustained 20 ms; dead channels are excluded) marks unresolved ECE `unassessed`.
Ambiguous observable support remains uncertain. The
untracked exports in `extend_saw_physics/` hold the spans and crash points; they
are additive research labels.

The population run is complete: 13319 of 16909 corpus shots have a usable record under the final rule. The other 3590 carry no label, for these reasons: 2835 × absent waveform or incompatible clock; 426 × insufficient physical or finite ECE core (core channels, finite window, no ECE group); 266 × unreadable file or object; 63 × other (for example a geometry time axis that is not increasing).
Population label shards are at
`$LABELER_ROOT/round4/saw/fix5/labels/` (cohort and population shards). Verify with
`sha256sum -c SHA256SUMS` from that directory. The `SHA256SUMS` file has sha256
`47081aebcad868911bf3e38817b3e0c79e4ba9f02efe5fcee84780ad00a0ef14`; individual CSV hashes are in
`outputs/labeler/sawtooth/fix5/label_manifest.json`. The manifest also hashes
`prior_inputs/fix2_inputs.json`, a snapshot of the earlier round's inputs the
rule reads, and `freeze.json`. Cohort shards are the `cohort-*.csv` files in the
same directory; `extend_saw_physics/` is the untracked integration copy. Git does
not carry the large label store.

The learned models use three whole-shot TRAIN folds with inner-shot selection
of checkpoint, hyperparameters and thresholds; CUDA training stops on
inner-selection loss patience. The trivial derivative and always-present
baselines use the same folds. The three reviewed shots and the blind test split
are excluded from training and tuning. `saw-hl3-full` receives the paper's nine
offline inputs and `saw-hl3-ece` the earlier four (EFIT-axis core ECE,
low-field-side outer ECE, Mirnov, Ip) while `saw-ours` receives the first 40 ECE
channels, so comparisons include input information as well as architecture. The
second held-out set is the 47 nonexpert fixed-validation shots. Headline scores
are **conditional agreement with the physics rule on assessed bins** and include
excluded-pick counts and paired shot-bootstrap comparisons. HL-3 crash timing is
**derivative picker gated by HL-3**, an adapted baseline, rather than a learned
crash head. Reviewed spans were anchored to old suggestions and used in previous
rule revisions; they are exploratory and provide no independent crash-time precision/recall; the
190637 span may include edge-originated relaxations.

## Blind crash-time annotation queue

The prediction-free input pack is
`$LABELER_ROOT/round4/saw/fix5/annotation_pack/`. It contains sensor windows,
observable masks, nominal geometry and blank annotation targets. Random windows
are frozen before prediction access. Candidate-free, model-negative, uncertain
and disagreement cases supplement the primary probability sample. Annotators
receive the annotation pack only: no repository outputs and no access to
`round4/saw`. Keep the private `selection_audit/` directory and every
detector/model prediction hidden (the `predictions/` directories are mode
`go-rwx`).
Mark crash times, timing tolerances, positive/negative observable spans and
ambiguity masks; lock annotations before revealing picks. Use preregistered
sampling weights and whole-shot bootstrap intervals. Approximately 97
independent positive events give a worst-case 95% recall half-width of 0.1;
shot clustering reduces the effective count. The owner is away: annotation is
pending and physical accuracy remains unvalidated. No model is recommended.

## Alias
sawtooth, sawtooth oscillation, sawtooth crash, st crash, sawtooth-free

## Future Implementations

## Reference
- S. von Goeler, W. Stodiek and N. Sauthoff, "Studies of internal disruptions and
  m = 1 oscillations in tokamak discharges with soft-X-ray techniques", Phys. Rev.
  Lett. 33, 1201 (1974).
- I. T. Chapman, "Controlling sawtooth oscillations in tokamak plasmas", Plasma
  Phys. Control. Fusion 53, 013001 (2011).

## Contact
- **Nathaniel Chen**: nathaniel [at] princeton [dot] edu
