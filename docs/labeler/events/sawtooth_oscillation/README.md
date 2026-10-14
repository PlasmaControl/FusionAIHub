# Sawtooth Oscillation

> The label data is `data/events/sawtooth_oscillation/`: the `raw/`, `format/`, `review/` and `extend_*/` paths
> below are relative to it. It is not in git. `scripts/labeler/hf_publish.py pull`
> restores the label tables (`format/`, `review/`, `extend_*/`), but not `raw/`: the
> third-party originals are not in the Hugging Face release; they are on the Stellar
> checkout's disk and archived in `/scratch/gpfs/EKOLEMEN/nc1514/labeler/raw_archive/`.

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
- saw_physics v2 | 2026_10_14 (superseded by v3 the same day: the 2026_10_04 rows on all 13,666 physics shots, gated by EFIT02 then EFIT01 only; on silver+gold 841.5 s absent, 200.2 s present, 923.3 s uncertain, 788.6 s not observable)
- saw_physics v3 | 2026_10_14 (superseded by v4 the same day: the 2026_10_04 rows on the roster, silver, gold and reviewed shots (609 shots) with every no-q = 1 second forced absent by the q = 1 gate, which now falls back EFIT02 -> EFIT02ER -> EFIT01; on silver+gold 845.5 s absent, 200.2 s present, 921.5 s uncertain, 786.4 s not observable; see "q = 1 gate")
- saw_physics v4 | 2026_10_14 (the review source since 2026-10-14: v3 rebuilt after 21 of its shots got an EFIT02ER or EFIT01 fallback file where EFIT02 has one slice; rows differ on 2 shots, 199166 and 199353; on silver+gold 845.7 s absent, 200.2 s present, 921.3 s uncertain, 786.4 s not observable)
- saw-hl3-full | 2026_10_05 | AUROC: 0.870 [0.827, 0.908] | AUPRC: 0.949 [0.929, 0.966] | F1: 0.870 [0.835, 0.899] (OuYang HL-3 CNN + bidirectional LSTM on the paper's nine offline inputs, label-free SXR chords; presence out of fold on 2 ms bins, 390 shots, conditional agreement with the physics rule on assessed bins (not an independent truth); crash F1 ±2 ms 0.894 [0.874, 0.910] (derivative picker gated by HL-3); three-class window accuracy 0.715 [0.662, 0.769] against 0.522 [0.447, 0.589] for the majority class; always present AUROC 0.500, AUPRC 0.736, F1 0.848; no-q = 1-negatives view (EFIT02 -> EFIT02ER -> EFIT01 gate) F1 0.483 [0.427, 0.537], AUROC 0.752 [0.703, 0.799], crash F1 ±2 ms 0.679 [0.611, 0.746])
- saw-hl3-ece | 2026_10_04 | AUROC: 0.765 [0.697, 0.823] | AUPRC: 0.890 [0.853, 0.919] | F1: 0.843 [0.796, 0.881] (the same network on the earlier four-input ECE adaptation; presence out of fold on 2 ms bins, 390 shots, conditional agreement with the physics rule on assessed bins (not an independent truth); crash F1 ±2 ms 0.894 [0.872, 0.912] (derivative picker gated by HL-3); three-class window accuracy 0.644 [0.591, 0.697] against 0.522 [0.447, 0.589] for the majority class; always present AUROC 0.500, AUPRC 0.736, F1 0.848; no-q = 1-negatives view (EFIT02 -> EFIT02ER -> EFIT01 gate) F1 0.411 [0.362, 0.464], AUROC 0.574 [0.520, 0.632], crash F1 ±2 ms 0.656 [0.583, 0.727])
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
[`sawtooth_reference_check.py`](../../../../scripts/labeler/sawtooth_reference_check.py)
and its committed reference record.
On shot 198658 it finds 45 sawtooth with a median period of 76 ms.

## Physics-rule labels and validation

These are **physics-rule labels validated only by the checks described** in the
[current-state report](../../sawtooth_results.md). The rule uses
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

## q = 1 gate

A sawtooth needs a q = 1 surface. `labeler.sawtooth.qgate` calls time `no_q1`
where q_min stays at or above a cut for at least 100 ms. Each whole-ms cell is
decided by the best equilibrium that covers it (owner, 2026-10-14: "if efit02
doesnt exist, use efit01 or efit02er"):
1. EFIT02 (MSE-constrained) `a_qmin >= 1.2`, from `$LABELER_ROOT/mhd/efit02/<shot>.npz`.
2. Else EFIT02ER (also MSE-constrained) `>= 1.2`, from
   `$LABELER_ROOT/mhd/efit_fallback/<shot>.npz` when its `tree` is `efit02er`.
3. Else the magnetics-only EFIT01 `>= 2.0`, from the fallback file when its
   `tree` is `efit01`, else from the local stores (features, corpus or raw
   cache; never fetched). Where both exist they agree sample for sample.

A tree is silent outside its valid time base and across gaps of more than
100 ms between valid slices, so one with a single slice covers nothing. The
gate never extrapolates: a cell that no tree covers is `unknown`. Each sample
records the tree that decided it. Every table and benchmark key built on the
gate records each equilibrium file it read (tree, path under `$LABELER_ROOT`,
size, sha256 of the file and of the q_min series), plus one digest over them.

**Review source `saw_physics` v4**
(`$LABELER_ROOT/suggestions/saw_physics/v4/sawtooth_oscillation_suggest_saw_physics_v4.csv`).
The pointer moved from v3 to v4 on 2026-10-14, from v2 to v3 earlier that day;
v2 had replaced `ece_sawtooth` v1 the same day.
- v4 is v3 rebuilt after the last fallback files landed: 39 shots whose EFIT02
  has one slice got EFIT02ER (12) or EFIT01 (27). 21 of them are in this table
  (8 EFIT02ER, 13 EFIT01). Rows differ on 2 silver shots, both now decided by
  EFIT02ER:
  - 199166: absent 0.685 -> 0.777 s, uncertain 1.577 -> 1.485 s
  - 199353: absent 0.683 -> 0.794 s, uncertain 1.829 -> 1.718 s

  The shot set and every other row are unchanged, 199597 included. The numbers
  below are v4's.
- Coverage: the sawtooth roster (`shots.csv`), silver, gold and every shot in
  `review/labels.csv`. That is 614 shots, none blind. Five roster shots (179310, 180090,
  180406, 180627, 184084) have no physics record and no window, so the table
  holds 10,624 rows on 609 shots.
- The review page's first read takes 0.53 s, against 10.5 s for v2's
  13,666 shots. The rest of the population stays in `extend_saw_physics`.
- No-q = 1 time is absent, whatever the rule called it (uncertain, either q-prior
  state, unassessed). Present rows keep their category and times. The one
  exception is that gaps under 1 ms between physics spans are closed onto the
  earlier span, which adds at most about 1 ms per shot.
- A shot with no physics record (64 shots) is absent in no-q = 1 time and not
  observable elsewhere in its catalog window.
- Equilibrium files read: EFIT02 for 520 shots, EFIT02ER for 15, and EFIT01 for 576
  (from the fallback file on 95 shots). Every shot has at least one tree.
- Seconds of gate time on the 609 shots, by tree and state:

  | tree | no q = 1 | q = 1 possible |
  |---|---|---|
  | EFIT02 | 668.8 | 1114.5 |
  | EFIT02ER | 3.3 | 25.3 |
  | EFIT01 | 250.8 | 1028.4 |

  No tree covers 55.2 s, which is unknown.
- On the 500 silver+gold shots, seconds by category:

  | | absent | present | uncertain | not observable |
  |---|---|---|---|---|
  | physics rows, before the gate | 85.5 | 200.2 | 1409.9 | 1058.0 |
  | v2 | 841.5 | 200.2 | 923.3 | 788.6 |
  | v3 | 845.5 | 200.2 | 921.5 | 786.4 |
  | v4 | 845.7 | 200.2 | 921.3 | 786.4 |

- Silver+gold shots with no absent time: 319 of 500 before the gate, 28 of 500
  in v3 and v4. Both counts are over the same 500 shots and include the 63 shots with
  no physics record.
- On all 609 shots, the added absent time came from these states:
  - 224.7 s uncertain
  - 186.6 s `q_prior_ece_contradicted`
  - 157.2 s `q_prior_untested`
  - 159.9 s unassessed
  - 137.3 s without a record
- Rows differ from v2 on 46 of the 609 shots:
  - 41 shots got their first EFIT02 file after v2 was built: the fallback run
    fetched EFIT02 for the roster shots that had never been tried.
  - 3 shots gain absent time from EFIT02ER's 1.2 cut where EFIT01's 2.0 cut
    decided before: 189879, 190046 and 202041. Four more shots use EFIT02ER
    (189090, 189383, 190271, 201395) with unchanged rows.
  - 170724 and 204497 had no equilibrium at all and are now gated by EFIT01.
- Present time inside no-q = 1 time is 0.567 s on 4 shots:
  - 196146: 42 ms
  - 196154: 277 ms
  - 199166: 150 ms, new in v4, where EFIT02ER q_min is >= 1.2
  - 200917: 98 ms

  199166 and 200917 are the silver shots among them; no gold shot has any.
- On the gold TM shots, everything from 100 ms on is absent up to:
  - 4181 ms on 199597, where EFIT02 q_min falls below 1.2 (one 44 ms present
    train follows, at 6070-6114 ms)
  - 5980 ms on 199603
  - 5307 ms on 199607

  These rows are the same as in v2.

**`crash` v3 for `mhd_n1.m1`**
(`$LABELER_ROOT/suggestions/crash/v3/mhd_n1.m1_suggest_crash_v3.csv`). This is
crash v1 with present time inside no-q = 1 time turned absent. Its `per_shot` meta
is recomputed from the v3 rows: runs, present ms, no-q = 1 ms, the present time
removed, and the equilibrium files read.
- All shots: present goes from 684.0 s to 631.3 s; 71 of 490 shots lose some.
- Silver+gold: 492.5 s before, 451.6 s after; 64 of 379 shots lose some.
- It differs from crash v2 on 2 shots outside silver and gold: 189998 and 195032.
  Both got their first EFIT02 after v2 was built.
- The MHD stream owns the `mhd_n1.m1` pointer.

**Benchmark view `no_q1_negatives`** (in `outputs/labeler/sawtooth/fix5/benchmark.json`
and in the detail file `$LABELER_ROOT/round4/hl3/work/benchmark.json`, next to
`old_negatives_sensitivity`). It was rerun on 2026-10-14 with the
EFIT02 -> EFIT02ER -> EFIT01 gate.
- It scores the saved fold predictions again, with observable no-q = 1 time added
  to the assessed bins. The labels are unchanged, so present spans stay positive:
  2 out-of-fold bins.
- No model is refit.
- Shots by the best tree that added negatives to them (`gate_shots`; "none"
  means the gate added nothing to the shot):

  | | EFIT02 | EFIT02ER | EFIT01 | none |
  |---|---|---|---|---|
  | out of fold (390 shots) | 241 | 3 | 128 | 18 |
  | fixed validation (45 shots) | 27 | 0 | 14 | 4 |

  The first gate (EFIT02, else the local EFIT01) gave 209 / 164 / 17 out of fold
  and 22 / 20 / 3 in fixed validation. Most of the move is shots that got their
  first EFIT02 file in the fallback run (53 out of fold, 6 in fixed validation).
- Added negative seconds by tree (`added_seconds_by_tree`, on the signal grid):
  out of fold 296.0 EFIT02, 2.4 EFIT02ER and 123.0 EFIT01; fixed validation 28.8,
  0 and 14.2. In 2 ms bins over the primary negatives the view adds 210,850 bins
  out of fold (202,295 with the first gate) and 21,536 in fixed validation (19,875).
- `equilibria` lists the files read by tree with digests: out of fold EFIT02 336,
  EFIT02ER 4, EFIT01 390 (digest `30b82b52...`); fixed validation 41, 1 and 45
  (`f11ce32e...`).
- The rerun was a full dry run (`evaluate --write-to <copy>`, which now also writes
  the split manifest to the copy). The copy was compared key by key with the
  installed files, and only `no_q1_negatives` was replaced in both. Every other key
  reproduces exactly, except the expert subkeys. Those moved slightly (for example
  expert positive bins 3542 to 3538) because `review/labels.csv` was regridded to
  10 ms after the 2026-10-09 run, so the files keep their installed values.

Presence F1 on 2 ms bins, and crash F1 at ±2 ms in the no-q = 1 view:

| model | OOF primary F1 | OOF old-negatives F1 | OOF no-q = 1 F1 | OOF no-q = 1 AUROC | OOF no-q = 1 crash F1 | fixed-val no-q = 1 F1 | fixed-val no-q = 1 AUROC |
|---|---|---|---|---|---|---|---|
| saw-hl3-ece | 0.843 | 0.491 | 0.411 [0.362, 0.464] | 0.574 | 0.656 | 0.516 [0.332, 0.653] | 0.706 |
| saw-hl3-full | 0.870 | 0.549 | 0.483 [0.427, 0.537] | 0.752 | 0.679 | 0.556 [0.375, 0.681] | 0.920 |
| saw-ours | 0.976 | 0.802 | 0.685 [0.623, 0.737] | 0.899 | 0.742 | 0.694 [0.470, 0.855] | 0.946 |
| saw-derivative | 0.994 | 0.855 | 0.729 [0.667, 0.785] | 0.870 | 0.657 | 0.838 [0.658, 0.943] | 0.913 |
| saw-always-present | 0.848 | 0.478 | 0.411 [0.358, 0.464] | 0.500 | - | 0.473 [0.285, 0.625] | 0.500 |

Against the first gate, no-q = 1 F1 fell by 0.007 to 0.012 out of fold and by
0.011 to 0.022 in fixed validation, AUROC moved by at most 0.008, and crash F1
fell by 0.008 out of fold. The order is the same except out of fold, where
saw-hl3-ece (0.4115) and always present (0.4110), tied within 0.002 under both
gates, swap places.

Bins in each view (positive / negative):

| view | primary | old negatives | no-q = 1 |
|---|---|---|---|
| out of fold | 84,109 / 30,200 | 84,109 / 183,696 | 84,111 / 241,050 |
| fixed validation | 10,505 / 1,846 | 10,505 / 19,496 | 10,505 / 23,382 |

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
