# Data notes to check

Shots and inputs whose labels look wrong for a reason outside the labeler's rules.
Nothing here has been changed in the tables; each item says what a reviewer should
look at. Add to the list as issues turn up; remove an item once it is settled.

## Equilibrium (q = 1 gate and MHD m/n)

The equilibrium order is EFIT02, else EFIT02ER, else EFIT01, and a tree counts only
with at least 2 time slices (owner, 2026-10-14). The q = 1 gate is described in
`docs/labeler/events/sawtooth_oscillation/README.md` ("q = 1 gate").

- **199166 (silver): a sparse EFIT02ER cuts a sawtooth train.** EFIT02 has one
  slice, so EFIT02ER decides the gate: 17 slices, median q_min 1.25. It calls 92 ms
  between two sawtooth runs "no q = 1 surface", so `saw_physics` v4 marks that time
  absent inside an otherwise continuous train. The fallback order ranks this sparse
  EFIT02ER above a dense EFIT01. Check the ECE/SXR traces there; if the sawteeth
  continue, the gate is wrong on this shot.
- **202189: EFIT02ER looks broken.** Its median q_min is 0.113, which is not a
  physical DIII-D value. The shot is in no label table today, so no label is
  affected; it should not be trusted if it is added.
- **mhd_n1 1/1 time without an equilibrium (151 s).** Where EFIT02 exists but covers
  only part of a shot, the uncovered time is not observable for the 1/1 child (the
  fallback trees are used only when EFIT02 is missing or has one slice). Covering it
  needs EFIT01 fetched for those shots and a rule that fills EFIT02 gaps from it.

## Resistive wall mode

- **36 of `rwm_onsets` v5's 628 shots are not screened on δBp.** 32 have Ip on an
  ambiguous legacy clock (23 up to 2018, 9 from 2021 on) and four older shots
  (132240, 132710, 138388, 142111) have probe records on clocks the resolver
  refuses. Their time is not observable outside the onset stretches.
- **From 2021 on δBp comes from single probes.** PTDATA has no MPID difference
  points there, so 0.5 x (MPI66M{a}F - its partner) stands in on 534 roster shots;
  the 337 deg probe is gone from 2018 on and 157's partner is the 340 deg probe,
  2.4 deg off. On 176088 and 178872 the stand-in matches MPID with gain 0.46-0.49
  and correlation 0.98-0.99 (`$LABELER_ROOT/round5/rwm/checks/fb_compare.out`;
  178872's MPID097U is dead). No 2021+ shot has a curated onset to check it on.
- **358 v5 candidates on Hanson's shots are unchecked.** These shots have no
  Mirnov record in the MHD cache, so the n = 1 track gate could not run on them,
  and N1RMS no longer drops anything. The classifier leaves them out of both
  classes; scoring them negative halves rwm-ours's F1 (0.529 to 0.260).
- **The start condition is βN > 2.4 li, set on the references.** 2.4 keeps the
  seven Okabayashi 2005 references (lowest start βN/li 2.76) and all 44 Hanson
  onsets a δBp run reaches (lowest 2.46); 4 li sits above these plasmas' no-wall
  limit. Where EFIT01 has no sample before a run (it often ends before a terminal
  mode) the condition is untested and the run kept: 10 Hanson and 194 roster
  candidates (`checks/rule_survey.csv`).
- **The two negatives stay candidates.** 120290's internal mode at 1680 ms
  (βN/li 4.0) and 119666's three runs (3.7-5.9) pass the start condition, and their
  N1RMS overlaps the references'. Okabayashi et al. told 120290's mode apart on the
  Mirnov dδBp/dt, which the screen does not read.
- **The comparison shots are not RWM-free on δBp.** The v5 screen keeps 805
  uncertain candidates (9.5 s) on 129 of 174 screened comparison shots, none
  Mirnov-checked (`outputs/labeler/rwm/onsets_source_v5.json#/comparison`).
- **The 2003-04 reference shots were prepared by hand.** Their PTDATA clocks are
  multi-rate, which `resolve_fdp` refuses. Their IP, MPID*U and I-coil points were
  written into the raw cache with only strictly increasing samples (attrs
  `prepared`; `$LABELER_ROOT/round5/rwm/checks/scripts/paper_prep.py`).
- **rwm-ours reads the labels' own sensor.** The v5 present spans end where δBp
  falls back under the threshold, so its presence score is partly agreement with
  the rule; Hanson's onset times are the only curated evidence.
- **The δBp running median absorbs perturbations longer than about 50 ms.** It also
  passes n = 3 and any locked n = 1 mode; only the Mirnov gate separates a locked
  tearing mode while it still rotates (the N1RMS cover is recorded, not used).
- **The pickup coupling is fitted over the whole shot.** rwm-ours's trailing δBp
  applies a per-shot coupling fitted on all of the shot, a static calibration
  trimmed of bursts.
- **rot_q2_khz is NaN where q_min >= 2:** 52% of the 2014 and 86% of the 2018
  slices.
