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

- **16 candidates on Hanson's shots are unchecked.** These shots have no Mirnov data,
  so the `rwm_onsets` v4 screen that drops co-rotating n = 1 tracks could not run on
  them.
