# Fishbone

## Description
Fishbones are bursting m/n = 1/1 internal kink modes driven by fast ions. The
resonance is with the trapped fast ions' toroidal precession, so the mode sits
near the precession frequency rather than near an Alfven gap, and each burst
chirps **downward** as the resonant fast-ion energy falls. On DIII-D a burst is
roughly 2-30 kHz and lasts a few ms, repeating through the beam-heated part of
the discharge. The name is the shape: the burst envelope on a Mirnov trace
looks like a fish skeleton.

Two kinds are distinguished by what sets the frequency: precession-frequency
fishbones under near-perpendicular neutral beams, and diamagnetic-frequency
fishbones (f ~ omega\*i) under tangential beams.

They expel fast ions, which shows up as neutron-rate drops and beam-ion losses,
and they can seed sawteeth and NTMs. A fishbone is close kin to a sawtooth
precursor - both are the 1/1 kink - and the two are easy to confuse on
magnetics alone.

First observed on PDX during near-perpendicular NBI (McGuire et al. 1983).

Typically found via the magnetic spectrogram (Mirnov / MHR probes) as a
repeated downward chirp in the 2-30 kHz band, confirmed as n = 1 from a
toroidal probe array, with supporting drops in the neutron rate.

## Method
Use the magnetic spectrogram and look for the n = 1 chirp: bursts in the
2-30 kHz band that sweep downward in frequency, on the beam-heated part of the
discharge, with toroidal mode number n = 1.

No detector exists yet. This is the stated method, not a description of one
that runs.

## Provenance
None. No curated table has been obtained, so `raw/` is empty and `format/`
holds nothing.

## Models
**stable**: none

**latest**: none

**all**: none

## Alias
- fishbone
- fishbones

## Reference
- K. McGuire et al., "Study of high-beta magnetohydrodynamic modes and
  fast-ion losses in PDX", Phys. Rev. Lett. 50, 891 (1983).
- L. Chen, R. B. White and M. N. Rosenbluth, "Excitation of internal kink modes
  by trapped energetic beam ions", Phys. Rev. Lett. 52, 1122 (1984).

## Contact
- **Nathaniel Chen**: nathaniel [at] princeton [dot] edu

## Tables

Inventory row: Fishbone; lexicon id: `fishbone`.

The scope inventory is [`discrete_labels.csv`](../discrete_labels.csv).
`raw/` holds the untouched provided lists; `format/` holds their
common-schema CSVs and metadata. Each `extend_<model>/` holds one
producing source's output on the project shot list. Categories without
a producer have no `extend_*` directory. See the [table guide](../README.md).

No raw table is registered for this category yet, so nothing appears in
`../events.yaml` and there is no `formatter.py`.

## Category

The CSV `category` column and grid values use integer IDs.

| ID | Label |
| --- | --- |
| 0 | Absent |
| 1 | Present |

Unknown or unclassified grid cells are stored separately from 0.

## Verification

[`verification.ipynb`](verification.ipynb) plots the magnetic spectrogram for
one shot and takes back corrections. The review roster is
[`shots.csv`](shots.csv). See the [table guide](../README.md) for the roster
schema.
