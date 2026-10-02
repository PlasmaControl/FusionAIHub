# Sawtooth Oscillation

## Description
Sawtooth are periodic relaxations of the plasma core when the central safety factor
falls below one: the m/n = 1/1 internal kink grows, magnetic reconnection
(Kadomtsev) flattens the core temperature and density inside the mixing radius, and
the profile then re-peaks over a few tens of ms - a slow ramp and a fast crash,
hence the name. The inversion radius r_inv (where the crash changes sign from a
drop inside to a rise outside) marks the q = 1 surface; on DIII-D it is typically at
rho ~ 0.3-0.6. Sawtooth crashes can seed NTMs, expel fast ions and trigger ELMs, and
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

**latest**: none

**all**:
- ece_sawtooth | 2026_09_12 (rule; omnimode inversion test, envelope-once port)

## Inputs
**ece_sawtooth**:
- `ECE` (48 ch)
- `SXR` (first lit fan of `SX90RM1F`, `SX90RP1F`, `SX90RM1S`, `SX90RP1S`)
- `Ip` (plasma start)

**sawtooth_frames (round three)**:
- `ECE Te, ch 20-23`, `ECE Te, ch 24-27`, `ECE Te, ch 28-31`, `ECE Te, ch 32-35`
- `SXR` (optional)

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
On shot 198658 it finds 45 sawtooth with a median period of 76 ms.

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
