# Improved Energy Confinement Mode

## Description
The I-mode is a regime with an edge barrier in the TEMPERATURE channel only: energy
confinement at or above H-mode levels (H98 ~ 1) while particle and impurity
transport stay at L-mode values, so the density pedestal is absent and there are no
ELMs. It is accessed with the ion grad-B drift pointing AWAY from the active
X-point (the "unfavourable" configuration, which raises P_LH and opens a power
window between L-mode and H-mode) and is marked by a weakly coherent mode (WCM) in
edge fluctuations at ~100-300 kHz and by a geodesic-acoustic-mode-like oscillation
at ~20 kHz.

First identified as a distinct regime on Alcator C-Mod (Whyte et al. 2010),
following earlier ASDEX Upgrade observations of an "improved L-mode"; DIII-D
demonstrated it in the 2014-2015 campaigns.

Typically found via the temperature-only pedestal in edge Thomson profiles, the
absence of the density rise at the transition, the WCM in reflectometry / BES /
magnetics, and unfavourable-drift shape files.

## Method
Not started. No detector, model or table writes this label. The inventory notes it
may be identifiable in dFL (fast-ion loss) data; a defensible rule would need the
grad-B drift direction (shape) plus a T_e pedestal without an n_e pedestal.

## Provenance
None yet. `raw/` is empty.

## Models
**stable**: none

**latest**: none

**all**:
- none

## Alias
- i-mode
- imode
- improved energy confinement mode
- improved l-mode

## Reference
- D. G. Whyte et al., "I-mode: an H-mode energy confinement regime with L-mode
  particle transport in Alcator C-Mod", Nucl. Fusion 50, 105005 (2010).
- A. Marinoni et al., "Characterization of density fluctuations during the search
  for an I-mode regime on the DIII-D tokamak", Nucl. Fusion 55, 093019 (2015).

## Contact
- **Nathaniel Chen**: nathaniel [at] princeton [dot] edu

## Tables

Inventory row: I-Mode; lexicon id pending producer task.

The scope inventory is [`discrete_labels.csv`](../discrete_labels.csv).
`raw/` holds the untouched provided lists; `format/` holds their
common-schema CSVs and metadata. Each `extend_<model>/` holds one
producing source's output on the project shot list. Categories without
a producer have no `extend_*` directory. See the [table guide](../README.md).

Regenerate registered raw tables from the repository root:

```bash
PYTHONPATH=src python scripts/labeler/labels_format.py
```

No raw table is registered for this category yet.

## Verification

[`verification.ipynb`](verification.ipynb) plots one shot's signals against its
saved labels and takes back corrections. The review roster is
[`shots.csv`](shots.csv). See the [table guide](../README.md) for the roster
schema.
