# Resistive Wall Mode

## Description
The resistive wall mode (RWM) is the external kink (mostly n = 1) that a perfectly
conducting wall would stabilise but a real, resistive wall only slows: the mode
grows on the wall's flux-penetration time tau_w (milliseconds on DIII-D) instead of
the Alfvén time. It exists in the window between the no-wall and ideal-wall beta
limits,

    beta_N^no-wall  <  beta_N  <  beta_N^ideal-wall

(empirically beta_N^no-wall ~ 4 l_i on DIII-D), and is stabilised by plasma rotation
and kinetic resonances or by active feedback with the C- and I-coils. Its onset
often ends a high-beta_N discharge in a disruption; error-field amplification
(resonant field amplification, RFA) is its marginally stable precursor.

First identified in the mid 1990s in DIII-D wall-stabilised high-beta experiments
and on the HBT-EP and PBX-M devices.

Typically found via low-frequency (0-10 kHz) n = 1 magnetics - saddle loops and
poloidal probe arrays - as a slowly growing, slowly rotating or locked mode near the
no-wall limit.

## Method
No detector writes `rwm`. The two curated onset tables are the only evidence; each
row becomes a point event with `evidence_kind = database`,
`source = database:rwm_onsets_<year>`, NaN confidence and NaN coverage - a listing
says nothing about the interval anybody examined, so an absent shot is never a
negative. `NTOR` (toroidal mode number) and `MODE_TYPE` travel in `attrs`. The
`extend_rwm/recommender_v1.csv` scan over the 500 project shots is header-only:
none of the 33 listed shots is in the corpus, and that is a completed zero, not an
absence claim.

## Provenance
Obtained using reference dataset from Jeremy Hanson: `rwm_onsets_2017.csv`
(30 onsets, 20 shots, 156785-158023, 0.856-3.061 s) and `rwm_onsets_2024.csv`
(26 onsets, 13 shots, 176067-176092, 1.654-4.400 s), 56 onsets over 33 distinct
shots, kept byte-for-byte in `raw/`. All 33 predate the FAITH corpus (185601+).

## Models
**stable**: none

**latest**: none

**all**:
- none

## Alias
- resistive wall mode
- rwm
- resonant field amplification (precursor)
- rfa

## Reference
- M. S. Chu and M. Okabayashi, "Stabilization of the external kink and the
  resistive wall mode", Plasma Phys. Control. Fusion 52, 123001 (2010).
- A. M. Garofalo et al., "Sustained stabilization of the resistive-wall mode by
  plasma rotation in the DIII-D tokamak", Phys. Rev. Lett. 89, 235001 (2002).

## Contact
- **Jeremy Hanson** (original dataset): hansonjm [at] fusion [dot] gat [dot] com
- **Nathaniel Chen**: nathaniel [at] princeton [dot] edu

## Tables

Inventory row: Resistive Wall Mode; lexicon id: `rwm`.

The scope inventory is [`discrete_labels.csv`](../discrete_labels.csv).
`raw/` holds the untouched provided lists; `format/` holds their
common-schema CSVs and metadata. Each `extend_<model>/` holds one
producing source's output on the project shot list. Categories without
a producer have no `extend_*` directory. See the [table guide](../README.md).

Regenerate registered raw tables from the repository root:

```bash
PYTHONPATH=src python scripts/labelmaker/labels_format.py
```

Registered tables: `rwm_onsets_2017` (30 events, 20 shots) and
`rwm_onsets_2024` (26 events, 13 shots). The header-only
`extend_rwm/recommender_v1.csv` records a completed zero-event scan;
it does not establish absence of RWM. Its regeneration command is
documented in the [table guide](../README.md#committed-rwm-evidence).
