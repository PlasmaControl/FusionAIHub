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
Reference dataset from Jeremy Hanson: `rwm_onsets_2017.csv` (20 shots, 156785-158023) and `rwm_onsets_2024.csv` (13 shots, 176067-176092), 56 onsets over 33 distinct shots in `raw/`.

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
pixi run -e labelmaker python data/events/resistive_wall_mode/formatter.py
```

Registered tables: `rwm_onsets_2017` (30 events, 20 shots) and
`rwm_onsets_2024` (26 events, 13 shots). The header-only
`extend_rwm/recommender_v1.csv` records a completed zero-event scan;
it does not establish absence of RWM. Its regeneration command is
documented in the [table guide](../README.md#committed-rwm-evidence).

## Local formatter and example

[`formatter.py`](formatter.py) reads `../events.yaml` and writes a CSV containing
only `shot,category,t_start,t_end,confidence` under `format/`. Source provenance and
conversion assumptions are stored in its `.meta.json` sidecar. Raw files stay
unchanged. The shared conversion implementation is in
`src/labeler/events/source_formatters.py`.

[`example.ipynb`](example.ipynb) opens the saved per-shot sparse labels and plots the rho–time grid.
It also plots the original annotations through a shared helper and supports
available `extend_*` datasets. For these sources
without radial localization, each time label is broadcast across 20 rho bins.
Use the labelmaker Python environment to rerun it. See the [storage guide](../README.md)
for the sparse per-shot grid format used by extensions.


## Category

The CSV `category` column and grid values use integer IDs. The same mapping
is recorded in each JSON sidecar under `categories`.

| ID | Label |
| --- | --- |
| 0 | Absent |
| 1 | Present |

Unknown or unclassified grid cells are stored separately from 0. A dataset
containing only positive annotations does not establish absence elsewhere.
Sampled grids use 50 ms bins and 20 rho bins.

Per-shot labels are saved in `format/shots/<shot>.npz`. Each file includes
time and rho coordinates, sparse integer values, unknown-cell coordinates,
and the category ID-to-name mapping. The formatted plot reads these saved files. A separate original-label plot
reads the source annotations; neither plot reruns the formatter.

The notebook's last cell plots the category's original annotations alongside
the saved 50 ms grid. Original-label plots require the source files; the
formatted and extended plots continue to read only their selected NPZ files.

## Verification

[`verification.ipynb`](verification.ipynb) plots one shot's signals against its
saved labels and takes back corrections. The review roster is
[`shots.csv`](shots.csv). See the [table guide](../README.md) for the roster
schema.
