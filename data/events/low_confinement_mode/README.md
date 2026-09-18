# Low Confinement Mode

## Description
The L-mode is the baseline confinement regime of a diverted tokamak: no edge
transport barrier, turbulent transport across the whole minor radius, flat edge
profiles and a divertor D-alpha level that is high and unstructured. Energy
confinement follows the ITER89-P scaling (tau_E ~ I_p^0.85 R^1.2 n^0.1 P^-0.5 ...)
and is roughly half of the H-mode value at the same conditions. Every DIII-D
discharge starts in L-mode; it becomes H-mode only if the heating power exceeds the
L-H threshold, and falls back at an H-L transition when power drops or at the ramp
down. L-mode is also the target regime for some heat-flux and turbulence studies.

Typically found via the absence of a D-alpha drop / pedestal, a low H98 factor and
ELM-free but noisy D-alpha.

## Method

The local formatter reads the explicit confinement intervals supplied in
`raw/Jalal_28042024_confinement_regime_shotlist.csv`. Category 1 means L;
category 0 means another explicitly annotated regime. Blank flags, unlabelled
rows, and gaps remain unknown. No regime is inferred from BES acquisition times
or the transition-note columns. The existing `dalpha_lh` detector remains a
separate source of transition points.

Intervals are aggregated into half-open 50 ms bins: any positive overlap makes
a bin 1. A bin spanning a transition can therefore be positive in both H and L
outputs; it means each regime occurred within that bin. Consecutive equal known
bins are compressed into CSV intervals. Exact original bounds remain in raw/.

## Provenance

Original labels from Jalal Butt, created in 28 April 2024. Both H and L
raw folders contain the same source table. Explicitly labelled `Test only` rows are used for the gold verification dataset.

## Models
**stable**: none

**latest**: none

**all**:
- dalpha_lh | 2026_09_12 (rule-based L-H / H-L transition detector, not a regime label)

## Alias
- l-mode
- lmode
- low confinement mode
- ohmic (only when no auxiliary heating is applied)

## Reference
- P. N. Yushmanov et al., "Scalings for tokamak energy confinement", Nucl. Fusion
  30, 1999 (1990). (ITER89-P)
- F. Wagner, "A quarter-century of H-mode studies", Plasma Phys. Control. Fusion 49,
  B1 (2007).

## Contact
- **Jalal Butt**
- **Kouroche Bouchiat**
- **Nathaniel Chen**: nathaniel [at] princeton [dot] edu

## Tables

Inventory row: L-Mode; interval lexicon id: `lmode`.

The scope inventory is [`discrete_labels.csv`](../discrete_labels.csv).
`raw/` holds the untouched provided lists; `format/` holds their
common-schema CSVs and metadata. Each `extend_<model>/` holds one
producing source's output on the project shot list. Categories without
a producer have no `extend_*` directory. See the [table guide](../README.md).

Regenerate registered raw tables from the repository root:

```bash
PYTHONPATH=src python scripts/labeler/labels_format.py
```

[`formatter.py`](formatter.py) writes:

- `format/low_confinement_mode_format_2026_v1.csv`
- its `.meta.json` sidecar
- `format/shots/<shot>.npz`

The CSV columns are `shot,category,t_start,t_end,confidence`; `events.yaml`
declares milliseconds. Blank confidence means unknown. Each per-shot sparse
file stores a time × 20-rho grid, with scalar labels broadcast across rho.
The time axis starts at 0 and ends at 6000 ms, extended for later annotations.
Shots without usable annotations receive all-unknown grids and no CSV rows.

[`example.ipynb`](example.ipynb) opens `format/shots/<shot>.npz` directly and plots the saved 50 ms grid.
Change `source` to an existing extended folder to view extended labels.

```bash
pixi run -e labelmaker python data/events/low_confinement_mode/formatter.py
```

## Category

| ID | Label |
| --- | --- |
| 0 | Low confinement mode absent within another explicitly labelled regime |
| 1 | Low confinement mode present (L) |

The JSON sidecar's `categories` mapping defines the same binary IDs;
`label_mapping.positive_raw_regimes` identifies the corresponding source flags.
Unknown cells are stored separately from category 0.


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
