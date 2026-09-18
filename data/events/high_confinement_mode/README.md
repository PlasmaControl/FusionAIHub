# High Confinement Mode

## Description
The H-mode is the tokamak confinement regime with an edge transport barrier: a
narrow region inside the separatrix where the E x B shear suppresses turbulence, so
density and temperature form a steep pedestal and global energy confinement is
roughly twice the L-mode value (H98(y,2) ~ 1). Access needs the heating power to
exceed a threshold P_LH that scales roughly as n_e^0.7 B_T^0.8 S^0.9 (Martin 2008)
and is lower with the ion grad-B drift toward the X-point. The transition is abrupt:
the divertor D-alpha drops in a few ms while the density rises. Most H-modes carry
ELMs (see `edge_localized_mode`); QH-mode and WPQH are ELM-free variants.

First discovered on ASDEX in 1982 (Wagner et al.).

Typically found via the D-alpha drop at the transition, the pedestal in edge
Thomson profiles, the density rise and the confinement factor.

## Method

The local formatter reads the explicit confinement intervals supplied in
`raw/Jalal_28042024_confinement_regime_shotlist.csv`. Category 1 means H, QH, or WP;
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
- h-mode
- hmode
- high confinement mode
- l-h transition
- h-mode transition

## Reference
- F. Wagner et al., "Regime of improved confinement and high beta in neutral-beam-
  heated divertor discharges of the ASDEX tokamak", Phys. Rev. Lett. 49, 1408 (1982).
- F. Wagner, "A quarter-century of H-mode studies", Plasma Phys. Control. Fusion 49,
  B1 (2007).
- Y. R. Martin et al., "Power requirement for accessing the H-mode in ITER",
  J. Phys.: Conf. Ser. 123, 012033 (2008).

## Contact
- **Jalal Butt**
- **Kouroche Bouchiat**
- **Nathaniel Chen**: nathaniel [at] princeton [dot] edu

## Tables

Inventory row: H-Mode; interval lexicon id: `hmode`.

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

- `format/high_confinement_mode_format_2026_v1.csv`
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
pixi run -e labelmaker python data/events/high_confinement_mode/formatter.py
```

## Category

| ID | Label |
| --- | --- |
| 0 | High confinement mode absent within another explicitly labelled regime |
| 1 | High confinement mode present (H, QH, or WP) |

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
