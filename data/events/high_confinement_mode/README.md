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
No H-mode INTERVAL label exists yet. What exists is the transition detector
`dalpha_lh` (`labelmaker.events.heuristics.lh_transitions`): a channel-median
D-alpha fall of >= 30% within 5 ms that is still >= 30% down 20-50 ms later, with the
line-averaged density up over the same window and NBI power > 500 kW just before it.
The way back out is written as `hl_transition`. An H-mode interval labeller would run
from an `lh_transition` to the next `hl_transition` or the end of the flat-top and
needs its own lexicon id; it is not started.

## Provenance
No reference dataset. The inventory notes prior work by Kouroche and
Azarakhsh Jalalvand on regime classification that is hard to access. `raw/` is empty.

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
- **Nathaniel Chen**: nathaniel [at] princeton [dot] edu

## Tables

Inventory row: H-Mode; interval lexicon id pending (existing `lh` names the L-H transition).

The scope inventory is [`discrete_labels.csv`](../discrete_labels.csv).
`raw/` holds the untouched provided lists; `format/` holds their
common-schema CSVs and metadata. Each `extend_<model>/` holds one
producing source's output on the project shot list. Categories without
a producer have no `extend_*` directory. See the [table guide](../README.md).

Regenerate registered raw tables from the repository root:

```bash
PYTHONPATH=src python scripts/labelmaker/labels_format.py
```

No raw table is registered for this category yet.
