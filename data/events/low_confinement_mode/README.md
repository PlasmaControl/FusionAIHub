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
No L-mode INTERVAL label exists yet. `dalpha_lh` (`labelmaker.events.heuristics`)
writes the `lh_transition` and `hl_transition` points; an L-mode interval is the
complement of the H-mode intervals inside the Ip flat-top, from the flat-top start
to the first `lh_transition` and from each `hl_transition` to the next
`lh_transition`. That labeller and its lexicon id are not started, and a shot with
no NBI trace is one the transition detector cannot speak about.

## Provenance
No reference dataset. The inventory notes prior work by Kouroche and
Azarakhsh Jalalvand that is hard to access. `raw/` is empty.

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
- **Nathaniel Chen**: nathaniel [at] princeton [dot] edu

## Tables

Inventory row: L-Mode; interval lexicon id pending (existing `lh` names the L-H transition).

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
