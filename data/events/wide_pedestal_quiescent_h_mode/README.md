# Wide Pedestal Quiescent H-Mode

## Description
Quiescent H-mode (QH-mode) is an ELM-free H-mode in which the edge harmonic
oscillation (EHO) - a saturated, low-n (n = 1-5) kink-peeling mode at 2-20 kHz with
a comb of harmonics - provides the continuous particle and impurity exhaust that
ELMs would otherwise supply; it is accessed at high edge rotation shear, historically
with counter-NBI torque. The WIDE-PEDESTAL QH-mode (WPQH) is the bifurcation found on
DIII-D in 2015-2016 when the injected torque is brought to near zero: the EHO is
replaced by broadband (~50-200 kHz) edge MHD turbulence, the pedestal width grows by
~50-100%, pedestal pressure and confinement rise (H98 ~ 1.3-1.5), and the regime is
stationary and ELM-free at ITER-relevant low torque.

First reported by Burrell et al. (2016) and analysed in Chen et al. (2017).

Typically found via ELM-free D-alpha with broadband (not harmonic) edge magnetics,
a wide edge-Thomson pedestal, near-zero NBI torque and high density (n_ped near the
Greenwald fraction).

## Method
No WPQH label yet - the category needs its own lexicon id (the existing `qh` is
QH-mode). What exists:

- `qh_proxy` (`labelmaker.events.heuristics.qh_candidates`): QH candidate intervals
  where an EHO-like TokEye track (2-20 kHz, >= 2 harmonics) runs inside an
  `elm_free` interval, inside NBI-on, inside the Ip flat-top; confidence is the
  weaker of the track's and the ELM-free fraction. Because WPQH REPLACES the EHO
  with broadband MHD, this proxy will miss WPQH by construction.
- A WPQH rule would be: `elm_free` inside the flat-top, NO EHO track, low |tinj|,
  and a broadband transient / coherent-mode signature in 50-200 kHz on mhr.

## Provenance
The inventory says to use Azarakhsh Jalalvand's previous WPQH database; it has not
yet been placed in `raw/`. The related `QH_Database.csv` (134 rows, 107 shots,
173694-175544; 0 in `recommender_v1`) is wired into the `qh` registry entry as the
only human-curated QH evidence.

## Models
**stable**: none

**latest**: none

**all**:
- qh_proxy | 2026_09_12 (QH-mode proxy from EHO tracks; not WPQH)

## Alias
- wpqh
- wpqh-mode
- wide pedestal qh
- wide-pedestal quiescent h-mode
- wide pedestal quiescent h-mode
- qh-mode
- quiescent h-mode
- eho (the QH-mode marker WPQH lacks)

## Reference
- K. H. Burrell et al., "Discovery of stationary operation of quiescent H-mode
  plasmas with net-zero neutral beam injection torque and high energy confinement
  on DIII-D", Phys. Plasmas 23, 056103 (2016).
- Xi Chen et al., "Bifurcation of quiescent H-mode to a wide pedestal regime in
  DIII-D and advances in the understanding of edge harmonic oscillations",
  Nucl. Fusion 57, 086008 (2017).
- K. H. Burrell et al., "Quiescent H-mode plasmas in the DIII-D tokamak",
  Phys. Plasmas 8, 2153 (2001).

## Contact
- **Azarakhsh Jalalvand**: aj17 [at] princeton [dot] edu
- **Nathaniel Chen**: nathaniel [at] princeton [dot] edu

## Tables

Inventory row: WPQH-Mode; specific lexicon id pending (existing `qh` names QH-Mode).

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
