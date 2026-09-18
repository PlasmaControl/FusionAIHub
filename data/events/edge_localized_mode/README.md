# Edge Localized Mode

## Description
Edge localized modes (ELMs) are quasi-periodic relaxations of the H-mode pedestal where the steep edge pressure gradient and the bootstrap current it drives cross the coupled peeling-ballooning stability boundary causing rapid filamentary bursts to expel stored energy in ~1 ms.

ELMs were first observed on ASDEX with the discovery of the H-mode (1982).

Type-I ELMs have frequencies of tens to a few hundred Hz on DIII-D and their frequency typically rises with heating power. Type-III ELMs are smaller,
faster and appear near the L-H power threshold. Breakthrough ELMs are large ELMs that occur during wide pedestal quiescent high confinement (WPQH) experiments.

These are typically found via filterscope D-alpha bursts, divertor Langmuir probes, magnetics and BES. Each ELM is a burst on the divertor D-alpha signal 2-5 ms wide.

## Method
The detector is `elm_clock` over the eight real D-alpha filterscope channels
(10 kHz): the column activity of the TokEye transient mask is smoothed (0.64 ms) and
peaks are picked with prominence 0.03 and a minimum separation of 3 ms
(`elmcycle.detect_elms`, ported). A candidate is accepted only if its
half-prominence width is <= 5 ms (`DALPHA_MAX_WIDTH_MS`), which rejects the broad
humps of gas puffs and L-H transitions. Each accepted burst is a point event
`elm`; the clock also writes `elm_free` intervals (rate < 5 Hz over a 100 ms window
for >= 50 ms) and the ELM rate. Coverage is the D-alpha span actually processed.

`d3d_elm_time_to_event_dsm` writes **forecasts** - the probability of an ELM within
5, 10, 20 and 50 ms - and a forecast is never reported as an observed event.

Known gap: a narrow ELM riding on a broad D-alpha hump measures wide and is dropped.

## Provenance
The original `wpqh_elm_hiro` label pickles are now in `raw/`.
The separate D-alpha clock described above remains a producer. David Smith (BES
group) is understood to hold a manual ELM label database that has not been
obtained. The DSM forecast was fitted by labelmaker on the ELM-survival rows of the
`wpqh_elm_hiro` project (629,023 rows, 60 non-BES columns), because upstream's
graphs need 64 BES channels the corpus fills on 2 of 24 sampled shots.

## Models
**stable**: d3d_elm_time_to_event_dsm | 2026_09_06 (forecast, not a detector)

**latest**: d3d_elm_time_to_event_dsm | 2026_09_06

**all**:
- d3d_elm_time_to_event_dsm | 2026_09_06 (`no_bes` fit; horizons 5/10/20/50 ms)
- elm_clock | 2026_09_13 (rule-based detector, `labeler.events.transients`)

## Alias
- edge localized mode
- edge localised mode
- elm
- elms
- elmy
- elming

## Reference
- H. Zohm, "Edge localized modes (ELMs)", Plasma Phys. Control. Fusion 38, 105
  (1996).
- A. W. Leonard, "Edge-localized-modes in tokamaks", Phys. Plasmas 21, 090501
  (2014).

## Contact
- **Nathaniel Chen**: nathaniel [at] princeton [dot] edu
- **Semin Joung**:
- **Hiro Farre Josep Kaga**:
- **Jalal Butt**:

## Tables

Inventory row: ELM; lexicon id: `elm`.

The scope inventory is [`discrete_labels.csv`](../discrete_labels.csv).
`raw/` holds the untouched provided lists; `format/` holds their
common-schema CSVs and metadata. Each `extend_<model>/` holds one
producing source's output on the project shot list. Categories without
a producer have no `extend_*` directory. See the [table guide](../README.md).

Regenerate registered raw tables from the repository root:

```bash
PYTHONPATH=src python scripts/labeler/labels_format.py
```

[`formatter.py`](formatter.py) reads the two `elm_labels_dict*.pkl` sources
registered in `../events.yaml`. Keys are shot IDs or `shot_phase`; times are
absolute milliseconds. Matching WPQH samples are deduplicated; conflicting
labels raise an error. Survival targets are not onset labels and are excluded.

Original labels sample onsets every 1 ms, after upstream burst-width
simplification (`wpqh_elm_hiro/hiro_scripts/data_processing.ipynb`). Count
positive onset samples in each half-open 50 ms bin; category is 1 when that
count is nonzero. Consecutive equal bins are compressed into intervals in
`format/edge_localized_mode_format_2026_v1.csv`. Confidence is unknown.
The final observed bin extends to its 50 ms boundary; this is an aggregation
window, not a claim about the physical ELM duration.

`format/shots/<shot>.npz` stores the 50 ms time ×
20-rho binary grid and an `event_count` array (one count per bin, not per rho).
Counts describe the supplied onset samples, not a new detector's burst count.
A bin with observed zeros is absent; bins with no samples remain unknown.
Rho is broadcast because the source has no radial localization.

[`example.ipynb`](example.ipynb) opens `format/shots/<shot>.npz` directly and plots the stored onset
counts alongside the binary grid. It can also open available extended labels.

```bash
pixi run -e labelmaker python data/events/edge_localized_mode/formatter.py
```


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
