# Shared confinement labels

Untouched labels attributed to **Kevin Gill** (the XLSX workbook and BES-time
HDF5 files) and **Jalal Butt** (the April 2024 CSV). This directory is a shared
source collection, not an additional catalog category (`events.yaml` lists it
under `non_category_dirs`). L/H/QH/WP remain distinct in the reconciled interval
table; the binary detector treats H/QH/WP as high confinement.

## Review

The review page offers confinement as one event, `confinement`, in place of
`high_confinement_mode`, `low_confinement_mode`,
`quiescent_high_confinement_mode` and
`wide_pedestal_quiescent_high_confinement_mode` (it neither lists nor opens
those four; their tables and rosters are untouched, and the frame models and the
paper still read them). A span is 1 high, 2 low, 3 qh, 4 wpqh or 5 uncertain;
time left unmarked is category 0 (absent), and nothing in it is observable.

- `shots.csv` is the roster: the H-mode roster's 450 shots, in its order, then
  389 curated shots outside it (unverified, not held out). They are the shots of
  the merged interval table below that have a D-alpha record, in the corpus or in
  the raw cache (the cache holds the rest, fetched while logged into fdp). Not on
  it: 2 blind shots, and 45 shots without a D-alpha record (6 have none to fetch,
  174601, 174602, 175658, 179212, 179216 and 179314; the other 39 were not reached
  before the fetch hit its time limit).
- `review/source.json` points at the suggestion an unsaved shot opens on,
  `suggestions/regimes/v1/confinement_suggest_regimes_v1.csv` under
  `$LABELER_ROOT`, built once by hand from two sources:
  - the curated intervals, `runs/labeler/confinement/v1/merged_intervals.csv`:
    H is 1 high, L is 2 low, QH is 3 qh, WP is 4 wpqh, and the one H|L conflict is
    5 uncertain. This is what the 401 shots that have any (the 12 on the 450 and
    the 389 added) open on, tiled over the shot's window with the rest unmarked.
    A shot outside the cohort has the population's window when it has one, else 0
    to 500 ms past its last interval, as the resistive-wall-mode reference shots
    do;
  - for every other shot, the H-mode D-alpha table renumbered
    (`suggestions/dalpha_lh/v1/confinement_suggest_dalpha_lh_v1.csv`): high stays
    1, uncertain becomes 5, and the table's no-H-mode and not-observable stretches
    are left unmarked, because the detector never established L-mode. Only
    curated shots show low, qh or wpqh.

  Its `.meta.json` records both sources' sha256 and the mapping.
- `review/labels.csv` and `review/history.jsonl` are written by the page when a
  shot is saved.
- The panels are the H-mode event's (D-alpha, density, NBI, beta_N). They are
  built once under `spectrograms/high_confinement_mode/` and shared
  (`labeler.config.STORE_EVENTS`).

From the repository root:

```bash
pixi run --frozen -e labelmaker python -m labeler.confinement labels
```

Writes the exact source union, conflicts, conservative 50 ms targets, shot splits,
class/shot-balanced training weights and source hashes under
`runs/labeler/confinement/v1`. Unknown samples and incomplete bins are excluded;
held-out class prevalence is preserved. Raw `Test only` assignments apply across
all sources for a shot. Raw files and corpus stores are read-only.

The workbook's labelled intervals exactly match entries in Jalal's CSV;
matching files must not be treated as independent expert annotations. See the
[analysis](../../../docs/labeler/confinement_analysis.md) for measured overlap,
the within-source conflict, source completeness and scientific review records.

The [executed notebook](example.ipynb) includes source statistics, ELM consistency,
diagnostic eligibility and frozen detector results. The
[model card](../../../docs/labeler/confinement_model_card.md) documents training,
no-BES inference, the measured limits and commands. Merged complementary H/L
exports are under `runs/labeler/confinement/v1/format/`; trained models and
their evaluation are in that same run directory. The BES classifier that serves
as the benchmark for these labels is in
[confinement_bes_benchmark.md](../../../docs/labeler/confinement_bes_benchmark.md)
with its published per-class scores. The original is not on disk, so it was rebuilt
from the paper's description and trained here on the 119 labelled shots that have BES
in the corpus: macro F1 0.68 on held-out shots (the paper's is 0.94), 0.93 when
windows of the same shots are split instead.
