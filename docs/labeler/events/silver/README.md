# Silver set

> The label data is `data/events/silver/`: the `raw/`, `format/`, `review/` and `extend_*/` paths
> below are relative to it. It is not in git. `scripts/labeler/hf_publish.py pull`
> restores the label tables (`format/`, `review/`, `extend_*/`), but not `raw/`: the
> third-party originals are not in the Hugging Face release; they are on the Stellar
> checkout's disk and archived in `/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/raw_archive/`.

## Description
450 DIII-D shots to review for EVERY event, not one roster per phenomenon. The
review page's silver view walks them shot by shot, in five batches of 90. A save
there is an ordinary saved label of that event (`<event>/review/labels.csv`).
Silver is not strict yet. Each event opens on whatever exists for the shot: a
saved label, its suggestion, a format table from another source, or an extend
table.

Every label sits on one 10 ms grid: frame k is [10k, 10k + 10) ms, and
`labeler.events.grid` holds the rule. The silver export is therefore one
(event × frame) array per shot.

## Data Provenance
### Dataset 1

**Dataset File(s)**: `silver_shots.csv`, picked 2026-10-10, redone 2026-10-11 without the gold set's 50 shots (500 → 450). The picking
scripts and their inputs are kept in
`/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/silver/v1/selection/`; the sha256 of
each is in `silver_shots.meta.json`.

The columns are `shot, batch, order, pool, classes, sources, notes`:
- `classes` lists the `event:category` labels some table already gives the
  shot (present, or a named regime);
- `sources` lists the kinds of table behind them (saved, format, extend,
  suggest, logbook).

The shots come from three pools:
- **corpus** (400): shots in `/scratch/gpfs/EKOLEMEN/foundation_model`. The
  catalog cohort's 50 blind test shots and the gold set's shots are never
  included. 325 shots of the list are cohort shots.
- **ae_audited** (40): AE shots the owner saved (some also in the ECE box with
  TAE/RSAE types).
- **rwm_onsets** (10): RWM present is known only on the onset tables'
  2014–2018 shots, none of which are in the corpus.

The 50 old shots (AE and RWM) have their other events' data fetched where it
exists. Some diagnostics are missing on them.

Silver shots outside the catalog population and the old ones get span
suggestions with `python -m labeler.events.spans --windows silver`, which windows
each such shot by its own Ip, as the catalog windows a population shot.

**Selection**: a greedy pick (`pick_silver.py`). Each class adds
w · (log(1 + n + 1) − log(1 + n)) for each shot, where n is how many picked shots
already carry that class. The weight w is saved 3, format/logbook 2, extend 1.5,
suggest 1. Rare classes therefore fill first, and common ones stop counting. Being
on more event rosters breaks ties. The picks are dealt round-robin into batches,
so each 90 is balanced like the whole. Shots are ordered by number inside a batch.
The gold set (`../gold/`) was taken out first; the first pick of 500 is kept as
`silver_shots_500_old.csv` in the selection folder.

| class | shots | per batch |
|---|---|---|
| alfven_eigenmode.rsae:present | 14 | 3 / 3 / 3 / 2 / 3 |
| alfven_eigenmode.tae:present | 15 | 3 / 3 / 3 / 3 / 3 |
| alfven_eigenmode:present | 40 | 8 / 8 / 8 / 8 / 8 |
| confinement:high | 381 | 77 / 74 / 76 / 76 / 78 |
| confinement:imode | 27 | 3 / 9 / 7 / 4 / 4 |
| confinement:low | 367 | 75 / 68 / 72 / 74 / 78 |
| confinement:qh | 43 | 12 / 8 / 3 / 8 / 12 |
| confinement:wpqh | 28 | 4 / 5 / 4 / 6 / 9 |
| detachment:attached | 11 | 2 / 2 / 4 / 3 / 0 |
| detachment:detached | 6 | 1 / 2 / 3 / 0 / 0 |
| edge_localized_mode:present | 349 | 72 / 66 / 70 / 71 / 70 |
| mhd_n1.m1:present | 389 | 77 / 79 / 79 / 77 / 77 |
| mhd_n1.m2:present | 41 | 8 / 10 / 1 / 14 / 8 |
| mhd_n1.m3:present | 8 | 1 / 1 / 3 / 3 / 0 |
| mhd_n1:present | 396 | 80 / 79 / 80 / 77 / 80 |
| mhd_n2.m3:present | 17 | 4 / 5 / 2 / 0 / 6 |
| mhd_n2:present | 318 | 69 / 58 / 61 / 66 / 64 |
| mhd_n3:present | 280 | 61 / 52 / 53 / 58 / 56 |
| minimum_safety_factor:elevated | 74 | 15 / 15 / 14 / 16 / 14 |
| minimum_safety_factor:high | 52 | 9 / 11 / 13 / 11 / 8 |
| minimum_safety_factor:hybrid | 197 | 43 / 43 / 36 / 41 / 34 |
| minimum_safety_factor:low | 139 | 30 / 24 / 26 / 21 / 38 |
| poloidal_beta:present | 150 | 33 / 26 / 34 / 32 / 25 |
| resistive_wall_mode:present | 10 | 2 / 2 / 2 / 2 / 2 |
