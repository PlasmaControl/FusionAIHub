# Edge Localized Mode

## Description
Edge localized modes (ELMs) are quasi-periodic relaxations of the H-mode pedestal:
the steep edge pressure gradient and the bootstrap current it drives cross the
coupled peeling-ballooning stability boundary, a filamentary burst expels a few
percent of the stored energy and particles in ~1 ms, and the pedestal rebuilds.
Type-I ELMs (the large ones) have frequencies of tens to a few hundred Hz on
DIII-D and their frequency RISES with heating power; type-III ELMs are smaller,
faster and appear near the L-H power threshold. Each ELM is a burst on the
divertor D-alpha signal 2-5 ms wide.

First observed on ASDEX with the discovery of the H-mode (1982); the
peeling-ballooning explanation dates from the late 1990s.

Typically found via filterscope D-alpha bursts, divertor Langmuir probes, magnetics
and (on DIII-D) BES.

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
Detector: our own D-alpha clock (above); nothing in `raw/` yet. David Smith (BES
group) is understood to hold a manual ELM label database that has not been
obtained. The DSM forecast was fitted by labelmaker on the ELM-survival rows of the
`wpqh_elm_hiro` project (629,023 rows, 60 non-BES columns), because upstream's
graphs need 64 BES channels the corpus fills on 2 of 24 sampled shots.

## Models
**stable**: d3d_elm_time_to_event_dsm | 2026_09_06 (forecast, not a detector)

**latest**: d3d_elm_time_to_event_dsm | 2026_09_06

**all**:
- d3d_elm_time_to_event_dsm | 2026_09_06 (`no_bes` fit; horizons 5/10/20/50 ms)
- elm_clock | 2026_09_13 (rule-based detector, `labelmaker.events.transients`)

## Alias
- edge localized mode
- edge localised mode
- elm
- elms
- elmy
- elming
- elm-free

## Reference
- H. Zohm, "Edge localized modes (ELMs)", Plasma Phys. Control. Fusion 38, 105
  (1996).
- A. W. Leonard, "Edge-localized-modes in tokamaks", Phys. Plasmas 21, 090501
  (2014).

## Contact
- **Nathaniel Chen**: nathaniel [at] princeton [dot] edu

## Tables

Inventory row: ELM; lexicon id: `elm`.

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
