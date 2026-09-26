# Event Catalog

The shared tables of the DIII-D event catalog (v1): 500 shots from 2021-2025,
each assessed for six phenomena - Alfvén eigenmodes, neoclassical tearing
modes, H-mode, ELMs, sawteeth and disruptions. A phenomenon's own labels stay in
its category directory (`../<category>/review/`); what belongs to the catalog
as a whole is here.

This directory is not an event category. `events.yaml` lists it under
`non_category_dirs`, so category scans (`labeler.events.databases.category_dirs`)
skip it.

## Tables

| File | Written by | Contents |
| --- | --- | --- |
| `cohort.csv` | `python -m labeler.events.catalog.cohort` | The drawn shots: group, cell, weight, split, review-queue rank and assessed window |
| `cohort_manifest.yaml` | the same command | The seed, rules, allocation, N and n per cell, the rejection counts and the checksums of every input |
| `papers.csv` | `python -m labeler.literature.osti links` | One row per verified shot-paper link |
| `cards/<method>@<version>.json` | the scoring library, later | A method's scores against the blind reference |

Each file is written under `$LABELER_ROOT` first (`catalog/` for the cohort,
`literature/osti/` for the links) and copied here only with the owner's
go-ahead. The cohort is drawn once, from inputs frozen beforehand, and is never
redrawn in place.
