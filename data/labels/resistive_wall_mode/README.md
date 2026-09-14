# Resistive Wall Mode

## Description
Temporary description. Something current induced in wall. write small equation.

First discovered ...

Typically found via ...

## Method
Using reference dataset, a thing was done and stuff.

## Provenance
Obtained using reference dataset from Jeremy Hanson.

## Models
**stable**: abcd | YYYY_MM_DD

**latest**: abcd | YYYY_MM_DD

**all**:
- abcd | YYYY_MM_DD

## Alias
- resistive wall mode
- rwm

## Reference


## Contact
- **Jeremy Hanson** (original dataset): hansonjm [at] fusion [dot] gat [dot] com
- **Nathaniel Chen**: nathaniel [at] princeton [dot] edu

## Tables

Inventory row: Resistive Wall Mode; lexicon id: `rwm`.

The scope inventory is [`discrete_labels.csv`](../discrete_labels.csv).
`raw/` holds the untouched provided lists; `format/` holds their
common-schema CSVs and metadata. Each `extend_<model>/` holds one
producing source's output on the project shot list. Categories without
a producer have no `extend_*` directory. See the [table guide](../README.md).

Regenerate registered raw tables from the repository root:

```bash
PYTHONPATH=src python scripts/labelmaker/labels_format.py
```

Registered tables: `rwm_onsets_2017` (30 events, 20 shots) and
`rwm_onsets_2024` (26 events, 13 shots). The header-only
`extend_rwm/recommender_v1.csv` records a completed zero-event scan;
it does not establish absence of RWM. Its regeneration command is
documented in the [table guide](../README.md#committed-rwm-evidence).
