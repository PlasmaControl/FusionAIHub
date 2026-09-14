# Poloidal Beta

## Description
Pending.

## Method
Pending.

## Provenance
Pending.

## Models
**stable**: pending | YYYY_MM_DD

**latest**: pending | YYYY_MM_DD

**all**:
- pending | YYYY_MM_DD

## Alias
Pending.

## Reference
Pending.

## Contact
Pending.

## Tables

Inventory row: none in `discrete_labels.csv` (37 rows); poloidal beta is not listed. Lexicon id pending producer task.

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
