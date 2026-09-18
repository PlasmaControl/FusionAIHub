# Poloidal Beta

## Description
The poloidal beta is the plasma pressure normalised to the poloidal magnetic
pressure,

    beta_p = 2 * mu_0 * <p> / <B_p>^2

with <B_p> the flux-surface-averaged poloidal field at the boundary (often written
with B_p = mu_0 I_p / l_p, l_p the poloidal perimeter). It measures how much of the
confinement is provided by the plasma current; beta_p ~ 1 marks the transition to
a large Shafranov shift and a high bootstrap fraction (f_bs ~ 0.5 * eps^0.5 *
beta_p). It is complementary to beta_N = beta_T [%] a B_T / I_p, which is the
stability-relevant normalisation. On DIII-D beta_p is an EFIT output (`betap`).

## Method
Not started. No threshold categories have been defined; the features store carries
`betan` but not yet `betap`, so the first step is to resolve the EFIT `betap`
scalar (archive, else fdp) and then choose bands, gated on the Ip flat-top like
`qmin_rule`.

## Provenance
None yet. `raw/` is empty.

## Models
**stable**: none

**latest**: none

**all**:
- none

## Alias
- poloidal beta
- beta_p
- betap
- beta poloidal

## Reference
- J. Wesson, Tokamaks, 4th ed. (Oxford University Press, 2011), sec. 3.4.
- J. P. Freidberg, Ideal MHD (Cambridge University Press, 2014).

## Contact
- **Nathaniel Chen**: nathaniel [at] princeton [dot] edu

## Tables

Inventory row: none in `discrete_labels.csv` (37 rows); poloidal beta is not listed. Lexicon id pending producer task.

The scope inventory is [`discrete_labels.csv`](../discrete_labels.csv).
`raw/` holds the untouched provided lists; `format/` holds their
common-schema CSVs and metadata. Each `extend_<model>/` holds one
producing source's output on the project shot list. Categories without
a producer have no `extend_*` directory. See the [table guide](../README.md).

Regenerate registered raw tables from the repository root:

```bash
PYTHONPATH=src python scripts/labeler/labels_format.py
```

No raw table is registered for this category yet.

## Verification

[`verification.ipynb`](verification.ipynb) plots one shot's signals against its
saved labels and takes back corrections. The review roster is
[`shots.csv`](shots.csv). See the [table guide](../README.md) for the roster
schema.
