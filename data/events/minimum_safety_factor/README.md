# Minimum Safety Factor

## Description
The safety factor q is the number of toroidal turns a field line makes per poloidal
turn,

$$q(r) \approx \frac{r B_{\phi}}{R B_{\theta}}$$

and q_min is its minimum over the profile, set by the current-density profile
(on-axis for a monotonic profile, off-axis for reversed shear). q_min < 1 admits
the m/n = 1/1 kink and sawteeth; q_min just above 1 with low central shear is the
"hybrid" scenario (no sawteeth, 3/2 NTM tolerated); q_min in 1.5-2 avoids the 3/2
and 2/1 rational surfaces in the core and is the elevated-q_min steady-state
scenario; q_min > 2 is the high-q_min / reversed-shear regime with internal
transport barriers. On DIII-D q_min comes from the EFIT equilibrium reconstruction
(EFIT01 magnetics-only; EFIT02 with MSE constrains it far better).

## Categories
- Low
- Hybrid
- Elevated
- High

## Method
`qmin_rule` (`labeler.events.heuristics.qmin_regimes`) thresholds the canonical
`qmin` feature inside the Ip flat-top (|Ip| > 0.9 max) and writes exclusive interval
events `qmin_hybrid` (0.95 < q <= 1.5), `qmin_elevated` (1.5 < q <= 2) and
`qmin_high` (q > 2), each over a contiguous finite run lasting >= 500 ms. A sample
belongs to at most one band; a dropout splits a band. The rows carry NO confidence
(a threshold on a reconstructed scalar has no calibrated probability) and
`attrs["efit"]` names the reconstruction (EFIT01 today). Coverage is the flat-top
intersected with the finite q_min record, so a ramp is an abstention and not an
absence. The `Low` category (q_min <= 0.95, sawtoothing) is not yet emitted.

Measured on the 500 `recommender_v1` shots: hybrid 271, elevated 60, high 50 shots.
Ungated, `q > 0.95` fires on 497 of 500 because every current ramp passes through
every band.

## Provenance
Derived from EFIT01 q_min in the labelmaker features store (archive, else fdp);
no curated table, so `raw/` is empty. The inventory asks for a higher-fidelity
equilibrium (EFIT02 or CAKE) when available.

## Models
**stable**: none

**latest**: none

**all**:
- qmin_rule | 2026_09_13 (rule; EFIT01; 500 ms minimum duration)

## Alias
- qmin
- q-min
- minimum safety factor
- q_min
- hybrid scenario
- elevated qmin
- high qmin
- reversed shear

## Reference
- M. R. Wade et al., "Development, physics basis and performance projections for
  hybrid scenario operation in ITER on DIII-D", Nucl. Fusion 45, 407 (2005).
- C. T. Holcomb et al., "Steady state scenario development with elevated minimum
  safety factor on DIII-D", Nucl. Fusion 54, 093009 (2014).
- J. Wesson, Tokamaks, 4th ed. (Oxford University Press, 2011), ch. 3.

## Contact
- **Nathaniel Chen**: nathaniel [at] princeton [dot] edu
- **Kei Yasoda**: keiyasoda [at] princeton [dot] edu

## Tables

Inventory row: High Q-Min; lexicon ids: qmin_hybrid, qmin_elevated, qmin_high.

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

## Sampled integer-label example

[`example.ipynb`](example.ipynb) reads the existing `extend_qmin_rule/` export:
465 interval rows and 500 per-shot sparse grids. Every grid has 20 rho bins;
scalar regime IDs are repeated across all bins. Class IDs are 1 (low), 2 (hybrid), 3 (elevated), and 4 (high).
The current rule does not emit low-q intervals, so unclassified cells remain
unknown rather than being assigned class 0. The interval CSV contains
only `shot,category,t_start,t_end,confidence`; matching class IDs and radial values live in the
per-shot files. See the [storage guide](../README.md).


## Category

The CSV `category` column and grid values use integer IDs. The same mapping
is recorded in each JSON sidecar under `categories`.

| ID | Label |
| --- | --- |
| 0 | Absent (reserved) |
| 1 | Low |
| 2 | Hybrid |
| 3 | Elevated |
| 4 | High |

Unknown or unclassified grid cells are stored separately from 0. A dataset
containing only positive annotations does not establish absence elsewhere.
Sampled grids use 50 ms bins and 20 rho bins.

The notebook's last cell plots the category's original annotations alongside
the saved 50 ms grid. Original-label plots require the source files; the
formatted and extended plots continue to read only their selected NPZ files.

## Verification

[`verification.ipynb`](verification.ipynb) plots one shot's signals against its
saved labels and takes back corrections. The review roster is
[`shots.csv`](shots.csv). See the [table guide](../README.md) for the roster
schema.
