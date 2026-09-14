# Minimum Safety Factor

## Description
The safety factor q is the number of toroidal turns a field line makes per poloidal
turn,

    q(r) ~= r * B_phi / (R * B_theta)

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
`qmin_rule` (`labelmaker.events.heuristics.qmin_regimes`) thresholds the canonical
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

## Tables

Inventory row: High Q-Min; lexicon ids pending q-min producer task.

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
