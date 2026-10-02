# Minimum Safety Factor

## Description
The safety factor q is the number of toroidal turns a field line makes per poloidal
turn,

$$q(r) \approx \frac{r B_{\phi}}{R B_{\theta}}$$

and q_min is its minimum over the profile, set by the current-density profile
(on-axis for a monotonic profile, off-axis for reversed shear). q_min < 1 allows
the q=1 surface and can support sawteeth. Raising q_min above 1.5 excludes the
q=3/2 surface; raising it above 2 excludes q=2. A minimum between 1.5 and 2
can still leave a q=2 surface. The labels below are bands of the reconstructed
scalar: q_min alone does not establish a hybrid scenario, reversed shear,
steady-state operation or an internal transport barrier.
On DIII-D q_min comes from the EFIT equilibrium reconstruction
(EFIT01 magnetics-only; EFIT02 with MSE constrains it far better).

### Categories
- Low
- Hybrid
- Elevated
- High

## Data Provenance
### Dataset 1

**Dataset File(s)**:

**Author**:

**Description**: Derived from EFIT01 q_min in the labelmaker features store (fdp); no curated table, so `raw/` is empty.

**Publications**:

## Models
**stable**: none

**latest**: none

**all**:
- qmin_rule | 2026_09_13 (rule; EFIT01; 500 ms minimum duration)
- qmin_bands (review rule; four exclusive bands; 500 ms minimum duration)

## Inputs
**qmin_rule**:
- `qmin` (EFIT01 q_min)
- `ip` (flat-top)

## Method
`qmin_rule` (`labeler.events.heuristics.qmin_regimes`) thresholds the canonical
`qmin` feature inside the Ip flat-top (|Ip| > 0.9 max) and writes exclusive interval
events `qmin_hybrid` (0.95 < q <= 1.5), `qmin_elevated` (1.5 < q <= 2) and
`qmin_high` (q > 2), each over a contiguous finite run lasting >= 500 ms. A sample
belongs to at most one band; a dropout splits a band. The rows carry NO confidence
(a threshold on a reconstructed scalar has no calibrated probability) and
`attrs["efit"]` names the reconstruction (EFIT01 today). Coverage is the flat-top
intersected with the finite q_min record, so a ramp is an abstention and not an
absence. This legacy export does not emit the `Low` category.

The review editor's `qmin_bands` method adds the low band (q_min <= 0.95) and
drafts all four exclusive bands, with 500 ms minimum duration. Its extra IDs
are 5 (uncertain) and 6 (not observable), preserving 3 as elevated. The q_min
row survives missing optional q-profile or Ip panels. See
[the editor guide](../../../docs/labeler/equilibrium_review.md).

Measured on the 500 `recommender_v1` shots: hybrid 271, elevated 60, high 50 shots.
Ungated, `q > 0.95` fires on 497 of 500 because every current ramp passes through
every band.

## Alias
qmin, q-min, minimum safety factor, q_min, hybrid scenario, elevated qmin, high qmin, reversed shear

## Future Implementations
- The inventory asks for a higher-fidelity equilibrium (EFIT02 or CAKE) when available.

## Reference
- M. R. Wade et al., "Development, physics basis and performance projections for
  hybrid scenario operation in ITER on DIII-D", Nucl. Fusion 45, 407 (2005).
- C. T. Holcomb et al., "Steady state scenario development with elevated minimum
  safety factor on DIII-D", Nucl. Fusion 54, 093009 (2014).
- J. Wesson, Tokamaks, 4th ed. (Oxford University Press, 2011), ch. 3.

## Contact
- **Nathaniel Chen**: nathaniel [at] princeton [dot] edu
- **Kei Yasoda**: keiyasoda [at] princeton [dot] edu
