# Poloidal Beta

## Description
The poloidal beta is the plasma pressure normalised to the poloidal magnetic
pressure,

    beta_p = 2 * mu_0 * <p> / <B_p>^2

with <B_p> the flux-surface-averaged poloidal field at the boundary (often written
with B_p = mu_0 I_p / l_p, l_p the poloidal perimeter). It compares plasma pressure
with the poloidal magnetic pressure and helps characterize pressure-driven
equilibrium shifts and bootstrap current. Neither effect has a universal
threshold at beta_p = 1; shaping, profiles and aspect ratio matter.
It is complementary to beta_N = beta_T [%] a B_T / I_p, which is the
stability-relevant normalisation. On DIII-D beta_p is an EFIT output (`betap`).

## Data Provenance
### Dataset 1

**Dataset File(s)**:

**Author**:

**Description**: EFIT01 `betap` and the Ip gate come from the canonical local signals. Draft sidecars retain per-input resolver, locator and clock metadata. No curated annotations are available; `raw/` is empty.

**Publications**:

## Models
**stable**: none

**latest**: none

**all**:
- betap_rule (review rule; beta_p > 1; 500 ms minimum duration)

## Inputs
- `betap` (dimensionless EFIT01 poloidal beta; archive, else fdp)
- `ip` (flat-top gate)

## Method
`betap_rule` drafts `beta_p > 1` intervals lasting at least 500 ms inside the
longest Ip flat-top. Short excursions are uncertain; finite values <= 1 are
absent. Gaps and time outside the gate are not observable. This threshold is a
review convention, not a validated advanced-tokamak scenario definition.
The beta_p and Ip rows can be reviewed and saved in the same page as ELMs and
sawtooth. See [the editor guide](../../../docs/labeler/equilibrium_review.md).

## Alias
poloidal beta, beta_p, betap, beta poloidal

## Future Implementations

## Reference
- J. Wesson, Tokamaks, 4th ed. (Oxford University Press, 2011), sec. 3.4.
- J. P. Freidberg, Ideal MHD (Cambridge University Press, 2014).

## Contact
- **Nathaniel Chen**: nathaniel [at] princeton [dot] edu
