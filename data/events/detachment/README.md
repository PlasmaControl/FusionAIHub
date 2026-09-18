# Detachment

## Description
Divertor detachment is the state in which the plasma at the divertor target has cooled to a few eV (T_e,target <~ 5 eV) so that volumetric losses, including radiation, charge exchange, recombination, dissipate most of the parallel heat and particle flux before it reaches the plate. The signature is a roll-over: as upstream density rises, the target ion saturation current and heat flux first grow and then FALL, the pressure along the field line is no longer conserved (p_target << p_upstream), and the radiation front moves from the target toward the X-point. Partial (outer strike point only) and full detachment are distinguished; a MARFE is the extreme case where the front moves onto the confined plasma edge.

First studied systematically in the 1990s (JET Mark I, DIII-D, ASDEX Upgrade) as
the route to tolerable divertor heat loads.

Typically found via divertor Langmuir probes (j_sat roll-over), divertor Thomson
scattering (T_e), bolometry (radiation front) and, visible divertor cameras.

## Method
Not started. No detector writes `detachment`; the only route today is the operator
logbook (`text_mentions`, which is a word somebody wrote and never an observation).
Candidate methods from the inventory: a Langmuir-probe j_sat roll-over rule, or the
tangential TV route from the adjacent plasma-TV project. Several existing algorithms
are restricted and may have to be re-created.

## Provenance
No reference dataset yet. Cheolsik Byun (contact) has worked on detachment
algorithms; `raw/` is empty until a table arrives.

## Models
**stable**: none

**latest**: none

**all**:
- none

## Alias
- detachment
- detach
- detached
- divertor detachment

## Reference
- S. I. Krasheninnikov and A. S. Kukushkin, "Physics of ultimate detachment of a
  tokamak divertor plasma", J. Plasma Phys. 83, 155830501 (2017).
- A. Loarte et al., "Plasma detachment in JET Mark I divertor experiments",
  Nucl. Fusion 38, 331 (1998).

## Contact
- **Cheolsik Byun**: csbyun [at] princeton [dot] edu
- **Nathaniel Chen**: nathaniel [at] princeton [dot] edu

## Verification

[`verification.ipynb`](verification.ipynb) plots one shot's signals against its
saved labels and takes back corrections. The review roster is
[`shots.csv`](shots.csv). See the [table guide](../README.md) for the roster
schema.
