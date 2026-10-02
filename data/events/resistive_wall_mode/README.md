# Resistive Wall Mode

## Description
The resistive wall mode (RWM) is the external kink (mostly n = 1) that a perfectly
conducting wall would stabilise but a real, resistive wall only slows: the mode
grows on the wall's flux-penetration time tau_w (milliseconds on DIII-D) instead of
the Alfvén time. It exists in the window between the no-wall and ideal-wall beta
limits,

    beta_N^no-wall  <  beta_N  <  beta_N^ideal-wall

(empirically beta_N^no-wall ~ 4 l_i on DIII-D), and is stabilised by plasma rotation
and kinetic resonances or by active feedback with the C- and I-coils. Its onset
often ends a high-beta_N discharge in a disruption; error-field amplification
(resonant field amplification, RFA) is its marginally stable precursor.

First identified in the mid 1990s in DIII-D wall-stabilised high-beta experiments
and on the HBT-EP and PBX-M devices.

Typically found via low-frequency (0-10 kHz) n = 1 magnetics - saddle loops and
poloidal probe arrays - as a slowly growing, slowly rotating or locked mode near the
no-wall limit.

## Data Provenance
### Dataset 1

**Dataset File(s)**: `rwm_onsets_2017.csv`, `rwm_onsets_2024.csv`

**Author**: Jeremy Hanson

**Description**: Reference dataset from Jeremy Hanson: `rwm_onsets_2017.csv` (20 shots, 156785-158023) and `rwm_onsets_2024.csv` (13 shots, 176067-176092), 56 onsets over 33 distinct shots in `raw/`.

**Publications**:

## Models
**stable**: none

**latest**: none

**all**:
- rwm_candidates (uncertain review screen; uncalibrated)

## Inputs
- Curated onset times from the two database tables.
- `rwm_candidates`: `n1rms`, `betan`, `li`, and `ip`.
- Review rows additionally show `n2rms` where available.

## Method
No calibrated detector writes `rwm`. The two curated onset tables are the only
confirmed evidence; each
row becomes a point event with `evidence_kind = database`,
`source = database:rwm_onsets_<year>`, NaN confidence and NaN coverage - a listing
says nothing about the interval anybody examined, so an absent shot is never a
negative. `NTOR` (toroidal mode number) and `MODE_TYPE` travel in `attrs`. The
`extend_rwm/recommender_v1.csv` scan over the 500 project shots is header-only:
none of the 33 listed shots is in the corpus, and that is a completed zero, not an
absence claim.

The review editor shows the original database onsets and a conservative
`rwm_candidates` screen: an n=1 magnetic RMS excursion above the flat-top median
plus six MAD, beta_N > 4 l_i, and at least 10 ms duration. A screen candidate is
uncertain; other measured time is unassessed, and missing inputs are not
observable. N1RMS does not distinguish RWM from tearing or applied-field response.
No onset is expanded into an RWM duration, and no unlisted shot becomes absent.
The roster includes the database's 33 reference shots alongside the cohort.
See [the editor guide](../../../docs/labeler/equilibrium_review.md).

## Alias
resistive wall mode, rwm, resonant field amplification (precursor), rfa

## Future Implementations

## Reference
- M. S. Chu and M. Okabayashi, "Stabilization of the external kink and the
  resistive wall mode", Plasma Phys. Control. Fusion 52, 123001 (2010).
- A. M. Garofalo et al., "Sustained stabilization of the resistive-wall mode by
  plasma rotation in the DIII-D tokamak", Phys. Rev. Lett. 89, 235001 (2002).

## Contact
- **Jeremy Hanson** (original dataset): hansonjm [at] fusion [dot] gat [dot] com
- **Nathaniel Chen**: nathaniel [at] princeton [dot] edu
