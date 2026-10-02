# Low Confinement Mode

## Description
The L-mode is the baseline confinement regime of a diverted tokamak: no edge
transport barrier, turbulent transport across the whole minor radius, flat edge
profiles and a divertor D-alpha level that is high and unstructured. Energy
confinement follows the ITER89-P scaling (tau_E ~ I_p^0.85 R^1.2 n^0.1 P^-0.5 ...)
and is roughly half of the H-mode value at the same conditions. Every DIII-D
discharge starts in L-mode; it becomes H-mode only if the heating power exceeds the
L-H threshold, and falls back at an H-L transition when power drops or at the ramp
down. L-mode is also the target regime for some heat-flux and turbulence studies.

Typically found via the absence of a D-alpha drop / pedestal, a low H98 factor and
ELM-free but noisy D-alpha.

## Data Provenance
### Dataset 1

**Dataset File(s)**: `Jalal_28042024_confinement_regime_shotlist.csv`

**Author**: Jalal Butt

**Description**: Original labels from Jalal Butt, created in 28 April 2024. Both H and L raw folders contain the same source table. Explicitly labelled `Test only` rows are used for the gold verification dataset.

**Publications**:

## Models
**stable**: none

**latest**: none

**all**:
- dalpha_lh | 2026_09_12 (rule-based L-H / H-L transition detector, not a regime label)
- d3d_confinement_bes_cnn | 2026_10_01 | AUROC: 0.945 | AUPRC: 0.783 | F1: 0.733

Scores are for L-mode as one class of the four-class BES benchmark classifier (L, H, QH, WP QH), on 2 ms windows of the 119 labelled shots that have BES in the corpus, held out by shot (8,209 L-mode windows from 38 shots; precision 0.765, recall 0.703; the paper's L-mode F1 is 0.94). Retrained here from the paper's description, because the original is not on disk. Protocol, caveats and the full table: [confinement_bes_benchmark.md](../../../docs/labeler/confinement_bes_benchmark.md).

## Inputs
**dalpha_lh**:
- `D-alpha FS01..FS08`
- `CO2 density R0` (co2 ch 0)
- `pinj_total`

**hmode_frames (round three, read as 1 - P(H))**:
- `D-alpha filterscopes` (channels pooled)
- `NBI power` (optional)

**d3d_confinement_bes_cnn**:
- `BES` inner 6 x 8 block of the 8 x 8 array (channels 8-55 of 64), 2.5-150 kHz, 1,024-sample windows (2 ms at 500 kHz)

## Method

The local formatter reads the explicit confinement intervals supplied in
`raw/Jalal_28042024_confinement_regime_shotlist.csv`. Category 1 means L;
category 0 means another explicitly annotated regime. Blank flags, unlabelled
rows, and gaps remain unknown. No regime is inferred from BES acquisition times
or the transition-note columns. The existing `dalpha_lh` detector remains a
separate source of transition points.

Intervals are aggregated into half-open 50 ms bins: any positive overlap makes
a bin 1. A bin spanning a transition can therefore be positive in both H and L
outputs; it means each regime occurred within that bin. Consecutive equal known
bins are compressed into CSV intervals. Exact original bounds remain in raw/.

Benchmark: the BES classifier of the published confinement-regime paper, rebuilt from its description, is scored against the merged confinement intervals (L, H, QH and WP QH; [shared confinement labels](../confinement/README.md)) with `scripts/labeler/confinement_bes_benchmark.py`; its results and the paper's per-class scores are in [confinement_bes_benchmark.md](../../../docs/labeler/confinement_bes_benchmark.md).

## Alias
l-mode, lmode, low confinement mode, ohmic (only when no auxiliary heating is applied)

## Future Implementations

## Reference
- P. N. Yushmanov et al., "Scalings for tokamak energy confinement", Nucl. Fusion
  30, 1999 (1990). (ITER89-P)
- F. Wagner, "A quarter-century of H-mode studies", Plasma Phys. Control. Fusion 49,
  B1 (2007).

## Contact
- **Jalal Butt**
- **Kouroche Bouchiat**
- **Nathaniel Chen**: nathaniel [at] princeton [dot] edu
