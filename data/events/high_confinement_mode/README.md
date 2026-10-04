# High Confinement Mode

## Description
The H-mode is the tokamak confinement regime with an edge transport barrier: a
narrow region inside the separatrix where the E x B shear suppresses turbulence, so
density and temperature form a steep pedestal and global energy confinement is
roughly twice the L-mode value (H98(y,2) ~ 1). Access needs the heating power to
exceed a threshold P_LH that scales roughly as n_e^0.7 B_T^0.8 S^0.9 (Martin 2008)
and is lower with the ion grad-B drift toward the X-point. The transition is abrupt:
the divertor D-alpha drops in a few ms while the density rises. Most H-modes carry
ELMs (see `edge_localized_mode`); QH-mode and WPQH are ELM-free variants.

First discovered on ASDEX in 1982 (Wagner et al.).

Typically found via the D-alpha drop at the transition, the pedestal in edge
Thomson profiles, the density rise and the confinement factor.

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
- confine-cnn | 2026_10_03 | AUROC: 0.925 | AUPRC: 0.940 | F1: 0.858
- confine-cnn | 2026_10_01 | AUROC: 0.948 | AUPRC: 0.979 | F1: 0.943 (first retrain; read 2 blind test shots)

Scores are for H-mode as one class of the four-class BES benchmark classifier (L, H, QH, WP QH), `confine-cnn`, rebuilt from the paper's description (its code is not on disk) and retrained on our labels. The 2026_10_03 line is our reimplementation under the paper's selection and split: the paper's beam gate, margins, optimiser, architecture and training length, a 6 x 8 block chosen from the channel positions, native 1 MHz BES (1,024-sample windows, 1 ms), 401 shots, five random splits by shot with the test windows pooled (the splits overlap, so the test windows are pooled once per split that tests them: 142 distinct test shots in 200 shot-tests and 106,232 distinct windows of 152,863 pooled over all classes, and one split scores macro F1 0.684 +/- 0.074 against the pooled 0.703): 79,557 pooled H-mode windows (58,748 distinct) on 69 distinct test shots, precision 0.841, recall 0.875, F1 0.858 [0.78, 0.91] (shot bootstrap); the paper's H-mode F1 is 0.97 on its own 44-shot test set. The 2026_10_01 line is the first retrain with none of that protocol, on 2 ms windows of the 119 labelled shots that have BES in the corpus, held out by shot (47,373 H-mode windows from 75 shots; precision 0.946, recall 0.939; the paper's H-mode F1 is 0.97); its two blind test shots are read, and it is not a benchmark number. Protocol, caveats and the full table: [confinement_bes_benchmark.md](../../../docs/labeler/confinement_bes_benchmark.md).

## Inputs
**dalpha_lh**:
- `D-alpha FS01..FS08`
- `CO2 density R0` (co2 ch 0)
- `pinj_total`

**hmode_frames (round three)**:
- `D-alpha filterscopes` (channels pooled)
- `NBI power` (optional)

**confine-cnn**:
- `BES` 6 x 8 block of the 8 x 8 array, 2.5-150 kHz, 1,024-sample windows. 2026_10_03: the block chosen per shot from the channel positions (rows 0-5 for 349 of 401 shots, rows 1-6 for 37, rows 2-7 for 15), 1 MHz (1 ms). 2026_10_01: channels 8-55 of 64 (rows 1-6), 500 kHz (2 ms)

## Method

The local formatter reads the explicit confinement intervals supplied in
`raw/Jalal_28042024_confinement_regime_shotlist.csv`. Category 1 means H, QH, or WP;
category 0 means another explicitly annotated regime. Blank flags, unlabelled
rows, and gaps remain unknown. No regime is inferred from BES acquisition times
or the transition-note columns. The existing `dalpha_lh` detector remains a
separate source of transition points.

Intervals are aggregated into half-open 50 ms bins: any positive overlap makes
a bin 1. A bin spanning a transition can therefore be positive in both H and L
outputs; it means each regime occurred within that bin. Consecutive equal known
bins are compressed into CSV intervals. Exact original bounds remain in raw/.

Benchmark: the BES classifier of the published confinement-regime paper, rebuilt from its description, is scored against the merged confinement intervals (L, H, QH and WP QH; [shared confinement labels](../confinement/README.md)) with `scripts/labeler/confinement_bes_benchmark.py` (first retrain) and `scripts/labeler/confinement_bes_ablation.py` (protocol-matched rows); its results and the paper's per-class scores are in [confinement_bes_benchmark.md](../../../docs/labeler/confinement_bes_benchmark.md).

## Alias
h-mode, hmode, high confinement mode, l-h transition, h-mode transition

## Future Implementations

## Reference
- F. Wagner et al., "Regime of improved confinement and high beta in neutral-beam-
  heated divertor discharges of the ASDEX tokamak", Phys. Rev. Lett. 49, 1408 (1982).
- F. Wagner, "A quarter-century of H-mode studies", Plasma Phys. Control. Fusion 49,
  B1 (2007).
- Y. R. Martin et al., "Power requirement for accessing the H-mode in ITER",
  J. Phys.: Conf. Ser. 123, 012033 (2008).

## Contact
- **Jalal Butt**
- **Kouroche Bouchiat**
- **Nathaniel Chen**: nathaniel [at] princeton [dot] edu
