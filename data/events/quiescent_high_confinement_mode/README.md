# Quiescent High Confinement Mode

## Description
No description, method, provenance or table has been written for this
category yet. The scope inventory row is in
[`discrete_labels.csv`](../discrete_labels.csv).

## Data Provenance
### Dataset 1

**Dataset File(s)**:

**Author**:

**Description**:

**Publications**:

## Models
**stable**: none

**latest**: none

**all**:
- d3d_confinement_bes_cnn | 2026_10_01 | AUROC: 0.942 | AUPRC: 0.496 | F1: 0.564

Scores are for QH-mode as one class of the four-class BES benchmark classifier (L, H, QH, WP QH), on 2 ms windows of the 119 labelled shots that have BES in the corpus, held out by shot (5,608 QH-mode windows from 17 shots; precision 0.505, recall 0.639; the paper's QH-mode F1 is 0.94). Most of the errors are against WP QH. Retrained here from the paper's description, because the original is not on disk. Protocol, caveats and the full table: [confinement_bes_benchmark.md](../../../docs/labeler/confinement_bes_benchmark.md).

## Inputs
**d3d_confinement_bes_cnn**:
- `BES` inner 6 x 8 block of the 8 x 8 array (channels 8-55 of 64), 2.5-150 kHz, 1,024-sample windows (2 ms at 500 kHz)

## Method

Benchmark: the BES classifier of the published confinement-regime paper, rebuilt from its description, is scored against the merged confinement intervals (L, H, QH and WP QH; [shared confinement labels](../confinement/README.md)) with `scripts/labeler/confinement_bes_benchmark.py`; its results and the paper's per-class scores are in [confinement_bes_benchmark.md](../../../docs/labeler/confinement_bes_benchmark.md). There is no QH detector of ours yet.

## Alias
qh, qh-mode, qh mode, quiescent h-mode, quiescent h mode

## Future Implementations

## Reference

## Contact
- **Nathaniel Chen**: nathaniel [at] princeton [dot] edu
