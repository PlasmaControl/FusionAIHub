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
- confine-cnn | 2026_10_03 | AUROC: 0.912 | AUPRC: 0.781 | F1: 0.766
- confine-cnn | 2026_10_01 | AUROC: 0.942 | AUPRC: 0.496 | F1: 0.564 (first retrain; read 2 blind test shots)

Scores are for QH-mode as one class of the four-class BES benchmark classifier (L, H, QH, WP QH), `confine-cnn`, rebuilt from the paper's description (its code is not on disk) and retrained on our labels. The 2026_10_03 line is our reimplementation under the paper's selection and split: the paper's beam gate, margins, optimiser, architecture and training length, a 6 x 8 block chosen from the channel positions, native 1 MHz BES (1,024-sample windows, 1 ms), 401 shots, five random splits by shot with the test windows pooled (the splits overlap, so the test windows are pooled once per split that tests them: 142 distinct test shots in 200 shot-tests and 106,232 distinct windows of 152,863 pooled over all classes, and one split scores macro F1 0.684 +/- 0.074 against the pooled 0.703): 44,061 pooled QH-mode windows (29,399 distinct) on 59 distinct test shots, precision 0.744, recall 0.790, F1 0.766 [0.66, 0.85] (shot bootstrap); the paper's QH-mode F1 is 0.94 on its own 44-shot test set. The 2026_10_01 line is the first retrain with none of that protocol, on 2 ms windows of the 119 labelled shots that have BES in the corpus, held out by shot (5,608 QH-mode windows from 17 shots; precision 0.505, recall 0.639; the paper's QH-mode F1 is 0.94); its two blind test shots are read, and it is not a benchmark number. Most of the first retrain's errors were against WP QH. Protocol, caveats and the full table: [confinement_bes_benchmark.md](../../../docs/labeler/confinement_bes_benchmark.md).

## Inputs
**confine-cnn**:
- `BES` 6 x 8 block of the 8 x 8 array, 2.5-150 kHz, 1,024-sample windows. 2026_10_03: the block chosen per shot from the channel positions (rows 0-5 for 349 of 401 shots, rows 1-6 for 37, rows 2-7 for 15), 1 MHz (1 ms). 2026_10_01: channels 8-55 of 64 (rows 1-6), 500 kHz (2 ms)

## Method

Benchmark: the BES classifier of the published confinement-regime paper, rebuilt from its description, is scored against the merged confinement intervals (L, H, QH and WP QH; [shared confinement labels](../confinement/README.md)) with `scripts/labeler/confinement_bes_benchmark.py` (first retrain) and `scripts/labeler/confinement_bes_ablation.py` (protocol-matched rows); its results and the paper's per-class scores are in [confinement_bes_benchmark.md](../../../docs/labeler/confinement_bes_benchmark.md). There is no validated QH detector of ours yet: `confine-ours` (0D signals, no BES; [confinement_ours.md](../../../docs/labeler/confinement_ours.md)) segments QH on the roster, per-bin F1 0.892 with folds that hold out run days, but is experimental and its QH label rests on the edge harmonic oscillation in the magnetic spectrogram, which it does not see.

## Alias
qh, qh-mode, qh mode, quiescent h-mode, quiescent h mode

## Future Implementations

## Reference

## Contact
- **Nathaniel Chen**: nathaniel [at] princeton [dot] edu
