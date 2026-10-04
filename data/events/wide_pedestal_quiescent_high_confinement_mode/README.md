# Wide Pedestal Quiescent H-Mode

## Description
Quiescent H-mode (QH-mode) is an ELM-free H-mode in which the edge harmonic
oscillation (EHO) - a saturated, low-n (n = 1-5) kink-peeling mode at 2-20 kHz with
a comb of harmonics - provides the continuous particle and impurity exhaust that
ELMs would otherwise supply; it is accessed at high edge rotation shear, historically
with counter-NBI torque. The WIDE-PEDESTAL QH-mode (WPQH) is the bifurcation found on
DIII-D in 2015-2016 when the injected torque is brought to near zero: the EHO is
replaced by broadband (~50-200 kHz) edge MHD turbulence, the pedestal width grows by
~50-100%, pedestal pressure and confinement rise (H98 ~ 1.3-1.5), and the regime is
stationary and ELM-free at ITER-relevant low torque.

First reported by Burrell et al. (2016) and analysed in Chen et al. (2017).

Typically found via ELM-free D-alpha with broadband (not harmonic) edge magnetics,
a wide edge-Thomson pedestal, near-zero NBI torque and high density (n_ped near the
Greenwald fraction).

## Data Provenance
### Dataset 1

**Dataset File(s)**:

**Author**: Azarakhsh Jalalvand

**Description**: The inventory says to use Azarakhsh Jalalvand's previous WPQH database; it has not yet been placed in `raw/`.

**Publications**:

### Dataset 2

**Dataset File(s)**: `QH_Database.csv`

**Author**:

**Description**: The related `QH_Database.csv` (134 rows, 107 shots, 173694-175544; 0 in `recommender_v1`) is wired into the `qh` registry entry as the only human-curated QH evidence.

**Publications**:

### Dataset 3

**Dataset File(s)**: `wpqh_on_off_times.pkl`

**Author**:

**Description**: Copied unchanged from `/projects/EKOLEMEN/wpqh_elm_hiro/hiro_scripts/data/wpqh_on_off_times.pkl`; also hard-linked in `edge_localized_mode/raw/`.

**Publications**:

### Dataset 4

**Dataset File(s)**: `WPQHphases-tau_min500-tau_inter250-pewid_max4.0-pewid_min3.0-tinj_wpqh2.1`

**Author**:

**Description**: Original phase-table directory copied unchanged from `/projects/EKOLEMEN/wpqh_elm_hiro/hiro_scripts/`; attribution to the requested Jalalvand database remains unconfirmed.

**Publications**:

## Models
**stable**: none

**latest**: none

**all**:
- qh_proxy | 2026_09_12 (QH-mode proxy from EHO tracks; not WPQH)
- confine-cnn | 2026_10_03 | AUROC: 0.785 | AUPRC: 0.324 | F1: 0.391
- confine-cnn | 2026_10_01 | AUROC: 0.870 | AUPRC: 0.546 | F1: 0.473 (first retrain; read 2 blind test shots)

Scores are for WP QH-mode as one class of the four-class BES benchmark classifier (L, H, QH, WP QH), `confine-cnn`, rebuilt from the paper's description (its code is not on disk) and retrained on our labels. The 2026_10_03 line is our reimplementation under the paper's selection and split: the paper's beam gate, margins, optimiser, architecture and training length, a 6 x 8 block chosen from the channel positions, native 1 MHz BES (1,024-sample windows, 1 ms), 401 shots, five random splits by shot with the test windows pooled (the splits overlap, so the test windows are pooled once per split that tests them: 142 distinct test shots in 200 shot-tests and 106,232 distinct windows of 152,863 pooled over all classes, and one split scores macro F1 0.684 +/- 0.074 against the pooled 0.703): 5,552 pooled WP QH-mode windows (3,496 distinct) on 7 distinct test shots, precision 0.658, recall 0.278, F1 0.391 [0.00, 0.69] (shot bootstrap); the paper's WP QH-mode F1 is 0.90 on its own 44-shot test set. The 2026_10_01 line is the first retrain with none of that protocol, on 2 ms windows of the 119 labelled shots that have BES in the corpus, held out by shot (5,342 WP QH windows from 20 shots; precision 0.495, recall 0.452; the paper's WP QH F1 is 0.90); its two blind test shots are read, and it is not a benchmark number. Most of the first retrain's errors were against QH. Protocol, caveats and the full table: [confinement_bes_benchmark.md](../../../docs/labeler/confinement_bes_benchmark.md).

## Inputs
**qh_proxy**:
- `TokEye` tracks (EHO) on `mhr` 0/4, `ece` 8/20/40, `co2` 0/2, `bes` 26/28 (`mirnov` 0/8 when no `mhr`)
- `elm_free` (`elm_clock` on `D-alpha FS01..FS08`)
- `nbi_on` (`pinj_total`)
- `ip` (flat-top)

**confine-cnn**:
- `BES` 6 x 8 block of the 8 x 8 array, 2.5-150 kHz, 1,024-sample windows. 2026_10_03: the block chosen per shot from the channel positions (rows 0-5 for 349 of 401 shots, rows 1-6 for 37, rows 2-7 for 15), 1 MHz (1 ms). 2026_10_01: channels 8-55 of 64 (rows 1-6), 500 kHz (2 ms)

## Method
No WPQH label yet - the category needs its own lexicon id (the existing `qh` is
QH-mode). What exists:

- `qh_proxy` (`labeler.events.heuristics.qh_candidates`): QH candidate intervals
  where an EHO-like TokEye track (2-20 kHz, >= 2 harmonics) runs inside an
  `elm_free` interval, inside NBI-on, inside the Ip flat-top; confidence is the
  weaker of the track's and the ELM-free fraction. Because WPQH REPLACES the EHO
  with broadband MHD, this proxy will miss WPQH by construction.
- The BES benchmark (the published BES classifier, rebuilt from its description;
  not a detector of ours): scored on the WP QH intervals of the [shared confinement
  labels](../confinement/README.md), which hold WP QH although this category has no
  catalog label of its own, with `scripts/labeler/confinement_bes_benchmark.py` (first retrain) and
  `scripts/labeler/confinement_bes_ablation.py` (protocol-matched rows).
  Results and the paper's per-class scores:
  [confinement_bes_benchmark.md](../../../docs/labeler/confinement_bes_benchmark.md).
- `confine-ours` (experimental; 0D signals, no BES; [confinement_ours.md](../../../docs/labeler/confinement_ours.md)) segments
  WPQH on the roster, per-bin F1 0.801 with folds that hold out run days; the WPQH label rests on the
  pedestal width from Thomson scattering, which it does not read.

## Alias
wpqh, wpqh-mode, wide pedestal qh, wide-pedestal quiescent h-mode, wide pedestal quiescent h-mode, qh-mode, quiescent h-mode, eho (the QH-mode marker WPQH lacks)

## Future Implementations
- A WPQH rule would be: `elm_free` inside the flat-top, NO EHO track, low |tinj|, and a broadband transient / coherent-mode signature in 50-200 kHz on mhr.

## Reference
- K. H. Burrell et al., "Discovery of stationary operation of quiescent H-mode
  plasmas with net-zero neutral beam injection torque and high energy confinement
  on DIII-D", Phys. Plasmas 23, 056103 (2016).
- Xi Chen et al., "Bifurcation of quiescent H-mode to a wide pedestal regime in
  DIII-D and advances in the understanding of edge harmonic oscillations",
  Nucl. Fusion 57, 086008 (2017).
- K. H. Burrell et al., "Quiescent H-mode plasmas in the DIII-D tokamak",
  Phys. Plasmas 8, 2153 (2001).

## Contact
- **Azarakhsh Jalalvand**: aj17 [at] princeton [dot] edu
- **Nathaniel Chen**: nathaniel [at] princeton [dot] edu
