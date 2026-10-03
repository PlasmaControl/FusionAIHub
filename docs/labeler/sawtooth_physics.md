# Physics-informed sawtooth labels and trained pickers

This is an additive research method. It does not replace `ece_sawtooth`, refresh
the production event store, or establish a stable sawtooth model. Independent
expert validation is weak, and the missing ECE radius calibration prevents the
requested inversion-radius consistency check. Treat its outputs as algorithmic
candidate labels requiring review.

## Detector

`labeler.sawtooth.physics.detect` follows the edge, coincidence and redistribution
recipe of Gude et al., PPCF 59 (2017) 095009, DOI 10.1088/1361-6587/aa79d8.
It uses the complete 48-channel ECE array, a first derivative of a Gaussian
(sigma 0.25 ms, truncated at 3.5 sigma), and POSR in a 10.3 ms frame: exclude
the kernel-length largest absolute filtered values before estimating noise.
The POSR threshold is 6.0, with at least two channels agreeing within 0.5 ms.
`noise_calibration.json` records the seeded Gaussian-noise calibration, rather
than estimating false alarms from test shots.

The signed step profile must have at least 2% relative total amplitude,
net/absolute redistribution below 0.9, a contiguous loss block of at least two
channels, and a neighboring gain block of at least two channels within six
channels. Without calibrated geometry, at least two of the established ECE
core-proxy channels 20–35 must lose temperature. Channel order and temperature
do not prove radial localization. The two-sided asymmetry test in Gude cannot
be reproduced with an uncalibrated one-sided array. Inverted central ECE crashes
are not included in this implementation.

SXR is corroborating evidence, not a second independent label generator. The
first lit 32-channel fan among offsets 0, 32, 160 and 192 contributes up to eight
bright channels. This is weaker than the spatially calibrated SXR profile test
in the paper. D-alpha spikes above local median plus six MAD noise scales within
2 ms veto candidates confined to edge channels. The ECE core-proxy requirement
also rejects edge-only candidates before this test. Pellets are not separately
identified because no pellet timing input is used.

Where local EFIT q-min is measured, values above 1.05 reject a candidate; below
0.3 MA measured plasma current also rejects it. EFIT q-min is not a measured
q=1 radius, and reconstruction bias can reject visible sawtooth trains. If
calibrated `ece_psi` and `qpsi` become available, the library establishes an
axis-connected q=1 surface, rejects known profiles without it, and requires
inversion rho to lie within 0.15 of q=1 rho. Here rho is the square root of
normalized poloidal flux. Geometry outside its clock coverage stays unknown.
Missing calibration produces null `inversion_rho`/`q1_rho`, never a guessed radius.

Crash points have `crowd=False` in attributes and in the CSV column. A span has
`crowd=True` and covers first through last crash in a train of at least three.
Adjacent periods must lie between 5 and 250 ms and differ by no more than a
factor 2.5. The train's median period and inversion channel are stored. Isolated
accepted profiles remain point candidates with null period and produce no span.
There is no causal or real-time claim: the filter and frame use future samples.

## Signal preparation and output scope

The corpus and all existing label/event stores are read-only. A 500 kHz ECE
record is sampled at approximately 50 kHz, then FIR-decimated to 10 kHz. The
first reduction lacks an antialias filter and may alias noise above 25 kHz;
the second is antialiased. This bounded-memory compromise needs validation
against native-rate filtering before publication. Nonuniform sampled clocks
are rejected. Missing data invalidate filter support rather than creating
edges by interpolation. ECE values outside 0–100 keV are excluded.

Cohort labels use the fixed plasma windows in `cohort.csv`. Population shots
outside the cohort use local Ip extent above 10% of peak when available,
otherwise ECE extent after 50 ms; that fallback may include startup/shutdown
artifacts. Neither labels nor inferred negatives establish diagnostic
observability when ECE is cut off.

Small CSV shards in `data/events/sawtooth_oscillation/extend_saw_physics/` cover
the cohort and all reviewed shots. Times are milliseconds and spans are
half-open. They include `category=1`, `crowd`, confidence and compact JSON
attributes (inversion radius, inversion channel and period); complete profile
attributes remain in the external per-shot records.
They are not automatically installed as a replacement catalog table. Population
CSV exports, per-shot JSON, signals, checkpoints, predictions and figures are
under `$LABELER_ROOT/round4/saw/`. Files ending `.partial` are incomplete atomic
JSON writes and are not read as records. Resume only within an unchanged frozen
rule; use a new `--work` directory after changing the method.

## Models and DIII-D adaptations

`saw-hl3` reimplements OuYang et al., PPCF 67 (2025) 105004,
DOI 10.1088/1361-6587/ae0786, from the paper digest without external code.
Its four input channels are ECE 20–27 mean (core proxy), ECE 8–15 mean (outer
proxy), mean of Mirnov channels 0–1, and cached Ip in MA. ECE replaces the
published core/edge SXR pair; the Mirnov average is a proxy whose poloidal
orientation has not been verified. Missing auxiliary inputs become the
training-normalized mean. Low-cadence Ip is interpolated without acquiring
10 kHz physical bandwidth.

The input is 200 samples (20 ms). The network has Conv1d(16, kernel 3),
Conv1d(32, kernel 3, stride 2) with a projected residual, batch normalization,
ReLU, channel attention, one bidirectional LSTM with 32 units per direction,
sequence attention, dense 128/32 with ReLU and dropout 0.5, and three logits.
The single LSTM follows the paper text; its figure depicts two layers.
Classes are absent, shorter-period train, and longer-period train. The boundary
is the median train period in each training fold, not the HL-3-specific 23 ms.

Training uses weighted cross-entropy, Adam at 0.001, weight decay 0.003,
exponential decay 0.97, batch 128, amplitude scaling 0.8–1.2 and Gaussian noise
0.1 after standardization. Time shifting is omitted because its units are
ambiguous in the paper and boundary-window labels can change. The CPU budget is
20 epochs with 80 replacement-sampled batches per epoch, 64 sampled windows per
training shot, and five-epoch validation-loss patience. These budget limits are
an adaptation, not a complete 100-epoch reproduction. Each fold records its
actual selected checkpoint, class boundary and calibration grid.

The published classifier has no crash-time output. For the requested crash
score, its predicted presence gates a separate Gaussian-derivative core-ECE
picker. Gate thresholds 0.3/0.5/0.7 and derivative cutoffs 2/4/6 MAD scales are
chosen by fixed validation-shot crash F1. This is a timing adaptation to the
published classifier, not a replicated HL-3 timing metric. AMPD was used by
the HL-3 paper to create its labels; this experiment intentionally trains both
models against the requested physics labels, not an AMPD relabeling.

`saw-ours` is a PhaseNet-style one-dimensional U-Net inspired by Zhu and Beroza,
GJI 216 (2019), arXiv:1803.03211. Its 48-channel input spans 100 ms. Four
kernel-seven, stride-four downsampling stages use widths 8/11/16/22/32; four
transposed-convolution stages have skip connections. Outputs are crash/noise
softmax and an independent train-presence sigmoid. Crash targets are Gaussians
with sigma 0.5 ms truncated at three sigma; crash loss has positive weight 20.
An interval BCE loss trains the second task. Training uses Adam at 0.001,
weight decay 0.0001, batch 32, and the same CPU epoch/step cap. Half the sampled
training windows are centered on candidate crashes and half are uniform. The
100 ms receptive field and presence head are adaptations; widths and Gaussian
target convention retain the PhaseNet idea.

## Evaluation and limitations

Three CV folds contain only the fixed cohort `train` shots; each whole shot
belongs to one fold. All reviewed shots are excluded from every training fold
and from validation. The fixed `val` shots select checkpoints and crash
thresholds. The fixed `test` shots are never loaded by the benchmark. Global
channel mean and standard deviation and the period class boundary are fitted
only on each fold's training windows. No windows cross shot boundaries.

Primary results pool out-of-fold predictions on train-cohort shots. One-to-one
nearest-first crash matching uses both +/-1 and +/-2 ms. Presence is scored
every 2 ms at threshold 0.5 with AUROC, average precision (AUPRC), precision,
recall and F1. Probability scores use a 512-bin histogram, including half
credit for AUROC ties. Intervals are 95% percentiles of 1000 shot-bootstrap
replicates, seed 20261003. The baseline additionally has three-class window
accuracy and a shot-bootstrap CI. Its legacy published accuracy is a different
task and population and is not directly comparable to crash F1.

The expert table supplies positive/negative spans, not crash times. Categories
at least two abstain. Expert presence metrics compare the three-fold model
ensemble against all known bins in all reviewed shots. Detector span matching
uses one-to-one IoU >=0.1, in addition to bin-level metrics and any-overlap recall.
The fraction of predicted points inside positive expert spans is only a
support check; crash precision, recall and F1 against experts are unavailable.
Only three reviewed shots make bootstrap intervals fragile. The physics rule
was frozen before expert validation, and its poor validation was not used to
retune it. It cannot presently be called an improved or independently validated
ground-truth set. Self-consistency of models with algorithmic labels does not
overcome that limitation.

The gallery is a uniform random sample of twelve usable nonreview train shots,
seed 20261003. It shows core/outer channel proxies, all accepted points, train
spans, inversion channel and cached q-min. Physical inversion radius versus q=1
cannot be shown without calibration. Both PDF and 150-dpi PNG are saved at
3.5-inch column width with fonts at least seven points. Every PNG was inspected.

## Reproduction

From this worktree, for each command set:

```bash
export TMPDIR=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/claude-89242e53/r4/tmp/saw
export LABELER_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker
export LABELER_LABEL_TABLES=/scratch/gpfs/nc1514/FusionAIHub/data/events
export LABELER_NO_FETCH=1 PYTHONPATH=$PWD/src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

Use the following prefix for every Python invocation:

```bash
pixi run --frozen --no-install \
  --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker python
```

Append these stage commands:

```text
scripts/labeler/sawtooth_physics.py labels --workers 8
scripts/labeler/sawtooth_physics.py labels --population --workers 8
scripts/labeler/sawtooth_physics.py validate
scripts/labeler/sawtooth_physics.py gallery
scripts/labeler/sawtooth_benchmark.py train --threads 4
scripts/labeler/sawtooth_benchmark.py evaluate
```

The population pass can instead use `sbatch scripts/labeler/sawtooth_population.sbatch`
and then the unsharded resumable population command to aggregate completed
records. No fetching, environment installation, GPU execution or network access
is needed. The existing required environment has CPU-only PyTorch.

Evaluation JSON records under `outputs/labeler/sawtooth/` accompany this document:
`cohort_labels.json`, `population_labels.json`, `noise_calibration.json`,
`validation.json`, `split_manifest.json`, `gallery.json`, each model's three
fold records, and `benchmark.json`. The results table is generated by the
committed `sawtooth_results.py` script from those records.
