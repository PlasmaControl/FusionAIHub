# AE supervision swap

Status: **incomplete experiment, executable workflow and saved-baseline scoring
finished**. All three SLURM training attempts failed because the binding Pixi
`labelmaker` environment contains `torch 2.14.0+cpu`. Allocating a GPU does not
make this build CUDA-enabled. No new checkpoints were trained, and the clean
supervision experiment cannot yet establish whether the detector ranking reverses.

Records: [evaluation.json](../../outputs/labeler/ae/supervision_swap/evaluation.json)
and [manifest.json](../../outputs/labeler/ae/supervision_swap/manifest.json).
Real-data target checks are recorded in
[verification.json](../../outputs/labeler/ae/supervision_swap/verification.json).
They include the exact shot lists, input hashes, thresholds, attempted runs,
and paired differences. External artifacts and logs are under
`/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/aeswap/`.

## Frozen data and selection

The current main-checkout `alfven_eigenmode/review/labels.csv` was snapshotted
without changing it. It has **877 CSV rows, 371 present intervals, 506 absent
rows, and 180 shots**; these distinguish total rows from positive intervals.
Neither total matches the brief's anticipated 943/954 interval count. Its SHA256 is
`b4308ff52d766779f33a92e9441a916bb07c854dcc1a98a2f61109bb6431e93a`.
The snapshot is `round4/aeswap/inputs/dense.csv`. Interval counts alone do not
measure a change in frame coverage.

The original 120 training shots are permuted with NumPy seed **20261003**, then
split into **100 train / 20 selection**. The original **60 validation shots** are
reserved for evaluation. None of these 180 shots belongs to the fixed catalog's
blind test split. Exact lists and per-shot dataset SHA256s are in the manifest.

| Group | Scorable 10 ms frames | Dense present | Legacy present |
|---|---:|---:|---:|
| Training | 20,000 | 8,506 | 4,499 |
| Selection | 4,000 | 1,855 | 754 |
| Evaluation | 12,000 | 8,352 | 2,348 |

Source: `manifest.json`, `frame_counts`. Frames cover 0–2 s only.

## Training and threshold protocol

`scripts/labeler/ae_supervision_swap.py` reuses `ae_train.py` and the shipped
`AeSeldNet` architecture: four CO2 channels, 348 frequency bins, frequency pool
sizes `(6, 2, 29)`, two bidirectional GRUs, and activity/frequency outputs. The
recipe remains SCE, 710-column windows, eight windows per shot, batch size 16,
AdamW at 1e-4 with weight decay 1e-4, cosine decay to 1e-6, up to 30 epochs, and
patience five. Epoch selection minimizes the original combined loss evaluated
only on the 20 selection shots.

Only activity supervision changes:

- **Legacy:** the audit's native-column annotation is averaged onto 10 ms
  frames; at least half the columns must be annotated. Classes 1–4 count as AE;
  LFM is excluded. These binary frame targets are expanded to the native input
  grid. Unannotated frames are training negatives by this experimental convention.
- **Dense:** the current reviewed spans use the catalog's any-touch frame rule;
  uncertain, unobservable, and unassessed frames have zero weight. Overlapping
  individual/crowd intervals are combined by state precedence.
- **Threeway:** preserve the original native-column annotation/TokEye agreement:
  positive when both are active, negative when neither is active, and ignore
  disagreements.

Every arm retains exactly the original frequency target and its
`annotated & active & finite_frequency` weight. This holds auxiliary supervision
fixed, preventing a second changed target from confounding the activity swap.
`verify` checks this equality on all real train/selection records and checks that
the threeway arm matches the original target builder exactly.

After epoch selection, the threshold maximizes 10 ms selection F1 against each
arm's own activity target, with ties choosing the highest threshold. For
threeway, at least half a frame's native columns must have agreement weight,
and at least half those agreeing columns must be positive. The score is the
frame's mean native probability. This threshold is frozen for both evaluation
references; evaluation frames never tune it. The hard call is mean probability
at least the selected threshold, rather than the previous benchmark's majority
of native hard calls.

One seed (0) is frozen because CUDA execution failed before training. The
workflow supports three seeds in a fresh manifest (`prepare --seeds 0 1 2`),
and a three-at-a-time SLURM array. Missing arms remain unavailable in JSON and
the LaTeX table.

## Saved Garcia detectors

The available spectrogram predictions are used for `ae-rcn` and `ae-lstm`.
AE probability is the maximum of the first four class outputs, then the mean
over the four CO2 chords. The output bin nearest each 10 ms frame centre is used.
Saved row ordering is checked against Garcia's saved truth and hashed source
arrays. Cross-power variants are outside this primary swap comparison.

Only **19 of the 60 evaluation shots** have predictions and were held out from
Garcia's training. The other 41 cannot be scored from the available files.
Older thresholds use only the six available shots from the 20-shot selection
set, also held out from Garcia's training:
`172000, 176039, 176044, 176053, 176548, 176551`.
There are 1,200 selection frames, 175 legacy positives. The thresholds, selected
against legacy supervision, are **0.5281208929558328** for RCN and
**0.5118714645504951** for LSTM. They are used against both references.
This limited selection coverage and resulting calibration uncertainty are material.

On the shared held-out set, both references score 3,800 frames: 2,595 dense
positives and 796 legacy positives. The 19 shots are:
`170660, 170661, 170663, 170666, 170669, 170677, 170678, 170718, 170725,
170729, 170730, 170792, 170793, 170798, 170801, 170803, 175241, 175245, 178879`.

All intervals below are 95% shot-bootstrap intervals: **1,000 replicates**, seed
**20261004**, pooled frame metrics within each draw. Paired differences use
identical shot draws. Source: `evaluation.json`, `results.fair_19.references`.

| Model | Reference | AUROC [95% CI] | AUPRC [95% CI] | Selection-threshold F1 [95% CI] |
|---|---|---|---|---|
| ae-rcn | Dense | 0.862 [0.810, 0.907] | 0.930 [0.906, 0.955] | 0.350 [0.244, 0.466] |
| ae-lstm | Dense | 0.736 [0.654, 0.802] | 0.855 [0.835, 0.885] | 0.382 [0.260, 0.521] |
| ae-rcn | Legacy | 0.911 [0.878, 0.939] | 0.691 [0.554, 0.798] | 0.571 [0.439, 0.660] |
| ae-lstm | Legacy | 0.898 [0.854, 0.930] | 0.668 [0.509, 0.779] | 0.602 [0.435, 0.718] |

RCN minus LSTM AUROC is **0.125 [0.058, 0.200]** against dense labels and
**0.013 [-0.012, 0.039]** against legacy labels. F1 differences are
**-0.032 [-0.079, 0.016]** and **-0.031 [-0.102, 0.066]**, respectively.
The selection-calibrated operating points give much lower dense F1 than the
older benchmark's published operating points; those are different protocols.
These baseline results alone do not answer the supervision-swap question.

The generated `round4/aeswap/table_supervision_swap.tex` shows these results and
explicit missing entries for every untrained `ae-ours` arm. It requires booktabs.
The optional `ae-lstm-retrained` experiment was deferred because the required
training environment cannot execute CUDA.

## Reproduction and controller continuation

From this worktree, set:

```bash
export TMPDIR=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/claude-89242e53/r4/tmp/aeswap
export LABELER_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker
export LABELER_LABEL_TABLES=/scratch/gpfs/nc1514/FusionAIHub/data/events
export LABELER_NO_FETCH=1
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

Preparation is already complete and refuses to overwrite its frozen manifest.
To score or verify:

```bash
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker python scripts/labeler/ae_supervision_swap.py verify
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker python scripts/labeler/ae_supervision_swap.py evaluate
```

SLURM array **2952124**, tasks 0–2, reached `stellar-m01g1` and failed with
`required Pixi environment has no CUDA torch`. Each arm's `attempt.json` records
the failure. The GPU-0 head-node fallback uses the same CPU-only environment and
therefore cannot resolve it.

The controller must authorize a CUDA-enabled Pixi environment instead of the
binding `-e labelmaker` before training. The existing manifest's `default`
environment specifies CUDA PyTorch. These are exact continuation commands for
that authorized exception, without installing or fetching anything:

```bash
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e default python -c 'import torch, pandas, scipy, h5py; assert torch.cuda.is_available(); print(torch.__version__)'
AESWAP_PIXI_ENV=default sbatch scripts/labeler/ae_supervision_swap.sbatch
# After all three runs have finished:
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker python scripts/labeler/ae_supervision_swap.py evaluate
```

No CUDA environment exception was exercised by the implementer. If that array
remains pending beyond 30 minutes, the equivalent head-node command for each
arm is:

```bash
for arm in legacy dense threeway; do
    CUDA_VISIBLE_DEVICES=0 pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e default python -u scripts/labeler/ae_supervision_swap.py train --supervision "$arm" --seed 0 > "$LABELER_ROOT/round4/aeswap/head-$arm.log" 2>&1
    bash "$LABELER_ROOT/scratch/bin/tmpsweep.sh"
done
```

The trainer caps CUDA allocations at 10 GiB and uses eight workers. On a V100,
which lacks bfloat16 support, it uses float32; A100 jobs use bfloat16 autocast.
The architecture, objective and optimizer recipe stay identical across arms.
