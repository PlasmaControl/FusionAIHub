#!/usr/bin/env python
"""Run the pinned TokEye U-Net over a chunk of a shot list, GPU-first.

    PYTHONPATH=$REPO/src python scripts/labelmaker/tokeye_masks.py \
        --shot-file $LABELMAKER_ROOT/recommender_v1.txt \
        --chunk $SLURM_ARRAY_TASK_ID --n-chunks 8 \
        --rank $SLURM_PROCID --world 2 \
        --root $LABELMAKER_ROOT --corpus /scratch/gpfs/EKOLEMEN/foundation_model \
        --plan round1 --device cuda --tile-batch 96 --prep-workers 18 \
        --prefetch 4 --amp --timeout 240

Writes exactly what `python -m labelmaker.run events` writes -
`masks/<shot>_masks.npz`, `events/<shot>_events.parquet` and the index rows -
and is required to write it byte for byte the same. What it changes is the
schedule: the shot list is split across SLURM array tasks (`--chunk`) and
across the ranks of one `srun` step (`--rank`), and within a rank the CPU
half of each block - the HDF5 read, the STFT, the standardisation - runs in
a pool of spawned prep workers while this process keeps the card busy with
the block before. On the 500-shot `recommender_v1` list a shot is ~15
core-seconds of that CPU work against ~0.84 s of A100 forward pass, which is
why the sequential loop measured 1 % GPU utilisation.

Everything is in `labelmaker.events.driver`, so that it is importable, and
testable (`tests/labelmaker/test_tokeye_masks.py`) without running a script.
`--help` documents every flag; `driver.py`'s module docstring documents the
memory bound and the failure model.

Not a batch script: `scripts/labelmaker/tokeye_masks.sbatch` (task L12) is
what submits this, and the pilot it is sized from is task L11.
"""
from __future__ import annotations

import sys

if __name__ == "__main__":
    from labelmaker.events.driver import main

    sys.exit(main())
