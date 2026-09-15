#!/usr/bin/env python
"""Run the pinned TokEye U-Net over a chunk of a shot list, GPU-first.

Three commands, in this order. Once, before the array - the shared logbook
subset, which no array task may write (a whole-file rewrite loses the other
tasks' records). This pre-pass is REQUIRED for multi-rank or multi-chunk
runs to match `python -m labeler.run events`. Without it, `readonly`
adds a `text` skip for uncovered shots and omits `text` from their declared
sources; a covered logbook record with no matches still declares `text`
with zero events. Check that each run JSON's `text_subset_missing` is empty:

    PYTHONPATH=$REPO/src python scripts/labeler/tokeye_masks.py \
        --build-text-subset --shot-file $LABELER_ROOT/recommender_v1.txt \
        --root $LABELER_ROOT

Then the array itself. `--prefetch` is the concurrency as well as the queue,
so it is at least `--prep-workers`, and `--no-index` keeps the tasks off the
one other file they all share:

    PYTHONPATH=$REPO/src python scripts/labeler/tokeye_masks.py \
        --shot-file $LABELER_ROOT/recommender_v1.txt \
        --chunk $SLURM_ARRAY_TASK_ID --n-chunks 8 \
        --rank $SLURM_PROCID --world 2 \
        --root $LABELER_ROOT --corpus /scratch/gpfs/EKOLEMEN/foundation_model \
        --plan round1 --device cuda --tile-batch 96 --prep-workers 18 \
        --prefetch 18 --tail-workers 1 --no-index --amp --timeout 240

Then once more, in an `afterok` job - the index, rebuilt in one pass from the
per-shot events files it is derived from:

    PYTHONPATH=$REPO/src python scripts/labeler/tokeye_masks.py \
        --rebuild-index --root $LABELER_ROOT

Writes exactly what `python -m labeler.run events` writes -
`masks/<shot>_masks.npz`, `events/<shot>_events.parquet` and the index rows -
and is required to write it byte for byte the same. What it changes is the
schedule: the shot list is split across SLURM array tasks (`--chunk`) and
across the ranks of one `srun` step (`--rank`), and within a rank the CPU
half of each block - the HDF5 read, the STFT, the standardisation - runs in
a pool of spawned prep workers while this process keeps the card busy with
the block before, and each shot's TAIL (the heuristics, the text and the two
writes) runs in another process beside the next shot's forward passes. On the
500-shot `recommender_v1` list a shot is ~15 core-seconds of that CPU work
and ~3.7 s of tail against ~0.84 s of A100 forward pass, which is why the
sequential loop measured 1 % GPU utilisation.

Everything is in `labeler.events.driver`, so that it is importable, and
testable (`tests/labeler/test_tokeye_masks.py`) without running a script.
`--help` documents every flag; `driver.py`'s module docstring documents the
memory bound and the failure model.

Not a batch script: `scripts/labeler/tokeye_masks.sbatch` (task L12) is
what submits this, and the pilot it is sized from is task L11.
"""
from __future__ import annotations

import sys

if __name__ == "__main__":
    from labeler.events.driver import main

    sys.exit(main())
