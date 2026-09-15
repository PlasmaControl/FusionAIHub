#!/usr/bin/env python
"""Gate a finished SLURM job on its utilisation; exit 1 if it wasted its reservation.

    # in the sbatch that submits the work
    JOBID=$(sbatch --parsable my_work.sbatch)
    sbatch --dependency=afterany:$JOBID --partition=serial --time=00:10:00 \
        --wrap "pixi run -e labelmaker python scripts/labeler/jobstats_check.py \
            --job-id $JOBID --wait-for-data 300 \
            --preserve-dir $LABELER_ROOT/runs/slurm"

    # offline, on a report you already have
    python scripts/labeler/jobstats_check.py \
        --jobstats-file 2925387_0.jobstats.txt --sacct-file 2925387_0.sacct.txt

Everything is in `labeler.jobstats`, so the parsing and the gate are testable
without a scheduler and importable from other tooling; this file only exists to
give SLURM a path to run. `--help` lists every flag.
"""
from labeler.jobstats import main

if __name__ == "__main__":
    raise SystemExit(main())
