#!/bin/bash
#SBATCH --job-name=convert_dtype     # create a short name for your job
#SBATCH --output=logs/convert_dtype.out
#SBATCH --error=logs/convert_dtype.err
#SBATCH --cpus-per-task=1           # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --nodes=1                    # node count
#SBATCH --mem-per-cpu=32G            # memory per cpu-core (4G is default)
#SBATCH --time=12:00:00               # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all              # send email on job start, end and fault
#SBATCH --mail-user=ps9551@princeton.edu

pixi run python -u ../data_preparation/convert_float64_to_float32.py
