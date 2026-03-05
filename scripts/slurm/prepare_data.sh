#!/bin/bash
#SBATCH --job-name=prepare_data      # create a short name for your job
#SBATCH --output=logs/prepare_data.out
#SBATCH --error=logs/prepare_data.err
#SBATCH --cpus-per-task=32           # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --nodes=1                    # node count
#SBATCH --mem-per-cpu=16G            # memory per cpu-core (4G is default)
#SBATCH --time=1:00:00               # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all              # send email on job start, end and fault
#SBATCH --mail-user=ps9551@princeton.edu

pixi run python -u ../data_preparation/prepare_data.py
