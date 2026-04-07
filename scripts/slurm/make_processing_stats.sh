#!/bin/bash
#SBATCH --job-name=make_processing_stats_parallel
#SBATCH --output=logs/make_processing_stats_parallel.out
#SBATCH --error=logs/make_processing_stats_parallel.err
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=12:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=ps9551@princeton.edu

pixi run python -u ../data_preparation/make_processing_stats.py
