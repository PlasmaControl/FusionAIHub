#!/bin/bash
#SBATCH --job-name=make_processing_stats
#SBATCH --output=logs/make_processing_stats.out
#SBATCH --error=logs/make_processing_stats.err
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=24:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=ps9551@princeton.edu

pixi run python -u ../data_preparation/make_processing_stats.py
