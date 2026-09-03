#!/bin/bash
#SBATCH --job-name=benchmark_data_loader
#SBATCH --output=logs/benchmark_data_loader.out
#SBATCH --error=logs/benchmark_data_loader.err
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=04:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=ps9551@princeton.edu

pixi run python ../training/benchmark_data_loader.py
