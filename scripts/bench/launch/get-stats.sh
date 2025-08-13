#!/bin/bash

#SBATCH --job-name=get-stats
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=a100_3g.40gb:1
#SBATCH --cpus-per-task=7
##SBATCH --hint=nomultithread
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --mail-type=FAIL
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err

module purge
uv run --group scripts -m scripts.bench.get_stats
