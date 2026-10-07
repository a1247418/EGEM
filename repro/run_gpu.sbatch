#!/bin/bash
# Run one refinement experiment on a GPU, e.g.
#   sbatch repro/run_gpu.sbatch --scenario isic-1 --data_root ~/EGEM_work/data/isic --refinement egem --poisoning uniform
#SBATCH --job-name=egem_refine
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=repro/logs/refine_%j.log
cd "$SLURM_SUBMIT_DIR"
~/.conda/envs/egem/bin/python -u repro/run_scenario.py --threads 8 "$@"
