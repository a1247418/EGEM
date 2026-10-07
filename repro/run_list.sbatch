#!/bin/bash
# Runs the experiments listed in a file (one line of run_scenario.py arguments each) one after another on a GPU:
#   sbatch repro/run_list.sbatch jobs.txt
#SBATCH --job-name=egem_list
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=repro/logs/list_%j.log
cd "$SLURM_SUBMIT_DIR"
while read -r line; do
    [ -z "$line" ] && continue
    echo "=== $line"
    ~/.conda/envs/egem/bin/python -u repro/run_scenario.py --threads 8 $line || echo "FAILED: $line"
done < "$1"
