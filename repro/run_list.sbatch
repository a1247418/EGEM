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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while read -r line; do
    [ -z "$line" ] && continue
    echo "=== $line"
    # retry: on shared GPUs other jobs can briefly take the memory
    for attempt in 1 2 3; do
        ~/.conda/envs/egem/bin/python -u repro/run_scenario.py --threads 8 $line && break
        [ $attempt = 3 ] && echo "FAILED: $line" || sleep 300
    done
done < "$1"
