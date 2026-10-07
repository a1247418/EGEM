#!/bin/bash
# Extract the ImageNet classes used by the scenarios into ~/EGEM_work/data/imagenet (shards go to node-local scratch).
#SBATCH --job-name=imagenet_subset
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=tmp:30G
#SBATCH --time=08:00:00
#SBATCH --output=repro/logs/imagenet_subset_%j.log
cd "$SLURM_SUBMIT_DIR"
echo "scratch: $TMPDIR"; df -h "$TMPDIR" | tail -1
~/.conda/envs/egem/bin/python -u repro/fetch_imagenet_subset.py --tmp "$TMPDIR" "$@"
