#!/bin/bash
# Fine-tune VGG-16 on ISIC 2019 (paper Supp. E). Needs the cache from `train_isic.py --build_cache`.
# Smoke test:  sbatch --time=00:15:00 repro/train_isic.sbatch --epochs 1 --max_steps 200 --out repro/scratch/isic_smoke.model
# Full run:    sbatch repro/train_isic.sbatch
#SBATCH --job-name=isic_vgg16
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=repro/logs/isic_train_%j.log
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv
~/.conda/envs/egem/bin/python -u repro/train_isic.py "$@"
