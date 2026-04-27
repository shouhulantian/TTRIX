#!/bin/bash
#SBATCH --account=hk-project-pai00057
#SBATCH --partition=dev_accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=trix_gdelt
#SBATCH --output=logs/gdelt_indt100_%j.out
#SBATCH --error=logs/gdelt_indt100_%j.err

cd /hkfs/work/workspace/scratch/st_ac139229-TRIX/TRIX
mkdir -p logs

# Use g++ for CUDA extension compilation (Intel icpx is not PyTorch-compatible)
export CXX=g++
export CC=gcc
export TORCH_CUDA_ARCH_LIST="8.0"

CKPT="/hkfs/work/workspace/scratch/st_ac139229-TRIX/TRIX/entity_prediction.pth"

echo "=== Zero-shot TRIX on GDELTIndT_100 (timestamps stripped) ==="
srun python src/run_entity.py -c config/run_entity_gdelt_indt100.yaml \
  --gpus [0] --epochs 0 --bpe null --ckpt $CKPT

echo ""
echo "Done."
