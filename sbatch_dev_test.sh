#!/bin/bash
#SBATCH --account=hk-project-pai00057
#SBATCH --partition=dev_accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=trix_dev
#SBATCH --output=logs/dev_test_%j.out
#SBATCH --error=logs/dev_test_%j.err

cd /hkfs/work/workspace/scratch/st_ac139229-TRIX/TRIX
mkdir -p logs

# Use g++ for CUDA extension compilation (not Intel icpx)
export CXX=g++
export CC=gcc
export TORCH_CUDA_ARCH_LIST="8.0"

CKPT="/hkfs/work/workspace/scratch/st_ac139229-TRIX/TRIX/entity_prediction.pth"

echo "=== Test 1: Zero-shot on ICEWS14 (structural, standard filtering) ==="
srun python src/run_entity.py -c config/run_entity_icews.yaml \
  --dataset ICEWS14 --gpus [0] --epochs 0 --bpe null --ckpt $CKPT

echo ""
echo "=== Test 2: Zero-shot on TemporalICEWS14 (time-aware filtering) ==="
srun python src/run_entity.py -c config/run_entity_icews.yaml \
  --dataset TemporalICEWS14 --gpus [0] --epochs 0 --bpe null --ckpt $CKPT

echo ""
echo "=== Test 3: Transfer ICEWS14->ICEWS0515 (zero-shot, no fine-tune) ==="
srun python src/run_entity.py -c config/run_entity_icews_transfer.yaml \
  --gpus [0] --epochs 0 --bpe null --ckpt $CKPT

echo ""
echo "All tests complete."
