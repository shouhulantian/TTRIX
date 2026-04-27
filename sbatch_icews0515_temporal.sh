#!/bin/bash
#SBATCH --account=hk-project-pai00057
#SBATCH --partition=dev_accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --job-name=trix_ic0515_t
#SBATCH --output=logs/icews0515_temporal_%j.out
#SBATCH --error=logs/icews0515_temporal_%j.err

cd /hkfs/work/workspace/scratch/st_ac139229-TRIX/TRIX
mkdir -p logs

export CXX=g++
export CC=gcc
export TORCH_CUDA_ARCH_LIST="8.0"

CKPT="/hkfs/work/workspace/scratch/st_ac139229-TRIX/TRIX/entity_prediction.pth"

echo "=== ICEWS0515 (static filter) ==="
srun python src/run_entity.py -c config/run_entity_icews.yaml \
  --dataset ICEWS0515 --gpus [0] --epochs 0 --bpe null --ckpt $CKPT

echo ""
echo "=== TemporalICEWS0515 (time-aware filter) ==="
srun python src/run_entity.py -c config/run_entity_icews.yaml \
  --dataset TemporalICEWS0515 --gpus [0] --epochs 0 --bpe null --ckpt $CKPT

echo ""
echo "Done."
