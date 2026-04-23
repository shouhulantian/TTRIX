#!/bin/bash
#SBATCH --account=hk-project-pai00057
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=trix_zeroshot
#SBATCH --output=logs/zeroshot_%j.out
#SBATCH --error=logs/zeroshot_%j.err

cd /hkfs/work/workspace/scratch/st_ac139229-TRIX/TRIX
mkdir -p logs

echo "=========================================="
echo "Phase 1: Standard KG zero-shot baselines"
echo "=========================================="

echo ""
echo "=== Entity prediction: FB15k237 (transductive) ==="
srun python src/run_entity.py -c config/run_entity_transductive.yaml \
  --dataset FB15k237 --gpus [0] --epochs 0 --bpe null \
  --ckpt entity_prediction.pth

echo ""
echo "=== Entity prediction: WN18RR (transductive) ==="
srun python src/run_entity.py -c config/run_entity_transductive.yaml \
  --dataset WN18RR --gpus [0] --epochs 0 --bpe null \
  --ckpt entity_prediction.pth

echo ""
echo "=========================================="
echo "Phase 2: ICEWS zero-shot (structural only)"
echo "=========================================="

echo ""
echo "=== Entity prediction: ICEWS14 (zero-shot from pretrained) ==="
srun python src/run_entity.py -c config/run_entity_icews.yaml \
  --dataset ICEWS14 --gpus [0] --epochs 0 --bpe null \
  --ckpt entity_prediction.pth

echo ""
echo "=== Entity prediction: ICEWS0515 (zero-shot from pretrained) ==="
srun python src/run_entity.py -c config/run_entity_icews.yaml \
  --dataset ICEWS0515 --gpus [0] --epochs 0 --bpe null \
  --ckpt entity_prediction.pth
