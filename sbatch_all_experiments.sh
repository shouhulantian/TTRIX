#!/bin/bash
#SBATCH --account=hk-project-pai00057
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --job-name=trix_all
#SBATCH --output=logs/all_experiments_%j.out
#SBATCH --error=logs/all_experiments_%j.err

cd /hkfs/work/workspace/scratch/st_ac139229-TRIX/TRIX
mkdir -p logs

CKPT="entity_prediction.pth"

echo "=========================================="
echo "Exp 1: TRIX zero-shot on FB15k237 (baseline verification)"
echo "=========================================="
srun python src/run_entity.py -c config/run_entity_transductive.yaml \
  --dataset FB15k237 --gpus [0] --epochs 0 --bpe null --ckpt $CKPT

echo ""
echo "=========================================="
echo "Exp 2: TRIX zero-shot on ICEWS14 (structural, standard filtering)"
echo "=========================================="
srun python src/run_entity.py -c config/run_entity_icews.yaml \
  --dataset ICEWS14 --gpus [0] --epochs 0 --bpe null --ckpt $CKPT

echo ""
echo "=========================================="
echo "Exp 3: TRIX zero-shot on ICEWS0515 (structural, standard filtering)"
echo "=========================================="
srun python src/run_entity.py -c config/run_entity_icews.yaml \
  --dataset ICEWS0515 --gpus [0] --epochs 0 --bpe null --ckpt $CKPT

echo ""
echo "=========================================="
echo "Exp 4: TRIX fine-tune on ICEWS14 (structural, 3 epochs)"
echo "=========================================="
srun python src/run_entity.py -c config/run_entity_icews.yaml \
  --dataset ICEWS14 --gpus [0] --epochs 3 --bpe 1000 --ckpt $CKPT

echo ""
echo "=========================================="
echo "Exp 5: TRIX zero-shot on TemporalICEWS14 (time-aware filtering)"
echo "=========================================="
srun python src/run_entity.py -c config/run_entity_icews.yaml \
  --dataset TemporalICEWS14 --gpus [0] --epochs 0 --bpe null --ckpt $CKPT

echo ""
echo "=========================================="
echo "Exp 6: TRIX zero-shot on TemporalICEWS0515 (time-aware filtering)"
echo "=========================================="
srun python src/run_entity.py -c config/run_entity_icews.yaml \
  --dataset TemporalICEWS0515 --gpus [0] --epochs 0 --bpe null --ckpt $CKPT

echo ""
echo "=========================================="
echo "Exp 7: TRIX transfer ICEWS14->ICEWS0515 (fine-tune 3 epochs, zero-shot test)"
echo "=========================================="
srun python src/run_entity.py -c config/run_entity_icews_transfer.yaml \
  --gpus [0] --epochs 3 --bpe 1000 --ckpt $CKPT

echo ""
echo "All experiments complete."
