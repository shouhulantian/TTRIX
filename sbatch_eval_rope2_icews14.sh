#!/bin/bash
#SBATCH --partition=slowlane
#SBATCH --job-name=trix_test_14
#SBATCH --output=%u_job_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=A40:1
#SBATCH --time=02:00:00

set -euo pipefail
echo "[sbatch] node=$(hostname) job=$SLURM_JOB_ID"
echo "[sbatch] start: $(date -Is)"

module purge
module load Miniconda3
source "${EBROOTMINICONDA3}/bin/activate"
conda activate ultra_env

export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

REPO=/mnt/nfs/home/ac139229/jiaxin/git/git/TTRIX
cd "$REPO"

# Best-by-val-MRR checkpoint from job 26939: epoch_3 with val MRR 0.6001.
CKPT=/mnt/nfs/home/ac139229/jiaxin/git/git/TTRIX/output/TRIX/TemporalICEWS14/2026-04-28-10-11-09/model_epoch_3.pth

PYTHONPATH=src python src/run_entity.py \
    -c config/eval_temporal_icews14_rope2.yaml \
    --gpus [0] \
    --ckpt $CKPT

echo "[sbatch] end: $(date -Is)"
