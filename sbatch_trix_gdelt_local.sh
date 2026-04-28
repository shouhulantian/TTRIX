#!/bin/bash
#SBATCH --partition=slowlane
#SBATCH --job-name=trix_gdelt
#SBATCH --output=%u_job_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=A100:1
#SBATCH --time=02:00:00

set -euo pipefail
echo "[sbatch] node=$(hostname) job=$SLURM_JOB_ID"
echo "[sbatch] start: $(date -Is)"

module purge
module load Miniconda3
source "${EBROOTMINICONDA3}/bin/activate"
conda activate ultra_env

export OMP_NUM_THREADS=16
export PYTHONUNBUFFERED=1

REPO=/mnt/nfs/home/ac139229/jiaxin/git/git/TTRIX
cd "$REPO"

python src/run_entity.py \
    -c config/run_entity_gdelt_indt100_local.yaml \
    --gpus [0] --epochs 0 --bpe null \
    --ckpt $REPO/entity_prediction.pth

echo "[sbatch] end: $(date -Is)"
