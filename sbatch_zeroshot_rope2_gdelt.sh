#!/bin/bash
#SBATCH --partition=slowlane
#SBATCH --job-name=trix_zs_gdelt
#SBATCH --output=%u_job_%j.out
#SBATCH --nodes=1
#SBATCH --nodelist=aisa-gpuB02
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=A40:1
#SBATCH --time=03:00:00

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

CKPT=/mnt/nfs/home/ac139229/jiaxin/git/git/TTRIX/output/TRIX/TemporalICEWS14/2026-04-28-10-11-09/model_epoch_1.pth

PYTHONPATH=src python src/run_entity.py \
    -c config/eval_gdelt_indt100_rope2.yaml \
    --gpus [0] \
    --ckpt $CKPT

echo "[sbatch] end: $(date -Is)"
