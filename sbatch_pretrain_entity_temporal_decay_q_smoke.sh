#!/bin/bash
#SBATCH --partition=slowlane
#SBATCH --job-name=trix_smoke_decayq
#SBATCH --output=%u_job_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gpus=A40:4
#SBATCH --time=01:00:00

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

MASTER_PORT=$((29500 + RANDOM % 1000))
PYTHONPATH=src python -m torch.distributed.launch --nproc_per_node=4 --master_port=$MASTER_PORT src/pretrain_entity_temporal.py \
    -c config/pretrain_entity_temporal_decay_q_smoke.yaml \
    --gpus [0,1,2,3]

echo "[sbatch] end: $(date -Is)"
