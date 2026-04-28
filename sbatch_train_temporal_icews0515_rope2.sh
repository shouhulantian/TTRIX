#!/bin/bash
#SBATCH --partition=slowlane
#SBATCH --job-name=trix_rope2_0515
#SBATCH --output=%u_job_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --gpus=A40:4
#SBATCH --time=12:00:00

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
PYTHONPATH=src python -m torch.distributed.launch --nproc_per_node=4 --master_port=$MASTER_PORT src/run_entity.py \
    -c config/run_entity_temporal_icews0515_rope2.yaml \
    --gpus [0,1,2,3]

echo "[sbatch] end: $(date -Is)"
