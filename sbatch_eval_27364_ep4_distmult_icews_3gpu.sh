#!/bin/bash
#SBATCH --partition=slowlane
#SBATCH --output=%u_job_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=16
#SBATCH --mem=144G
#SBATCH --gpus=A100:3
#SBATCH --time=02:00:00

# Submit via:
#   sbatch --export=CFG=eval_27364_ep4_distmult_icews14.yaml \
#       --job-name=ev27364_ep4_14 sbatch_eval_27364_ep4_distmult_icews_3gpu.sh
#   sbatch --export=CFG=eval_27364_ep4_distmult_icews0515.yaml \
#       --job-name=ev27364_ep4_0515 sbatch_eval_27364_ep4_distmult_icews_3gpu.sh

set -euo pipefail
echo "[sbatch] node=$(hostname) job=$SLURM_JOB_ID cfg=${CFG}"
echo "[sbatch] start: $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if command -v module >/dev/null 2>&1; then
    module purge
    module load Miniconda3 2>/dev/null && source "${EBROOTMINICONDA3}/bin/activate" && conda activate ultra_env
fi
export PATH="/mnt/nfs/home/ac139229/.conda/envs/ultra_env/bin:${PATH}"

export OMP_NUM_THREADS=16
export PYTHONUNBUFFERED=1

REPO=/mnt/nfs/home/ac139229/jiaxin/git/git/TTRIX
cd "$REPO"

MASTER_PORT=$((29500 + RANDOM % 1000))
PYTHONPATH=src python -m torch.distributed.launch --nproc_per_node=3 --master_port=$MASTER_PORT src/run_entity.py \
    -c "config/${CFG}" \
    --gpus [0,1,2]

echo "[sbatch] end: $(date -Is)"
