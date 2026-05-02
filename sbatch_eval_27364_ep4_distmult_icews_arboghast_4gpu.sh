#!/bin/bash
#SBATCH --partition=arboghast
#SBATCH --output=%u_job_%j.out
#SBATCH --nodelist=aisa-arboghast01
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --gpus=A100:4
#SBATCH --time=04:00:00

# Submit via:
#   sbatch --export=CFG=eval_27364_ep4_distmult_icews0515.yaml \
#       --job-name=ev27364_ep4_0515_arb \
#       sbatch_eval_27364_ep4_distmult_icews_arboghast_4gpu.sh
#
# Replaces the contention-stuck slowlane runs (27489/27495) with an
# arboghast slot. arboghast was the node that ran 27367 (RoPE_q ×
# ICEWS0515) cleanly in 1h24m. bs=8 fits on 40 GB A100 for vanilla
# TRIX (distmult is lighter than RoPE2_decay_q).

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
PYTHONPATH=src python -m torch.distributed.launch --nproc_per_node=4 --master_port=$MASTER_PORT src/run_entity.py \
    -c "config/${CFG}" \
    --gpus [0,1,2,3]

echo "[sbatch] end: $(date -Is)"
