#!/bin/bash
#SBATCH --partition=arboghast
#SBATCH --nodelist=aisa-arboghast01
#SBATCH --output=%u_job_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --gpus=A100:4
#SBATCH --time=01:00:00

# Same as sbatch_eval_dump_ranks_icews14_3gpu.sh but pinned to arboghast01
# (free node) and 4 GPUs. Submit via:
#   sbatch --export=CFG=eval_dump_27364_ep4_distmult_icews14.yaml \
#       --job-name=evdmp_distmult sbatch_eval_dump_ranks_icews14_arboghast_4gpu.sh
#   sbatch --export=CFG=eval_dump_27204_ep3_rope_q_icews14.yaml \
#       --job-name=evdmp_rope_q sbatch_eval_dump_ranks_icews14_arboghast_4gpu.sh

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
PYTHONPATH=src python -m torch.distributed.launch --nproc_per_node=4 --master_port=$MASTER_PORT src/eval_dump_ranks.py \
    -c "config/${CFG}" \
    --gpus [0,1,2,3]

echo "[sbatch] end: $(date -Is)"
