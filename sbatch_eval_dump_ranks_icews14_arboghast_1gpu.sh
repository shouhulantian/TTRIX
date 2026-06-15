#!/bin/bash
#SBATCH --partition=arboghast
#SBATCH --nodelist=aisa-arboghast01
#SBATCH --output=%u_job_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=A100:1
#SBATCH --time=00:45:00

# 1-GPU version: avoids the distributed-launcher rendezvous hang we hit
# on arboghast01 with the 4-GPU launcher (job 32910). ICEWS14 test is
# small (8963 queries) so 1 GPU is fine.
# Submit via:
#   sbatch --export=CFG=eval_dump_27364_ep4_distmult_icews14.yaml \
#       --job-name=evdmp_distmult sbatch_eval_dump_ranks_icews14_arboghast_1gpu.sh
#   sbatch --export=CFG=eval_dump_27204_ep3_rope_q_icews14.yaml \
#       --job-name=evdmp_rope_q sbatch_eval_dump_ranks_icews14_arboghast_1gpu.sh

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

# No torch.distributed.launch -- run directly. eval_dump_ranks.py's
# util.get_world_size() returns 1 in this case and the DistributedSampler
# falls back to local-only iteration.
PYTHONPATH=src python src/eval_dump_ranks.py \
    -c "config/${CFG}" \
    --gpus [0]

echo "[sbatch] end: $(date -Is)"
