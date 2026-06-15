#!/bin/bash
#SBATCH --partition=slowlane
#SBATCH --nodelist=aisa-gpuB02
#SBATCH --output=%u_job_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --gpus=A40:4
#SBATCH --time=00:45:00

# 4-GPU eval_dump_ranks.py on B02 (A40 x 8, currently idle). Submit both
# the distmult and the rope_q configs; B02 has 8 GPUs, so both 4-GPU
# jobs can run in parallel.
#   sbatch --export=CFG=eval_dump_27364_ep4_distmult_icews14.yaml \
#       --job-name=evdmp_distmult sbatch_eval_dump_ranks_icews14_B02_4gpu.sh
#   sbatch --export=CFG=eval_dump_27204_ep3_rope_q_icews14.yaml \
#       --job-name=evdmp_rope_q sbatch_eval_dump_ranks_icews14_B02_4gpu.sh

set -euo pipefail
echo "[sbatch] node=$(hostname) job=$SLURM_JOB_ID cfg=${CFG}"
echo "[sbatch] start: $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if command -v module >/dev/null 2>&1; then
    module purge
    module load Miniconda3 2>/dev/null && source "${EBROOTMINICONDA3}/bin/activate" && conda activate ultra_env
fi
export PATH="/mnt/nfs/home/ac139229/.conda/envs/ultra_env/bin:${PATH}"

export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

REPO=/mnt/nfs/home/ac139229/jiaxin/git/git/TTRIX

# Per-job cwd so the rank-0 -> rank-N working_dir.tmp coordination file
# doesn't collide with a sibling DDP launcher on the same node (B02 has
# 8 GPUs so we want both 4-GPU jobs to run in parallel without trampling
# each other's working_dir.tmp -- learned from 32923 failure).
STAGE_DIR=$(mktemp -d -p "$REPO" stage_${SLURM_JOB_ID}_XXXX)
cd "$STAGE_DIR"

MASTER_PORT=$((29500 + RANDOM % 1000))
PYTHONPATH="$REPO/src" python -m torch.distributed.launch --nproc_per_node=4 --master_port=$MASTER_PORT "$REPO/src/eval_dump_ranks.py" \
    -c "$REPO/config/${CFG}" \
    --gpus [0,1,2,3]

# Best-effort cleanup of the empty staging dir (the actual outputs are
# under cfg.output_dir, not here).
cd "$REPO"
rmdir "$STAGE_DIR" 2>/dev/null || true

echo "[sbatch] end: $(date -Is)"
