#!/bin/bash
#SBATCH --partition=slowlane
#SBATCH --job-name=trix_decayq_zs_gdelt
#SBATCH --output=%u_job_%j.out
#SBATCH --nodes=1
#SBATCH --nodelist=aisa-gpuB03
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --gpus=A40:4
#SBATCH --time=04:00:00

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

# Best-by-val-MRR ckpt from job 27037 (T-RoPE-Gq ICEWS14 dim=64).
# Best epoch was ep_8 with val MRR 0.6200, saved as model_epoch_9.pth
# (TRIX names ckpts as model_epoch_<X+1>.pth where X is the just-trained
# epoch index).
CKPT=/mnt/nfs/home/ac139229/jiaxin/git/git/TTRIX/output/TRIX/TemporalICEWS14/2026-04-28-22-34-18/model_epoch_9.pth

MASTER_PORT=$((29500 + RANDOM % 1000))
PYTHONPATH=src python -m torch.distributed.launch --nproc_per_node=4 --master_port=$MASTER_PORT src/run_entity.py \
    -c config/eval_gdelt_indt100_rope2_decay_q.yaml \
    --gpus [0,1,2,3] \
    --ckpt $CKPT

echo "[sbatch] end: $(date -Is)"
