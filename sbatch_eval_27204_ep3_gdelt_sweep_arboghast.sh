#!/bin/bash
#SBATCH --partition=arboghast
#SBATCH --output=%u_job_%j.out
#SBATCH --nodelist=aisa-arboghast01
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --gpus=A100:4
#SBATCH --time=02:00:00

# TRIX 27204 ep3 ckpt eval on a GDELT inductive sweep variant.
# Submit with --export=DATASET=GDELTIndT_25_inter_Temporal (etc).
# bs=16 is conservative; arboghast 40 GB A100 should fit much higher
# given GDELT's tiny 272-entity all_negative footprint.

set -euo pipefail
echo "[sbatch] node=$(hostname) job=$SLURM_JOB_ID dataset=${DATASET}"
echo "[sbatch] start: $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

module purge
module load Miniconda3
source "${EBROOTMINICONDA3}/bin/activate"
conda activate ultra_env

export OMP_NUM_THREADS=16
export PYTHONUNBUFFERED=1

REPO=/mnt/nfs/home/ac139229/jiaxin/git/git/TTRIX
cd "$REPO"

MASTER_PORT=$((29500 + RANDOM % 1000))
PYTHONPATH=src python -m torch.distributed.launch --nproc_per_node=4 --master_port=$MASTER_PORT src/run_entity.py \
    -c config/eval_27204_ep3_gdelt_sweep.yaml \
    --dataset "${DATASET}" \
    --gpus [0,1,2,3]

echo "[sbatch] end: $(date -Is)"
