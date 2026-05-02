#!/bin/bash
#SBATCH --partition=slowlane
#SBATCH --job-name=ev27204_ep9_wiki25in
#SBATCH --output=%u_job_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=16
#SBATCH --mem=144G
#SBATCH --gpus=A100:3
#SBATCH --time=01:30:00

set -euo pipefail
echo "[sbatch] node=$(hostname) job=$SLURM_JOB_ID gpus=$SLURM_GPUS_ON_NODE"
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
PYTHONPATH=src python -m torch.distributed.launch --nproc_per_node=3 --master_port=$MASTER_PORT src/run_entity.py \
    -c config/eval_27204_ep9_wikiind_25_inter.yaml \
    --gpus [0,1,2]

echo "[sbatch] end: $(date -Is)"
