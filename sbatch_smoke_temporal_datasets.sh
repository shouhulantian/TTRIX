#!/bin/bash
#SBATCH --partition=slowlane
#SBATCH --job-name=ttrix_smoke
#SBATCH --output=%u_job_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=A40:1
#SBATCH --time=00:30:00

set -euo pipefail
echo "[sbatch] node=$(hostname) job=$SLURM_JOB_ID"

module purge
module load Miniconda3
source "${EBROOTMINICONDA3}/bin/activate"
conda activate ultra_env

cd /mnt/nfs/home/ac139229/jiaxin/git/git/TTRIX

PYTHONPATH=src python - <<'PYEOF'
from trix.datasets import (
    GDELTIndT100Temporal, ICEWS14IndT100Temporal, ICEWS0515IndT100Temporal,
    TemporalICEWS14, TemporalICEWS0515, TemporalGDELT,
    JointTemporalDataset,
)

ROOT = "/mnt/nfs/home/ac139229/jiaxin/git/git/TTRIX/datasets"

def report(name, ds):
    train, valid, test = ds[0], ds[1], ds[2]
    print(f"\n=== {name} ===")
    for split, d in [("train", train), ("valid", valid), ("test", test)]:
        nt = int(d.num_time.item()) if hasattr(d, "num_time") and d.num_time is not None else None
        et = d.edge_time.shape if hasattr(d, "edge_time") and d.edge_time is not None else None
        tet = d.target_edge_time.shape if hasattr(d, "target_edge_time") and d.target_edge_time is not None else None
        print(f"  {split}: nodes={d.num_nodes} rels={d.num_relations} num_time={nt} "
              f"edge_index={tuple(d.edge_index.shape)} edge_time={et} target_edge_time={tet}")

# Inductive temporal (4-file INGRAM)
for cls, label in [(GDELTIndT100Temporal, "GDELTIndT100Temporal"),
                   (ICEWS14IndT100Temporal, "ICEWS14IndT100Temporal"),
                   (ICEWS0515IndT100Temporal, "ICEWS0515IndT100Temporal")]:
    try:
        ds = cls(root=ROOT)
        report(label, ds)
    except Exception as e:
        print(f"\n=== {label} FAILED: {type(e).__name__}: {e} ===")

# Transductive temporal
for cls, label in [(TemporalICEWS14, "TemporalICEWS14"),
                   (TemporalICEWS0515, "TemporalICEWS0515"),
                   (TemporalGDELT, "TemporalGDELT")]:
    try:
        ds = cls(root=ROOT)
        report(label, ds)
    except Exception as e:
        print(f"\n=== {label} FAILED: {type(e).__name__}: {e} ===")

# Joint temporal
try:
    jds = JointTemporalDataset(root=ROOT, graphs=["ICEWS14", "ICEWS0515"])
    tr, va, te = jds._data
    print(f"\n=== JointTemporalDataset(['ICEWS14','ICEWS0515']) ===")
    for i, (a, b, c) in enumerate(zip(tr, va, te)):
        nt = int(a.num_time.item()) if hasattr(a, "num_time") else None
        print(f"  graph[{i}]: nodes={a.num_nodes} rels={a.num_relations} num_time={nt} "
              f"train_edges={a.target_edge_index.shape[1]} valid={b.target_edge_index.shape[1]} test={c.target_edge_index.shape[1]}")
except Exception as e:
    print(f"\n=== JointTemporalDataset FAILED: {type(e).__name__}: {e} ===")
    import traceback; traceback.print_exc()

print("\nSMOKE OK")
PYEOF
