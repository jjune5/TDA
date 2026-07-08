#!/bin/bash
# RGCN ablation matrix (seed 0~9)
#
# (a) RGCN only              — backbone only, no topology
# (b) RGCN + PDGNN(manual)   — predefined metapath topology
# (c) RGCN + GTN-PDGNN       — learned metapath topology
#
# 실행: bash experiments/run_rgcn_ablation.sh [dataset] [config]
# 기본: dataset=acm, config=configs/acm.json

DATASET="${1:-acm}"
CONFIG="${2:-configs/acm.json}"
SEEDS=$(seq 0 9)

echo "[rgcn ablation] dataset=${DATASET} config=${CONFIG}"

for seed in $SEEDS; do
  echo "=== seed ${seed} / (a) RGCN only ==="
  python -m tda.train --config "${CONFIG}" --dataset "${DATASET}" \
    --backbone rgcn --no-topology \
    --seed "${seed}" --output-dir "runs/rgcn_a/seed${seed}"

  echo "=== seed ${seed} / (b) RGCN + PDGNN (manual) ==="
  python -m tda.train --config "${CONFIG}" --dataset "${DATASET}" \
    --backbone rgcn --topology-source manual \
    --seed "${seed}" --output-dir "runs/rgcn_b/seed${seed}"

  echo "=== seed ${seed} / (c) RGCN + GTN-PDGNN ==="
  python -m tda.train --config "${CONFIG}" --dataset "${DATASET}" \
    --backbone rgcn --topology-source gtn \
    --seed "${seed}" --output-dir "runs/rgcn_c/seed${seed}"
done

echo "[done] 결과 집계:"
python experiments/collect_rgcn_results.py --dataset "${DATASET}"
