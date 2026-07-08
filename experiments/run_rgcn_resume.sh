#!/bin/bash
# 완료된 seed는 건너뛰고 나머지만 실행 (resume용)
DATASET="${1:-acm}"
CONFIG="${2:-configs/acm.json}"

cd /teamspace/studios/this_studio/repos/TDA

for seed in $(seq 0 9); do
  # (a) RGCN only
  OUT_A="runs/rgcn_a/seed${seed}/metrics.json"
  if [ -f "$OUT_A" ]; then
    echo "[skip] (a) seed${seed} already done"
  else
    echo "=== seed ${seed} / (a) RGCN only ==="
    python -m tda.train --config "${CONFIG}" --dataset "${DATASET}" \
      --backbone rgcn --no-topology \
      --seed "${seed}" --output-dir "runs/rgcn_a/seed${seed}"
  fi

  # (b) RGCN + PDGNN (manual)
  OUT_B="runs/rgcn_b/seed${seed}/metrics.json"
  if [ -f "$OUT_B" ]; then
    echo "[skip] (b) seed${seed} already done"
  else
    echo "=== seed ${seed} / (b) RGCN + PDGNN (manual) ==="
    python -m tda.train --config "${CONFIG}" --dataset "${DATASET}" \
      --backbone rgcn --topology-source manual \
      --seed "${seed}" --output-dir "runs/rgcn_b/seed${seed}"
  fi

  # (c) RGCN + GTN-PDGNN
  OUT_C="runs/rgcn_c/seed${seed}/metrics.json"
  if [ -f "$OUT_C" ]; then
    echo "[skip] (c) seed${seed} already done"
  else
    echo "=== seed ${seed} / (c) RGCN + GTN-PDGNN ==="
    python -m tda.train --config "${CONFIG}" --dataset "${DATASET}" \
      --backbone rgcn --topology-source gtn \
      --seed "${seed}" --output-dir "runs/rgcn_c/seed${seed}"
  fi
done

echo "[done] 결과 집계:"
python experiments/collect_rgcn_results.py --dataset "${DATASET}"
