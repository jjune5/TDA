#!/bin/bash
# 실험 (f): RGCN + GTN-PDGNN feature (learned meta-path)
# 7개 데이터셋 × 10 seeds, 완료된 run은 건너뜀.
#
# 실행: bash experiments/run_f_rgcn.sh [data-root]
# 기본 data-root: ./data

set -euo pipefail
DATA_ROOT="${1:-./data}"
SEEDS=(312132 238623 792965 15092 661491 588722 825661 500973 88015 251219)
DATASETS=(acm dblp imdb freebase mag aifb yelp)

cd "$(dirname "$0")/.."

echo "[exp-f] RGCN + GTN-PDGNN  |  datasets=${DATASETS[*]}  |  seeds=${SEEDS[*]}"

total=0 skip=0 run=0
for ds in "${DATASETS[@]}"; do
  cfg="configs/campaign/${ds}__f_rgcn.json"
  if [ ! -f "$cfg" ]; then
    echo "[warn] config not found: $cfg — skipping $ds"
    continue
  fi
  for seed in "${SEEDS[@]}"; do
    out="runs/campaign/${ds}__f_rgcn__s${seed}"
    total=$((total+1))
    if [ -f "${out}/metrics.json" ]; then
      echo "[skip] ${ds} seed=${seed}"
      skip=$((skip+1))
      continue
    fi
    echo "=== ${ds}  seed=${seed} ==="
    python -m tda.train \
      --config "$cfg" \
      --dataset "$ds" \
      --data-root "$DATA_ROOT" \
      --topology-source gtn \
      --seed "$seed" \
      --output-dir "$out"
    run=$((run+1))
  done
done

echo ""
echo "[exp-f] done.  total=${total}  skipped=${skip}  ran=${run}"
echo "[exp-f] 결과 집계:"
python experiments/collect_f_results.py
