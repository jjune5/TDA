"""실험 (f) RGCN + GTN-PDGNN 결과 집계.

runs/campaign/{ds}__f_rgcn__s{seed}/metrics.json → mean±std 표 출력
results/f_rgcn_results.md 저장.
"""
import json
import os
import statistics

SEEDS = [312132, 238623, 792965, 15092, 661491, 588722, 825661, 500973, 88015, 251219]
DATASETS = ["acm", "dblp", "imdb", "freebase", "mag", "aifb", "yelp"]
BASE = "runs/campaign"


def load(ds):
    recs = []
    for s in SEEDS:
        p = os.path.join(BASE, f"{ds}__f_rgcn__s{s}", "metrics.json")
        if os.path.exists(p):
            recs.append(json.load(open(p)))
    return recs


rows = []
for ds in DATASETS:
    recs = load(ds)
    if not recs:
        rows.append({"ds": ds, "n": 0, "f1": "–", "std": "–", "acc": "–"})
        continue
    f1s = [r["test_macro_f1"] for r in recs]
    accs = [r["test_accuracy"] for r in recs]
    rows.append({
        "ds": ds, "n": len(f1s),
        "f1": f"{statistics.mean(f1s):.4f}",
        "std": f"{statistics.stdev(f1s):.4f}" if len(f1s) > 1 else "–",
        "acc": f"{statistics.mean(accs):.4f}",
    })

print(f"\n{'데이터셋':<12} {'n':>4}  {'F1 mean':>8}  {'F1 std':>8}  {'Acc mean':>9}")
print("-" * 52)
for r in rows:
    print(f"{r['ds']:<12} {r['n']:>4}  {r['f1']:>8}  {r['std']:>8}  {r['acc']:>9}")

os.makedirs("results", exist_ok=True)
with open("results/f_rgcn_results.md", "w") as f:
    f.write("# 실험 (f): RGCN + GTN-PDGNN (7 datasets × 10 seeds)\n\n")
    f.write("| 데이터셋 | n | F1 mean | F1 std | Acc mean |\n")
    f.write("|---------|---|---------|--------|----------|\n")
    for r in rows:
        f.write(f"| {r['ds']} | {r['n']} | {r['f1']} | {r['std']} | {r['acc']} |\n")
print("\n[saved] results/f_rgcn_results.md")
