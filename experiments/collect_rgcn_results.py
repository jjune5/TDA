"""RGCN ablation 결과 집계.

runs/rgcn_{a,b,c}/seed{0..9}/metrics.json 을 읽어
mean ± std 표를 출력하고 results/rgcn_ablation.md 에 저장한다.

사용: python experiments/collect_rgcn_results.py [--dataset acm] [--runs-dir runs]
"""
import argparse
import json
import os
import statistics

CONDITIONS = [
    ("a", "RGCN only"),
    ("b", "RGCN + PDGNN (manual metapath)"),
    ("c", "RGCN + GTN-PDGNN (learned metapath)"),
]
METRICS = ["test_macro_f1", "test_accuracy", "val_macro_f1"]


def load_seeds(runs_dir: str, tag: str):
    results = []
    for seed_dir in sorted(os.listdir(os.path.join(runs_dir, f"rgcn_{tag}"))):
        path = os.path.join(runs_dir, f"rgcn_{tag}", seed_dir, "metrics.json")
        if os.path.exists(path):
            with open(path) as f:
                results.append(json.load(f))
    return results


def summarize(records, metric):
    vals = [r[metric] for r in records if metric in r]
    if not vals:
        return "N/A", "N/A", 0
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{mean:.4f}", f"{std:.4f}", len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="acm")
    ap.add_argument("--runs-dir", default="runs")
    args = ap.parse_args()

    rows = []
    for tag, label in CONDITIONS:
        cond_dir = os.path.join(args.runs_dir, f"rgcn_{tag}")
        if not os.path.isdir(cond_dir):
            print(f"[skip] {cond_dir} not found")
            continue
        records = load_seeds(args.runs_dir, tag)
        row = {"condition": label, "n": 0}
        for metric in METRICS:
            mean, std, n = summarize(records, metric)
            row[metric] = f"{mean} ± {std}"
            row["n"] = n
        rows.append(row)

    # 출력
    header = f"{'Condition':<40} {'n':>4} {'test_macro_f1':>18} {'test_accuracy':>18} {'val_macro_f1':>18}"
    sep = "-" * len(header)
    print(f"\n[RGCN ablation] dataset={args.dataset}")
    print(sep)
    print(header)
    print(sep)
    for row in rows:
        print(f"{row['condition']:<40} {row['n']:>4} "
              f"{row.get('test_macro_f1','N/A'):>18} "
              f"{row.get('test_accuracy','N/A'):>18} "
              f"{row.get('val_macro_f1','N/A'):>18}")
    print(sep)

    # 저장
    os.makedirs("results", exist_ok=True)
    out_path = f"results/rgcn_ablation_{args.dataset}.md"
    with open(out_path, "w") as f:
        f.write(f"# RGCN Ablation — {args.dataset}\n\n")
        f.write(f"| Condition | n | test Macro-F1 | test Accuracy | val Macro-F1 |\n")
        f.write(f"|-----------|---|--------------|--------------|-------------|\n")
        for row in rows:
            f.write(f"| {row['condition']} | {row['n']} "
                    f"| {row.get('test_macro_f1','N/A')} "
                    f"| {row.get('test_accuracy','N/A')} "
                    f"| {row.get('val_macro_f1','N/A')} |\n")
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
