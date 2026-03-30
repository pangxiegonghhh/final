# scripts/p0_export_results.py
# -*- coding: utf-8 -*-
"""
Step 6: 汇总两域三分类器结果，生成论文级总表（CSV + MD）与柱状图。

运行命令：
    python scripts\p0_export_results.py --metrics_dir outputs\p0_baseline\metrics
"""
import argparse, json, os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_all_results(metrics_dir):
    rows = []
    for fp in sorted(glob.glob(os.path.join(metrics_dir, "*_all_classifiers.json"))):
        data = json.load(open(fp, "r", encoding="utf-8"))
        for clf_name, s in data.items():
            rows.append({
                "Dataset": s["dataset"],
                "Classifier": clf_name,
                "Accuracy": f"{s['accuracy_mean']:.4f}±{s['accuracy_std']:.4f}",
                "F1": f"{s['f1_mean']:.4f}±{s['f1_std']:.4f}",
                "AUROC": f"{s['auroc_mean']:.4f}±{s['auroc_std']:.4f}",
                "TPR@FPR=1%": f"{s['tpr_at_fpr_1pct_mean']:.4f}±{s['tpr_at_fpr_1pct_std']:.4f}",
                "TPR@FPR=5%": f"{s['tpr_at_fpr_5pct_mean']:.4f}±{s['tpr_at_fpr_5pct_std']:.4f}",
                # 数值版本用于画图
                "_acc_mean": s["accuracy_mean"],
                "_acc_std": s["accuracy_std"],
                "_f1_mean": s["f1_mean"],
                "_f1_std": s["f1_std"],
                "_auroc_mean": s["auroc_mean"],
                "_auroc_std": s["auroc_std"],
            })
    return pd.DataFrame(rows)

def save_table(df, out_dir):
    display_cols = ["Dataset", "Classifier", "Accuracy", "F1", "AUROC", "TPR@FPR=1%", "TPR@FPR=5%"]
    df_display = df[display_cols]

    csv_path = os.path.join(out_dir, "p0_baseline_results.csv")
    df_display.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"saved: {csv_path}")

    # Markdown
    md_lines = ["| " + " | ".join(display_cols) + " |"]
    md_lines.append("| " + " | ".join(["---"] * len(display_cols)) + " |")
    for _, row in df_display.iterrows():
        md_lines.append("| " + " | ".join(str(row[c]) for c in display_cols) + " |")
    md_path = os.path.join(out_dir, "p0_baseline_results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"saved: {md_path}")

def plot_metrics(df, out_dir):
    fig_dir = os.path.join(out_dir, "..", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    datasets = df["Dataset"].unique()
    classifiers = df["Classifier"].unique()

    for metric, label in [("_f1", "F1"), ("_auroc", "AUROC"), ("_acc", "Accuracy")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(datasets))
        width = 0.25

        for j, clf in enumerate(classifiers):
            means = []
            stds = []
            for ds in datasets:
                row = df[(df["Dataset"] == ds) & (df["Classifier"] == clf)]
                if len(row) == 0:
                    means.append(0); stds.append(0)
                else:
                    means.append(row[f"{metric}_mean"].values[0])
                    stds.append(row[f"{metric}_std"].values[0])
            offset = (j - (len(classifiers) - 1) / 2) * width
            ax.bar(x + offset, means, width, label=clf, yerr=stds, capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_ylabel(label)
        ax.set_title(f"P0 Baseline: {label} by Dataset & Classifier")
        ax.legend()
        ax.set_ylim(0, 1.05)
        plt.tight_layout()

        for ext in ["png", "svg"]:
            fp = os.path.join(fig_dir, f"p0_{label.lower()}.{ext}")
            plt.savefig(fp, dpi=300)
        plt.close()
        print(f"saved: {fig_dir}/p0_{label.lower()}.png/svg")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_dir", default="outputs/p0_baseline/metrics")
    args = ap.parse_args()

    df = load_all_results(args.metrics_dir)
    if df.empty:
        print("No results found!")
        return

    save_table(df, args.metrics_dir)
    plot_metrics(df, args.metrics_dir)

if __name__ == "__main__":
    main()
