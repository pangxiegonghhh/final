# scripts/p1_sensitivity_analysis.py
# -*- coding: utf-8 -*-
"""
P1 Step 4: 汇总逐 prompt 结果，画敏感性图，做 ANOVA 方差分析。
"""
import argparse, json, os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_dir", default="outputs/p1_sensitivity/metrics")
    ap.add_argument("--out_dir", default="outputs/p1_sensitivity")
    args = ap.parse_args()

    # 收集所有单分类器结果
    rows = []
    for fp in sorted(glob.glob(os.path.join(args.metrics_dir, "*_LR.json"))):
        data = json.load(open(fp, "r", encoding="utf-8"))
        parts = data["dataset"].split("_", 1)
        domain = parts[0]
        prompt_id = parts[1] if len(parts) > 1 else "P0_baseline"
        rows.append({
            "domain": domain,
            "prompt_id": prompt_id,
            "acc_mean": data["accuracy_mean"],
            "acc_std": data["accuracy_std"],
            "f1_mean": data["f1_mean"],
            "f1_std": data["f1_std"],
            "auroc_mean": data["auroc_mean"],
            "auroc_std": data["auroc_std"],
            "f1_list": data["f1_list"],
            "acc_list": data["accuracy_list"],
        })

    df = pd.DataFrame(rows)

    # 保存总表
    tables_dir = os.path.join(args.out_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    display = df[["domain", "prompt_id", "acc_mean", "acc_std",
                   "f1_mean", "f1_std", "auroc_mean", "auroc_std"]].copy()
    display.to_csv(os.path.join(tables_dir, "p1_prompt_sensitivity.csv"), index=False)

    # Markdown
    md_lines = []
    md_lines.append("| Domain | Prompt | Acc | F1 | AUROC |")
    md_lines.append("| --- | --- | --- | --- | --- |")
    for _, r in display.iterrows():
        md_lines.append(f"| {r['domain']} | {r['prompt_id']} | "
                        f"{r['acc_mean']:.4f}±{r['acc_std']:.4f} | "
                        f"{r['f1_mean']:.4f}±{r['f1_std']:.4f} | "
                        f"{r['auroc_mean']:.4f}±{r['auroc_std']:.4f} |")
    with open(os.path.join(tables_dir, "p1_prompt_sensitivity.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    # 画柱状图
    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    for domain in df["domain"].unique():
        sub = df[df["domain"] == domain].sort_values("f1_mean", ascending=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(sub))
        ax.bar(x, sub["f1_mean"].values, yerr=sub["f1_std"].values,
               capsize=4, color="steelblue")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["prompt_id"].values, rotation=30, ha="right")
        ax.set_ylabel("F1 (mean +/- std)")
        ax.set_title(f"Prompt Sensitivity: {domain} (LR, 10-seed)")
        ax.set_ylim(0.5, 1.0)
        plt.tight_layout()
        for ext in ["png", "svg"]:
            plt.savefig(os.path.join(fig_dir, f"p1_sensitivity_{domain}.{ext}"), dpi=300)
        plt.close()

    # ANOVA
    print("\n=== ANOVA (F1 across prompts) ===")
    anova_results = []
    for domain in df["domain"].unique():
        sub = df[df["domain"] == domain]
        groups = [r["f1_list"] for _, r in sub.iterrows()]
        if len(groups) >= 2:
            F_stat, p_val = stats.f_oneway(*groups)
            print(f"  {domain}: F={F_stat:.4f}, p={p_val:.6f}")
            anova_results.append({"domain": domain, "F": float(F_stat), "p": float(p_val)})

    with open(os.path.join(tables_dir, "p1_anova.json"), "w", encoding="utf-8") as f:
        json.dump(anova_results, f, indent=2)

    print(f"\nAll outputs saved to {args.out_dir}")


if __name__ == "__main__":
    main()
