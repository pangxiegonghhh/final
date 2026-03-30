# scripts/p2_collect_and_compare.py
# -*- coding: utf-8 -*-
"""
P2 Step 4: 汇总所有策略结果，与 P1 最佳单 prompt 比较
输出: 总表 (CSV/MD) + 柱状图 (PNG/SVG) + 配对检验 (JSON/MD)
"""
import argparse, os, json, glob
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# P1 最佳单 prompt 的结果路径
BEST_SINGLE = {
    "ccnews": "outputs/p1_sensitivity/metrics/ccnews_P4_formal_LR.json",
    "squad":  "outputs/p1_sensitivity/metrics/squad_P5_concise_LR.json",
}

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def paired_ttest(a, b):
    t_stat, p_val = stats.ttest_rel(a, b)
    diff = np.array(a) - np.array(b)
    ci = stats.t.interval(0.95, len(diff)-1,
                          loc=np.mean(diff), scale=stats.sem(diff))
    return float(t_stat), float(p_val), (float(ci[0]), float(ci[1]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", default=r"outputs\p2_ensemble\metrics")
    parser.add_argument("--out_dir", default=r"outputs\p2_ensemble")
    args = parser.parse_args()

    tables_dir = os.path.join(args.out_dir, "tables")
    figures_dir = os.path.join(args.out_dir, "figures")
    stats_dir = os.path.join(args.out_dir, "stats")
    for d in [tables_dir, figures_dir, stats_dir]:
        os.makedirs(d, exist_ok=True)

    # 收集所有结果
    all_results = []
    for jf in sorted(glob.glob(os.path.join(args.metrics_dir, "*.json"))):
        r = load_json(jf)
        # 从 tag 解析 domain 和 strategy
        tag = r.get("tag", os.path.basename(jf).replace(".json", ""))
        parts = tag.split("_", 1)
        domain = parts[0]
        strategy = parts[1] if len(parts) > 1 else tag
        r["domain"] = domain
        r["strategy"] = strategy
        all_results.append(r)

    # 加载 P1 最佳单 prompt 结果作为 baseline
    baselines = {}
    for domain, path in BEST_SINGLE.items():
        if os.path.exists(path):
            baselines[domain] = load_json(path)
        else:
            print(f"  Warning: P1 baseline not found: {path}")

    # === 输出总表 ===
    rows = []
    header = "| Domain | Strategy | Dims | F1 (mean±std) | AUROC (mean±std) | Cost |"
    sep = "|--------|----------|------|---------------|------------------|------|"
    rows.append(header)
    rows.append(sep)

    # 先加 P1 baselines
    for domain in ["ccnews", "squad"]:
        if domain in baselines:
            b = baselines[domain]
            prompt_name = "P4_formal" if domain == "ccnews" else "P5_concise"
            rows.append(
                f"| {domain} | Best-Single ({prompt_name}) | 5 | "
                f"{b['f1_mean']:.4f}±{b['f1_std']:.4f} | "
                f"{b['auroc_mean']:.4f}±{b['auroc_std']:.4f} | 1 |"
            )

    for r in sorted(all_results, key=lambda x: (x["domain"], x["strategy"])):
        cost = 7 if "top3" not in r["strategy"] else 3
        rows.append(
            f"| {r['domain']} | {r['strategy']} | {r['dims']} | "
            f"{r['f1_mean']:.4f}±{r['f1_std']:.4f} | "
            f"{r['auroc_mean']:.4f}±{r['auroc_std']:.4f} | {cost} |"
        )

    table_md = "\n".join(rows)
    with open(os.path.join(tables_dir, "p2_ensemble_summary.md"), "w", encoding="utf-8") as f:
        f.write("# P2 Ensemble Results Summary\n\n")
        f.write(table_md)
    print(table_md)

    # === 配对 t-test ===
    paired_results = []
    for r in all_results:
        domain = r["domain"]
        if domain not in baselines:
            continue
        b = baselines[domain]
        t_stat, p_val, ci = paired_ttest(r["f1_list"], b["f1_list"])
        diff_mean = r["f1_mean"] - b["f1_mean"]
        pr = {
            "domain": domain,
            "strategy": r["strategy"],
            "baseline": f"Best-Single",
            "f1_diff": float(diff_mean),
            "t_stat": t_stat,
            "p_val": p_val,
            "ci_95": ci,
            "significant": p_val < 0.05,
        }
        paired_results.append(pr)
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
        print(f"  [{domain}] {r['strategy']} vs Best-Single: "
              f"ΔF1={diff_mean:+.4f}, t={t_stat:.3f}, p={p_val:.4f} {sig}")

    with open(os.path.join(stats_dir, "p2_paired_tests.json"), "w", encoding="utf-8") as f:
        json.dump(paired_results, f, indent=2, ensure_ascii=False)

    # === 柱状图 ===
    for domain in ["ccnews", "squad"]:
        domain_results = [r for r in all_results if r["domain"] == domain]
        if domain in baselines:
            b = baselines[domain]
            prompt_name = "P4_formal" if domain == "ccnews" else "P5_concise"
            domain_results.insert(0, {
                "strategy": f"Best-Single\n({prompt_name})",
                "f1_mean": b["f1_mean"],
                "f1_std": b["f1_std"],
            })

        names = [r["strategy"].replace("ensemble_", "Ens-") for r in domain_results]
        means = [r["f1_mean"] for r in domain_results]
        stds = [r["f1_std"] for r in domain_results]

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(names))
        colors = ["#aaaaaa"] + ["#4C72B0", "#55A868", "#C44E52"][:len(domain_results)-1]
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=9)
        ax.set_ylabel("F1 Score")
        ax.set_title(f"P2 Ensemble vs Best Single Prompt — {domain.upper()}")
        ax.set_ylim(min(means) - 0.05, max(means) + 0.05)
        ax.grid(axis="y", alpha=0.3)

        for ext in ["png", "svg"]:
            fig.savefig(os.path.join(figures_dir, f"p2_ensemble_{domain}.{ext}"),
                       dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Figure saved: p2_ensemble_{domain}.png/svg")

    print(f"\nAll P2 outputs saved to {args.out_dir}")

if __name__ == "__main__":
    main()