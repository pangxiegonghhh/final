# scripts/p3_collect_and_compare.py
# -*- coding: utf-8 -*-
"""
P3 Step 3: 汇总路由结果，与 P1/P2 所有策略对比
输出: 全景总表 + 成本-性能曲线 + 配对检验
"""
import argparse, os, json
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# P1/P2 结果路径（根据你的实际文件调整）
PREV_RESULTS = {
    "ccnews": {
        "Best-Single": "outputs/p1_sensitivity/metrics/ccnews_P4_formal_LR.json",
        "Ens-Top3": "outputs/p2_ensemble/metrics/ccnews_ensemble_top3.json",
        "Ens-All": "outputs/p2_ensemble/metrics/ccnews_ensemble_all.json",
        "Ens-All+Cons": "outputs/p2_ensemble/metrics/ccnews_ensemble_all_cons.json",
    },
    "squad": {
        "Best-Single": "outputs/p1_sensitivity/metrics/squad_P5_concise_LR.json",
        "Ens-Top3": "outputs/p2_ensemble/metrics/squad_ensemble_top3.json",
        "Ens-All": "outputs/p2_ensemble/metrics/squad_ensemble_all.json",
        "Ens-All+Cons": "outputs/p2_ensemble/metrics/squad_ensemble_all_cons.json",
    },
}

COST_MAP = {
    "Random": 1, "Best-Single": 1, "Route-Top1": 1,
    "Route-Top2": 2, "Ens-Top3": 3, "Ens-All": 7, "Ens-All+Cons": 7,
    "Oracle-Top1": 1,
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
    parser.add_argument("--routing_dir", default=r"outputs\p3_routing\metrics")
    parser.add_argument("--out_dir", default=r"outputs\p3_routing")
    args = parser.parse_args()

    tables_dir = os.path.join(args.out_dir, "tables")
    figures_dir = os.path.join(args.out_dir, "figures")
    stats_dir = os.path.join(args.out_dir, "stats")
    for d in [tables_dir, figures_dir, stats_dir]:
        os.makedirs(d, exist_ok=True)

    all_paired = []

    for domain in ["ccnews", "squad"]:
        print(f"\n=== {domain} ===")

        # 加载路由结果
        routing = load_json(os.path.join(args.routing_dir, f"{domain}_routing.json"))

        # 收集所有策略
        strategies = {}

        # 路由策略
        strategies["Random"] = routing["random_top1"]["f1_list"]
        strategies["Route-Top1"] = routing["routing_top1"]["f1_list"]
        strategies["Route-Top2"] = routing["routing_top2"]["f1_list"]
        strategies["Oracle-Top1"] = routing["oracle_top1"]["f1_list"]

        # 加载 P1/P2 结果
        for name, path in PREV_RESULTS[domain].items():
            if os.path.exists(path):
                prev = load_json(path)
                strategies[name] = prev["f1_list"]
            else:
                print(f"  Warning: {path} not found, skipping {name}")

        # === 全景总表 ===
        print(f"\n  {'Strategy':<20} {'Cost':>4} {'F1 (mean±std)':>20}")
        print(f"  {'-'*20} {'-'*4} {'-'*20}")

        rows = []
        for name in ["Random", "Best-Single", "Route-Top1", "Route-Top2",
                      "Ens-Top3", "Ens-All", "Ens-All+Cons", "Oracle-Top1"]:
            if name not in strategies:
                continue
            fl = strategies[name]
            mean_f1 = float(np.mean(fl))
            std_f1 = float(np.std(fl))
            cost = COST_MAP.get(name, "?")
            print(f"  {name:<20} {cost:>4} {mean_f1:.4f}±{std_f1:.4f}")
            rows.append({
                "domain": domain, "strategy": name, "cost": cost,
                "f1_mean": mean_f1, "f1_std": std_f1, "f1_list": fl,
            })

        # === 配对检验 ===
        comparisons = [
            ("Route-Top1", "Random"),
            ("Route-Top1", "Best-Single"),
            ("Route-Top2", "Ens-Top3"),
            ("Route-Top2", "Ens-All"),
        ]

        for a_name, b_name in comparisons:
            if a_name not in strategies or b_name not in strategies:
                continue
            a_list = strategies[a_name]
            b_list = strategies[b_name]
            # 长度可能不同（P1/P2 用 80/20, P3 用 60/20/20）
            min_len = min(len(a_list), len(b_list))
            t_stat, p_val, ci = paired_ttest(a_list[:min_len], b_list[:min_len])
            diff = np.mean(a_list[:min_len]) - np.mean(b_list[:min_len])
            sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
            print(f"  {a_name} vs {b_name}: ΔF1={diff:+.4f}, t={t_stat:.3f}, p={p_val:.4f} {sig}")
            all_paired.append({
                "domain": domain, "a": a_name, "b": b_name,
                "diff": float(diff), "t": t_stat, "p": p_val, "ci": ci, "sig": sig,
            })

        # === 成本-性能曲线 ===
        fig, ax = plt.subplots(figsize=(8, 5))
        for r in rows:
            if r["strategy"] == "Oracle-Top1":
                # 用虚线标注上界
                ax.axhline(y=r["f1_mean"], color="gray", linestyle="--", alpha=0.5)
                ax.text(0.5, r["f1_mean"] + 0.005, "Oracle", fontsize=8, color="gray")
                continue
            marker = "o" if "Route" in r["strategy"] else ("s" if "Ens" in r["strategy"] else "^")
            color = {"Random": "#999999", "Best-Single": "#aaaaaa",
                     "Route-Top1": "#E74C3C", "Route-Top2": "#E74C3C",
                     "Ens-Top3": "#4C72B0", "Ens-All": "#4C72B0",
                     "Ens-All+Cons": "#2ECC71"}.get(r["strategy"], "black")
            ax.errorbar(r["cost"], r["f1_mean"], yerr=r["f1_std"],
                       marker=marker, color=color, markersize=8, capsize=4, linewidth=1.5)
            ax.annotate(r["strategy"], (r["cost"], r["f1_mean"]),
                       textcoords="offset points", xytext=(8, 5), fontsize=7)

        ax.set_xlabel("Rewrite Cost (number of prompt calls)")
        ax.set_ylabel("F1 Score")
        ax.set_title(f"Cost-Performance Tradeoff — {domain.upper()}")
        ax.set_xticks([1, 2, 3, 7])
        ax.grid(alpha=0.3)

        for ext in ["png", "svg"]:
            fig.savefig(os.path.join(figures_dir, f"p3_cost_performance_{domain}.{ext}"),
                       dpi=300, bbox_inches="tight")
        plt.close()

    # 保存全景总表
    with open(os.path.join(tables_dir, "p3_full_comparison.md"), "w", encoding="utf-8") as f:
        f.write("# P0-P3 Full Comparison\n\n")
        f.write("| Domain | Strategy | Cost | F1 (mean±std) |\n")
        f.write("|--------|----------|------|---------------|\n")
        # 重新遍历输出
        for domain in ["ccnews", "squad"]:
            routing = load_json(os.path.join(args.routing_dir, f"{domain}_routing.json"))
            entries = [
                ("Random", 1, routing["random_top1"]),
                ("Route-Top1", 1, routing["routing_top1"]),
                ("Route-Top2", 2, routing["routing_top2"]),
                ("Oracle-Top1", 1, routing["oracle_top1"]),
            ]
            for name, cost, data in entries:
                f.write(f"| {domain} | {name} | {cost} | "
                       f"{data['f1_mean']:.4f}±{data['f1_std']:.4f} |\n")
            # P1/P2
            for name, path in PREV_RESULTS[domain].items():
                if os.path.exists(path):
                    prev = load_json(path)
                    cost = COST_MAP.get(name, "?")
                    f.write(f"| {domain} | {name} | {cost} | "
                           f"{prev['f1_mean']:.4f}±{prev['f1_std']:.4f} |\n")

    # 保存配对检验
    with open(os.path.join(stats_dir, "p3_paired_tests.json"), "w", encoding="utf-8") as f:
        json.dump(all_paired, f, indent=2, ensure_ascii=False)

    print(f"\nAll outputs saved to {args.out_dir}")

if __name__ == "__main__":
    main()