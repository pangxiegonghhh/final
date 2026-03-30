import os, json, glob, math
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.abspath(os.path.dirname(__file__) + r"\..")
OUT10 = os.path.join(ROOT, "outputs", "10seed测试")
OUT_PAPER = os.path.join(ROOT, "outputs", "paper_tables")
OUT_FIG = os.path.join(ROOT, "outputs", "figures")
OUT_STATS = os.path.join(ROOT, "outputs", "stats")
os.makedirs(OUT_PAPER, exist_ok=True)
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_STATS, exist_ok=True)

# ---- configure paths ----
# C1 baseline: you already have these
BASELINE = {
    ("Arxiv","repo"):     os.path.join(OUT10, "arxiv_repo_350.json"),
    ("Arxiv","deepseek"): os.path.join(OUT10, "arxiv_deepseek_350.json"),
    ("Code","repo"):      os.path.join(OUT10, "code_repo_164.json"),
    ("Code","deepseek"):  os.path.join(OUT10, "code_deepseek_164.json"),
    ("Yelp","repo"):      os.path.join(OUT10, "yelp_repo_401.json"),
    ("Yelp","deepseek"):  os.path.join(OUT10, "yelp_deepseek_401.json"),
}

# C2 top3 outputs
TOP3 = {
    ("Arxiv","repo"):     os.path.join(OUT10, "prompt_ensemble", "Arxiv", "repo", "top3.json"),
    ("Arxiv","deepseek"): os.path.join(OUT10, "prompt_ensemble", "Arxiv", "deepseek", "top3.json"),
    ("Code","repo"):      os.path.join(OUT10, "prompt_ensemble", "Code", "repo", "top3.json"),
    ("Code","deepseek"):  os.path.join(OUT10, "prompt_ensemble", "Code", "deepseek", "top3.json"),
    ("Yelp","repo"):      os.path.join(OUT10, "prompt_ensemble", "Yelp", "repo", "top3.json"),
    ("Yelp","deepseek"):  os.path.join(OUT10, "prompt_ensemble", "Yelp", "deepseek", "top3.json"),
}

PROMPT_COUNT = {"ALL": {"Arxiv": 7, "Code": 5, "Yelp": 7}, "TOP3": 3}

def load_ms(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    j = json.load(open(path, "r", encoding="utf-8"))
    ms = j.get("multi_seed_summary", None)
    if ms is None:
        raise RuntimeError(f"missing multi_seed_summary in {path}")
    return ms

def fmt(m, s):
    return f"{m:.4f} ± {s:.4f}"

def paired_stats(a, b):
    # diff = TOP3 - ALL
    a = np.array(a, float); b = np.array(b, float)
    d = b - a
    n = len(d)

    tt = stats.ttest_rel(b, a, alternative="two-sided")
    ww = stats.wilcoxon(b, a, alternative="two-sided", zero_method="wilcox", method="auto")

    sd = float(np.std(d, ddof=1)) if n > 1 else 0.0
    dz = float(np.mean(d) / sd) if sd > 0 else float("nan")

    boot = stats.bootstrap((d,), np.mean, confidence_level=0.95, n_resamples=20000,
                           method="BCa", random_state=0)
    ci_low = float(boot.confidence_interval.low)
    ci_high = float(boot.confidence_interval.high)

    return {
        "n": int(n),
        "diff_mean": float(np.mean(d)),
        "diff_std": sd,
        "ttest_p": float(tt.pvalue),
        "wilcoxon_p": float(ww.pvalue),
        "cohen_dz": dz,
        "boot_ci95": {"low": ci_low, "high": ci_high},
    }

rows = []
stats_out = {"meta": {"compare": "Top3 - All", "bootstrap_resamples": 20000}, "groups": {}}

for (dom, rw), p_all in BASELINE.items():
    p_top3 = TOP3[(dom, rw)]
    ms_all = load_ms(p_all)
    ms_top = load_ms(p_top3)

    # seeds alignment
    if ms_all["seeds"] != ms_top["seeds"]:
        raise RuntimeError(f"seed mismatch: {(dom,rw)}")

    # summary rows (paper)
    rows.append({
        "Domain": dom,
        "Rewriter": rw,
        "Setting": "ALL",
        "#Prompts": PROMPT_COUNT["ALL"][dom],
        "Accuracy(mean±std)": fmt(ms_all["acc_mean"], ms_all["acc_std"]),
        "F1(mean±std)": fmt(ms_all["f1_mean"], ms_all["f1_std"]),
        "Path": p_all,
    })
    rows.append({
        "Domain": dom,
        "Rewriter": rw,
        "Setting": "TOP3",
        "#Prompts": PROMPT_COUNT["TOP3"],
        "Accuracy(mean±std)": fmt(ms_top["acc_mean"], ms_top["acc_std"]),
        "F1(mean±std)": fmt(ms_top["f1_mean"], ms_top["f1_std"]),
        "Path": p_top3,
    })

    # paired stats on seed lists (top3 - all)
    acc_stat = paired_stats(ms_all["acc_list"], ms_top["acc_list"])
    f1_stat  = paired_stats(ms_all["f1_list"],  ms_top["f1_list"])
    stats_out["groups"][f"{dom}/{rw}"] = {"acc": acc_stat, "f1": f1_stat, "paths": {"all": p_all, "top3": p_top3}}

df = pd.DataFrame(rows).sort_values(["Domain","Rewriter","Setting"]).reset_index(drop=True)

# ---- write table files ----
csv_path = os.path.join(OUT_PAPER, "stageC_all_vs_top3_summary.csv")
df[["Domain","Rewriter","Setting","#Prompts","Accuracy(mean±std)","F1(mean±std)"]].to_csv(csv_path, index=False, encoding="utf-8-sig")

pipe = chr(124)
md_path = os.path.join(OUT_PAPER, "stageC_all_vs_top3_summary.md")
md = []
md.append("# Stage C Summary: ALL prompts vs TOP-3 prompts")
md.append("")
md.append(f"{pipe} Domain {pipe} Rewriter {pipe} Setting {pipe} #Prompts {pipe} Accuracy (mean±std) {pipe} F1 (mean±std) {pipe}")
md.append(f"{pipe}---{pipe}---{pipe}---{pipe}---:{pipe}---:{pipe}---:{pipe}")
for rec in df.to_dict("records"):
    md.append(f"{pipe} {rec['Domain']} {pipe} {rec['Rewriter']} {pipe} {rec['Setting']} {pipe} {int(rec['#Prompts'])} {pipe} {rec['Accuracy(mean±std)']} {pipe} {rec['F1(mean±std)']} {pipe}")
open(md_path, "w", encoding="utf-8").write("\n".join(md))

# ---- write paired stats ----
stats_json = os.path.join(OUT_STATS, "stageC_paired_all_vs_top3.json")
json.dump(stats_out, open(stats_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

stats_md = os.path.join(OUT_PAPER, "stageC_paired_all_vs_top3.md")
md2 = []
md2.append("# Stage C Paired Tests: TOP-3 minus ALL (10 seeds paired)")
md2.append("")
md2.append("说明：差值定义为 TOP3 - ALL；paired t-test + Wilcoxon；CI 为 bootstrap(BCa) 95%。")
md2.append("")
md2.append(f"{pipe} Group {pipe} Metric {pipe} mean(diff) {pipe} t p {pipe} wilcoxon p {pipe} dz {pipe} CI95 low {pipe} CI95 high {pipe}")
md2.append(f"{pipe}---{pipe}---:{pipe}---:{pipe}---:{pipe}---:{pipe}---:{pipe}---:{pipe}---:{pipe}")
for g, v in stats_out["groups"].items():
    for m in ["acc","f1"]:
        s = v[m]
        md2.append(
            f"{pipe} {g} {pipe} {m} {pipe} {s['diff_mean']:.4f} {pipe} {s['ttest_p']:.4g} {pipe} {s['wilcoxon_p']:.4g} {pipe} "
            f"{s['cohen_dz']:.3f} {pipe} {s['boot_ci95']['low']:.4f} {pipe} {s['boot_ci95']['high']:.4f} {pipe}"
        )
open(stats_md, "w", encoding="utf-8").write("\n".join(md2))

# ---- simple bar plots (F1 and Acc) ----
import matplotlib.pyplot as plt

def parse_ms(s):
    a = str(s).split("±")
    return float(a[0].strip()), float(a[1].strip())

plot_df = df.copy()
plot_df[["acc_mean","acc_std"]] = plot_df["Accuracy(mean±std)"].apply(lambda x: pd.Series(parse_ms(x)))
plot_df[["f1_mean","f1_std"]]   = plot_df["F1(mean±std)"].apply(lambda x: pd.Series(parse_ms(x)))

def plot(metric, ylab, fn):
    plt.figure()
    domains = ["Arxiv","Code","Yelp"]
    rewriters = ["repo","deepseek"]
    settings = ["ALL","TOP3"]
    x = np.arange(len(domains))
    w = 0.18

    # bars: (rewriter, setting) as groups
    idx = 0
    for rw in rewriters:
        for st in settings:
            sub = plot_df[(plot_df["Rewriter"]==rw) & (plot_df["Setting"]==st)].set_index("Domain").reindex(domains)
            m = sub[f"{metric}_mean"].values
            e = sub[f"{metric}_std"].values
            xpos = x + (idx - 1.5) * w
            plt.bar(xpos, m, width=w, label=f"{rw}-{st}")
            plt.errorbar(xpos, m, yerr=e, fmt="none", capsize=3)
            idx += 1

    plt.xticks(x, domains)
    plt.ylabel(ylab)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG, fn + ".png"), dpi=300)
    plt.savefig(os.path.join(OUT_FIG, fn + ".svg"))
    plt.close()

plot("f1", "F1", "fig_stageC_f1_all_vs_top3")
plot("acc", "Accuracy", "fig_stageC_acc_all_vs_top3")

print("saved:", csv_path)
print("saved:", md_path)
print("saved:", stats_json)
print("saved:", stats_md)
print("saved figures to:", OUT_FIG)
