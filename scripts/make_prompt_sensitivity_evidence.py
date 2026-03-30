import os, json, glob, re
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.abspath(os.path.dirname(__file__) + r"\..")
PAPER_DIR = os.path.join(ROOT, "outputs", "paper_tables")
PROMPT_CSVS = [
    os.path.join(PAPER_DIR, "prompt_sensitivity_arxiv_deepseek.csv"),
    os.path.join(PAPER_DIR, "prompt_sensitivity_arxiv_repo.csv"),
    os.path.join(PAPER_DIR, "prompt_sensitivity_code_deepseek.csv"),
    os.path.join(PAPER_DIR, "prompt_sensitivity_code_repo.csv"),
    os.path.join(PAPER_DIR, "prompt_sensitivity_yelp_deepseek.csv"),
    os.path.join(PAPER_DIR, "prompt_sensitivity_yelp_repo.csv"),
]

OUT_STATS_DIR = os.path.join(ROOT, "outputs", "stats")
OUT_PAPER_DIR = os.path.join(ROOT, "outputs", "paper_tables")
os.makedirs(OUT_STATS_DIR, exist_ok=True)
os.makedirs(OUT_PAPER_DIR, exist_ok=True)

# ---- helpers ----
def infer_domain_and_rewriter(csv_path: str):
    name = os.path.basename(csv_path).lower()
    # prompt_sensitivity_<domain>_<rewriter>.csv
    m = re.match(r"prompt_sensitivity_([a-z]+)_([a-z]+)\.csv", name)
    if not m:
        raise ValueError(f"bad csv name: {name}")
    dom = m.group(1)
    rewriter = m.group(2)
    # normalize
    if dom == "arxiv": dom2="Arxiv"
    elif dom == "code": dom2="Code"
    elif dom == "yelp": dom2="Yelp"
    else: dom2=dom
    return dom2, rewriter

def try_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding="utf-8")

def locate_prompt_json_dir(domain: str, rewriter: str) -> str:
    # outputs\10seed测试\prompt_sensitivity\<Domain>\<rewriter>\
    return os.path.join(ROOT, "outputs", "10seed测试", "prompt_sensitivity", domain, rewriter)

def load_prompt_jsons(prompt_dir: str):
    files = sorted(glob.glob(os.path.join(prompt_dir, "prompt_*.json")))
    if not files:
        raise FileNotFoundError(f"No prompt_*.json under {prompt_dir}")
    items = []
    for p in files:
        j = json.load(open(p, "r", encoding="utf-8"))
        ms = j.get("multi_seed_summary", None)
        if not ms:
            # fallback to single metrics
            raise RuntimeError(f"missing multi_seed_summary in {p}")
        items.append({
            "path": p,
            "slug": os.path.splitext(os.path.basename(p))[0].replace("prompt_", ""),
            "seeds": ms["seeds"],
            "acc_list": np.array(ms["acc_list"], float),
            "f1_list":  np.array(ms["f1_list"], float),
            "acc_mean": float(ms["acc_mean"]),
            "acc_std":  float(ms["acc_std"]),
            "f1_mean":  float(ms["f1_mean"]),
            "f1_std":   float(ms["f1_std"]),
        })
    return items

def dispersion_stats(vals: np.ndarray):
    vals = np.array(vals, float)
    return {
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "range": float(np.max(vals) - np.min(vals)),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)) if len(vals) >= 2 else 0.0,
    }

def top1_instability(metric_lists: np.ndarray):
    """
    metric_lists shape = (num_prompts, num_seeds)
    returns: winner counts, top1_change_rate
    """
    winners = np.argmax(metric_lists, axis=0)  # per seed winner prompt index
    counts = np.bincount(winners, minlength=metric_lists.shape[0])
    maxc = int(np.max(counts))
    n = metric_lists.shape[1]
    change_rate = float(1.0 - maxc / n)
    return counts.tolist(), change_rate

def oracle_gap(metric_lists: np.ndarray, metric_means: np.ndarray):
    """
    oracle_mean = mean over seeds of max(metric)
    best_single_mean = max over prompts of mean(metric)
    """
    oracle_mean = float(np.mean(np.max(metric_lists, axis=0)))
    best_single_mean = float(np.max(metric_means))
    return {
        "oracle_mean": oracle_mean,
        "best_single_mean": best_single_mean,
        "gap": float(oracle_mean - best_single_mean),
    }

def avg_pairwise_corr(metric_lists: np.ndarray):
    # correlation across seeds among prompts
    k = metric_lists.shape[0]
    if k < 2:
        return float("nan")
    C = np.corrcoef(metric_lists)
    # take upper triangle excluding diag
    triu = C[np.triu_indices(k, 1)]
    return float(np.nanmean(triu))

def friedman_p(metric_lists: np.ndarray):
    # Friedman requires >= 3 groups and >= 2 blocks; here blocks = seeds
    k, n = metric_lists.shape
    if k < 3 or n < 2:
        return float("nan")
    args = [metric_lists[i, :] for i in range(k)]
    stat, p = stats.friedmanchisquare(*args)
    return float(p)

# ---- main aggregation ----
evidence = {
    "meta": {
        "root": ROOT,
        "note": "Stage-B evidence for needing ensemble (C) / routing (D). Uses per-prompt 10-seed json under outputs/10seed测试/prompt_sensitivity/..",
        "metrics": ["acc", "f1"],
    },
    "groups": {},
}

rows_for_csv = []

for csv_path in PROMPT_CSVS:
    domain, rewriter = infer_domain_and_rewriter(csv_path)
    group_key = f"{domain}/{rewriter}"
    prompt_dir = locate_prompt_json_dir(domain, rewriter)

    prompts = load_prompt_jsons(prompt_dir)
    # ensure same seeds across prompts
    seeds0 = prompts[0]["seeds"]
    for it in prompts[1:]:
        if it["seeds"] != seeds0:
            raise RuntimeError(f"seed mismatch in {group_key}: {prompts[0]['path']} vs {it['path']}")

    acc_lists = np.stack([it["acc_list"] for it in prompts], axis=0)
    f1_lists  = np.stack([it["f1_list"]  for it in prompts], axis=0)
    acc_means = np.array([it["acc_mean"] for it in prompts], float)
    f1_means  = np.array([it["f1_mean"]  for it in prompts], float)

    acc_disp = dispersion_stats(acc_means)
    f1_disp  = dispersion_stats(f1_means)

    acc_win_counts, acc_top1_change = top1_instability(acc_lists)
    f1_win_counts,  f1_top1_change  = top1_instability(f1_lists)

    acc_oracle = oracle_gap(acc_lists, acc_means)
    f1_oracle  = oracle_gap(f1_lists,  f1_means)

    acc_corr = avg_pairwise_corr(acc_lists)
    f1_corr  = avg_pairwise_corr(f1_lists)

    acc_fried = friedman_p(acc_lists)
    f1_fried  = friedman_p(f1_lists)

    # record evidence
    evidence["groups"][group_key] = {
        "domain": domain,
        "rewriter": rewriter,
        "prompt_dir": prompt_dir,
        "num_prompts": int(len(prompts)),
        "seeds": seeds0,
        "dispersion": {"acc": acc_disp, "f1": f1_disp},
        "top1": {
            "acc": {"winner_counts": acc_win_counts, "top1_change_rate": acc_top1_change},
            "f1":  {"winner_counts": f1_win_counts,  "top1_change_rate": f1_top1_change},
        },
        "oracle": {"acc": acc_oracle, "f1": f1_oracle},
        "avg_pairwise_corr": {"acc": acc_corr, "f1": f1_corr},
        "friedman_p": {"acc": acc_fried, "f1": f1_fried},
        "prompts": [
            {
                "slug": it["slug"],
                "acc_mean": it["acc_mean"],
                "acc_std": it["acc_std"],
                "f1_mean": it["f1_mean"],
                "f1_std": it["f1_std"],
                "path": it["path"],
            }
            for it in prompts
        ],
    }

    # flat row for CSV/MD
    rows_for_csv.append({
        "Domain": domain,
        "Rewriter": rewriter,
        "#Prompts": len(prompts),
        "F1_range_across_prompts": f1_disp["range"],
        "F1_std_across_prompts": f1_disp["std"],
        "F1_top1_change_rate": f1_top1_change,
        "F1_oracle_gap": f1_oracle["gap"],
        "F1_avg_pairwise_corr": f1_corr,
        "F1_friedman_p": f1_fried,
        "Acc_range_across_prompts": acc_disp["range"],
        "Acc_std_across_prompts": acc_disp["std"],
        "Acc_top1_change_rate": acc_top1_change,
        "Acc_oracle_gap": acc_oracle["gap"],
        "Acc_avg_pairwise_corr": acc_corr,
        "Acc_friedman_p": acc_fried,
        "PromptDir": prompt_dir,
    })

# ---- write outputs ----
out_json = os.path.join(OUT_STATS_DIR, "prompt_sensitivity_evidence.json")
json.dump(evidence, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

df = pd.DataFrame(rows_for_csv).sort_values(["Domain", "Rewriter"]).reset_index(drop=True)
out_csv = os.path.join(OUT_PAPER_DIR, "prompt_sensitivity_evidence.csv")
df.to_csv(out_csv, index=False, encoding="utf-8-sig")

# markdown table for paper
pipe = chr(124)
md = []
md.append("# Stage-B Evidence: Prompt Sensitivity ⇒ Need Ensemble (C) / Routing (D)")
md.append("")
md.append("解释：")
md.append("- **range/std across prompts** 大 ⇒ 同域下 prompt 选择敏感（单 prompt 不可靠）")
md.append("- **top1 change rate** 高 ⇒ 不同 seed 下最优 prompt 频繁切换（需要集成/路由来稳定）")
md.append("- **oracle gap** > 0 ⇒ 存在可提升上界（集成/路由有客观空间）")
md.append("- **Friedman p** 小 ⇒ prompt 差异显著（选择 prompt 不是噪声）")
md.append("")
md.append(f"{pipe} Domain {pipe} Rewriter {pipe} #Prompts {pipe} F1 range {pipe} F1 std {pipe} Top1 change {pipe} Oracle gap {pipe} Friedman p {pipe}")
md.append(f"{pipe}---{pipe}---{pipe}---:{pipe}---:{pipe}---:{pipe}---:{pipe}---:{pipe}---:{pipe}")
for rec in df.to_dict("records"):
    md.append(
        f"{pipe} {rec['Domain']} {pipe} {rec['Rewriter']} {pipe} {int(rec['#Prompts'])} {pipe} "
        f"{rec['F1_range_across_prompts']:.4f} {pipe} {rec['F1_std_across_prompts']:.4f} {pipe} "
        f"{rec['F1_top1_change_rate']:.2f} {pipe} {rec['F1_oracle_gap']:.4f} {pipe} "
        f"{rec['F1_friedman_p']:.4g} {pipe}"
    )

out_md = os.path.join(OUT_PAPER_DIR, "prompt_sensitivity_evidence.md")
open(out_md, "w", encoding="utf-8").write("\n".join(md))

print("saved:", out_json)
print("saved:", out_csv)
print("saved:", out_md)
