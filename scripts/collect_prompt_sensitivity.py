# scripts/collect_prompt_sensitivity.py
import argparse, json, re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def parse_ms(s):
    a = re.split(r"\s*±\s*", str(s))
    return float(a[0]), (float(a[1]) if len(a) > 1 else 0.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="dir of per-prompt result json")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--out_fig_prefix", required=True)  # without ext
    args = ap.parse_args()

    rows = []
    for p in sorted(Path(args.in_dir).glob("*.json")):
        j = json.load(open(p, "r", encoding="utf-8"))
        ms = j["multi_seed_summary"]
        rows.append({
            "prompt_file": p.name,
            "acc_mean": ms["acc_mean"], "acc_std": ms["acc_std"],
            "f1_mean": ms["f1_mean"], "f1_std": ms["f1_std"],
            "seeds": ms["multi_seeds"],
        })
    df = pd.DataFrame(rows).sort_values("f1_mean", ascending=False).reset_index(drop=True)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    # markdown
    md = []
    md.append("| rank | prompt_file | Acc (mean±std) | F1 (mean±std) |")
    md.append("|---:|---|---:|---:|")
    for i, r in df.iterrows():
        md.append(f"| {i+1} | {r['prompt_file']} | {r['acc_mean']:.4f} ± {r['acc_std']:.4f} | {r['f1_mean']:.4f} ± {r['f1_std']:.4f} |")
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    open(args.out_md, "w", encoding="utf-8").write("\n".join(md))

    # figure: F1 with errorbar
    plt.figure()
    x = range(len(df))
    plt.bar(list(x), df["f1_mean"].values)
    plt.errorbar(list(x), df["f1_mean"].values, yerr=df["f1_std"].values, fmt="none", capsize=3)
    plt.xticks(list(x), df["prompt_file"].tolist(), rotation=70, ha="right")
    plt.ylabel("F1")
    plt.tight_layout()
    plt.savefig(args.out_fig_prefix + "_f1.png", dpi=300)
    plt.savefig(args.out_fig_prefix + "_f1.svg")
    plt.close()

    print("saved:", args.out_csv)
    print("saved:", args.out_md)
    print("saved figs:", args.out_fig_prefix + "_f1.(png|svg)")

if __name__ == "__main__":
    main()