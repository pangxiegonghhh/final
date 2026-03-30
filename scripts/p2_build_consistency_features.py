# scripts/p2_build_consistency_features.py
# -*- coding: utf-8 -*-
"""
P2 Step 2: 计算 prompt 输出间一致性特征，追加到 Ensemble-All 特征后面
输出: {domain}_ensemble_all_cons.npz (39维 = 35 + 4)
"""
import argparse, os, json
import numpy as np
from itertools import combinations
from pathlib import Path

try:
    from rapidfuzz import fuzz as rfuzz
    def token_set_ratio(a, b):
        return rfuzz.token_set_ratio(a, b) / 100.0
except ImportError:
    from difflib import SequenceMatcher
    def token_set_ratio(a, b):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

ALL_PROMPTS = [
    "P0_baseline", "P1_fluency", "P2_synonym",
    "P3_restructure", "P4_formal", "P5_concise", "P6_no_length"
]

DOMAIN_DIRS = {
    "ccnews": "CCNews",
    "squad": "SQuAD",
}

def load_rewrites(project_root, domain, text_type, prompt):
    path = os.path.join(project_root, DOMAIN_DIRS[domain], "rewrites_p1",
                        f"{text_type}_{prompt}.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [item["rewrite"] for item in data]

def compute_consistency_for_sample(rewrites_per_prompt):
    valid = [r for r in rewrites_per_prompt if r and len(r.strip()) > 0]
    if len(valid) < 2:
        return np.zeros(4, dtype=np.float32)

    pairwise_sims = []
    for a, b in combinations(valid, 2):
        pairwise_sims.append(token_set_ratio(a, b))

    sims = np.array(pairwise_sims)
    mean_sim = float(np.mean(sims))
    std_sim = float(np.std(sims))

    changes = []
    for i, r in enumerate(valid):
        others = [token_set_ratio(r, valid[j]) for j in range(len(valid)) if j != i]
        changes.append(1.0 - np.mean(others))

    mean_change_var = float(np.var(changes))
    range_change = float(max(changes) - min(changes))

    return np.array([mean_sim, std_sim, mean_change_var, range_change],
                    dtype=np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--ensemble_npz_dir", default=r"outputs\p2_ensemble\features")
    parser.add_argument("--out_dir", default=r"outputs\p2_ensemble\features")
    parser.add_argument("--n_samples", type=int, default=2000)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for domain in ["ccnews", "squad"]:
        print(f"\n=== Processing {domain} ===")

        npz_path = os.path.join(args.ensemble_npz_dir, f"{domain}_ensemble_all.npz")
        data = np.load(npz_path)
        X_base = data["X"]
        y = data["y"]
        n_total = len(y)
        n_human = args.n_samples

        print("  Loading rewrite caches...")
        human_rewrites = {p: load_rewrites(args.project_root, domain, "human", p)
                          for p in ALL_PROMPTS}
        ai_rewrites = {p: load_rewrites(args.project_root, domain, "ai", p)
                       for p in ALL_PROMPTS}

        print("  Computing consistency features...")
        cons_features = []
        for i in range(n_total):
            if i < n_human:
                idx = i
                rewrites = [human_rewrites[p][idx] if idx < len(human_rewrites[p]) else ""
                           for p in ALL_PROMPTS]
            else:
                idx = i - n_human
                rewrites = [ai_rewrites[p][idx] if idx < len(ai_rewrites[p]) else ""
                           for p in ALL_PROMPTS]
            cons_features.append(compute_consistency_for_sample(rewrites))

            if (i + 1) % 500 == 0:
                print(f"    {i+1}/{n_total}")

        C = np.array(cons_features)
        X_cons = np.hstack([X_base, C])

        out_path = os.path.join(args.out_dir, f"{domain}_ensemble_all_cons.npz")
        np.savez(out_path, X=X_cons, y=y,
                 dims=X_cons.shape[1],
                 cons_dim=4,
                 cons_names=json.dumps([
                     "mean_pairwise_sim", "std_pairwise_sim",
                     "mean_change_var", "range_change"
                 ]))
        print(f"  Ensemble-All+Cons: shape={X_cons.shape} -> {out_path}")

    print("\nDone.")

if __name__ == "__main__":
    main()