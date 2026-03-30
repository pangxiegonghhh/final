# scripts/p2_build_ensemble_features.py
# -*- coding: utf-8 -*-
"""
P2 Step 1: 将 P1 的逐 prompt 特征矩阵拼接为集成特征矩阵
- Ensemble-All: 7 prompts × 5 dims = 35 dims
- Ensemble-Top3: 3 prompts × 5 dims = 15 dims
"""
import argparse, os, json
import numpy as np
from pathlib import Path

# P1 的全部 7 个 prompt
ALL_PROMPTS = [
    "P0_baseline", "P1_fluency", "P2_synonym",
    "P3_restructure", "P4_formal", "P5_concise", "P6_no_length"
]

# Top-3 选择（来自 P1 结果）
TOP3 = {
    "ccnews": ["P4_formal", "P1_fluency", "P0_baseline"],
    "squad":  ["P5_concise", "P0_baseline", "P3_restructure"],
}

def load_features(feat_dir, domain, prompt):
    """加载 P1 阶段的单 prompt 特征 npz"""
    path = os.path.join(feat_dir, f"{domain}_{prompt}.npz")
    data = np.load(path)
    X_human = data["X_human"]  # label=0
    X_ai = data["X_ai"]        # label=1
    X = np.vstack([X_human, X_ai])
    y = np.concatenate([np.zeros(len(X_human)), np.ones(len(X_ai))])
    return X, y

def concat_features(feat_dir, domain, prompt_list):
    """按 prompt_list 拼接特征，返回 (X_concat, y)"""
    Xs = []
    y_ref = None
    for p in prompt_list:
        X, y = load_features(feat_dir, domain, p)
        Xs.append(X)
        if y_ref is None:
            y_ref = y
        else:
            assert np.array_equal(y, y_ref), f"标签不一致: {p}"
    return np.hstack(Xs), y_ref

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat_dir", default=r"outputs\p1_sensitivity\features",
                        help="P1 特征目录")
    parser.add_argument("--out_dir", default=r"outputs\p2_ensemble\features",
                        help="输出目录")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for domain in ["ccnews", "squad"]:
        # Ensemble-All (35维)
        X_all, y = concat_features(args.feat_dir, domain, ALL_PROMPTS)
        out_all = os.path.join(args.out_dir, f"{domain}_ensemble_all.npz")
        np.savez(out_all, X=X_all, y=y,
                 prompts=json.dumps(ALL_PROMPTS),
                 dims=X_all.shape[1])
        print(f"[{domain}] Ensemble-All: shape={X_all.shape} -> {out_all}")

        # Ensemble-Top3 (15维)
        top3 = TOP3[domain]
        X_top3, _ = concat_features(args.feat_dir, domain, top3)
        out_top3 = os.path.join(args.out_dir, f"{domain}_ensemble_top3.npz")
        np.savez(out_top3, X=X_top3, y=y,
                 prompts=json.dumps(top3),
                 dims=X_top3.shape[1])
        print(f"[{domain}] Ensemble-Top3 ({top3}): shape={X_top3.shape} -> {out_top3}")

    print("\nDone. All ensemble features saved.")

if __name__ == "__main__":
    main()