# scripts/p0_extract_features.py
# -*- coding: utf-8 -*-
"""
Step 4: 提取 Levenshtein ratio + n-gram overlap 特征。
特征定义对齐 RAIDAR 原文，共 5 维：
  [lev_ratio, 1-gram overlap, 2-gram overlap, 3-gram overlap, 4-gram overlap]

运行命令：
    # SQuAD
    python scripts\p0_extract_features.py ^
      --human_rewrite_json SQuAD\rewrite_squad_human.json ^
      --ai_rewrite_json SQuAD\rewrite_squad_ai.json ^
      --out_npz outputs\p0_baseline\features\squad_features.npz

    # CC-News
    python scripts\p0_extract_features.py ^
      --human_rewrite_json CCNews\rewrite_ccnews_human.json ^
      --ai_rewrite_json CCNews\rewrite_ccnews_ai.json ^
      --out_npz outputs\p0_baseline\features\ccnews_features.npz
"""
import argparse, json
import numpy as np
from pathlib import Path

try:
    from Levenshtein import distance as lev_distance
except ImportError:
    # fallback: 纯 python（慢但可用）
    def lev_distance(s1, s2):
        if len(s1) < len(s2):
            return lev_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
            prev = curr
        return prev[-1]

def levenshtein_ratio(text_a, text_b):
    """RAIDAR 定义: 1 - lev(a,b) / max(len(a), len(b))"""
    if not text_a and not text_b:
        return 1.0
    d = lev_distance(text_a, text_b)
    return 1.0 - d / max(len(text_a), len(text_b))

def get_ngrams(text, n):
    words = text.lower().split()
    return [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]

def ngram_overlap(text_a, text_b, n):
    """RAIDAR 定义: |共有 n-gram| / |input 的 n-gram 数|"""
    ngrams_a = get_ngrams(text_a, n)
    ngrams_b = get_ngrams(text_b, n)
    if not ngrams_a:
        return 0.0
    set_a = set(ngrams_a)
    set_b = set(ngrams_b)
    shared = set_a & set_b
    return len(shared) / len(set_a)

def extract_features(input_text, rewrite_text):
    """返回 5 维特征向量: [lev_ratio, 1gram, 2gram, 3gram, 4gram]"""
    lev = levenshtein_ratio(input_text, rewrite_text)
    ng1 = ngram_overlap(input_text, rewrite_text, 1)
    ng2 = ngram_overlap(input_text, rewrite_text, 2)
    ng3 = ngram_overlap(input_text, rewrite_text, 3)
    ng4 = ngram_overlap(input_text, rewrite_text, 4)
    return [lev, ng1, ng2, ng3, ng4]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human_rewrite_json", required=True)
    ap.add_argument("--ai_rewrite_json", required=True)
    ap.add_argument("--out_npz", required=True, help="输出 .npz，含 X_human, X_ai")
    args = ap.parse_args()

    human_data = json.load(open(args.human_rewrite_json, "r", encoding="utf-8"))
    ai_data = json.load(open(args.ai_rewrite_json, "r", encoding="utf-8"))

    print(f"human samples: {len(human_data)}, ai samples: {len(ai_data)}")

    X_human = []
    for item in human_data:
        feat = extract_features(item["input"], item["rewrite"])
        X_human.append(feat)

    X_ai = []
    for item in ai_data:
        feat = extract_features(item["input"], item["rewrite"])
        X_ai.append(feat)

    X_human = np.array(X_human, dtype=np.float64)
    X_ai = np.array(X_ai, dtype=np.float64)

    Path(args.out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_npz, X_human=X_human, X_ai=X_ai)

    print(f"X_human shape: {X_human.shape}")
    print(f"X_ai shape: {X_ai.shape}")
    print(f"Feature names: lev_ratio, 1gram, 2gram, 3gram, 4gram")
    print(f"saved: {args.out_npz}")

    # 打印特征统计（sanity check）
    names = ["lev_ratio", "1gram", "2gram", "3gram", "4gram"]
    print("\n--- Human feature stats ---")
    for i, name in enumerate(names):
        print(f"  {name}: mean={X_human[:,i].mean():.4f} std={X_human[:,i].std():.4f}")
    print("--- AI feature stats ---")
    for i, name in enumerate(names):
        print(f"  {name}: mean={X_ai[:,i].mean():.4f} std={X_ai[:,i].std():.4f}")

if __name__ == "__main__":
    main()
