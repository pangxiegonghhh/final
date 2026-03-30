# scripts/p1_extract_features_multi.py
# -*- coding: utf-8 -*-
"""
P1 Step 2: 对每个 prompt 的重写结果，提取 5 维特征（与 P0 完全一致）。
"""
import argparse, json, os
import numpy as np
from pathlib import Path

try:
    from Levenshtein import distance as lev_distance
except ImportError:
    def lev_distance(s1, s2):
        if len(s1) < len(s2): return lev_distance(s2, s1)
        if len(s2) == 0: return len(s1)
        prev = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                curr.append(min(prev[j+1]+1, curr[j]+1, prev[j]+(c1!=c2)))
            prev = curr
        return prev[-1]


def levenshtein_ratio(a, b):
    if not a and not b: return 1.0
    return 1.0 - lev_distance(a, b) / max(len(a), len(b))


def get_ngrams(text, n):
    words = text.lower().split()
    return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]


def ngram_overlap(a, b, n):
    ng_a = get_ngrams(a, n)
    if not ng_a: return 0.0
    return len(set(ng_a) & set(get_ngrams(b, n))) / len(set(ng_a))


def extract_features(inp, rew):
    return [levenshtein_ratio(inp, rew),
            ngram_overlap(inp, rew, 1),
            ngram_overlap(inp, rew, 2),
            ngram_overlap(inp, rew, 3),
            ngram_overlap(inp, rew, 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human_json", required=True)
    ap.add_argument("--ai_json", required=True)
    ap.add_argument("--out_npz", required=True)
    args = ap.parse_args()

    human = json.load(open(args.human_json, "r", encoding="utf-8"))
    ai = json.load(open(args.ai_json, "r", encoding="utf-8"))

    X_h = np.array([extract_features(x["input"], x["rewrite"]) for x in human], dtype=np.float64)
    X_a = np.array([extract_features(x["input"], x["rewrite"]) for x in ai], dtype=np.float64)

    Path(args.out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_npz, X_human=X_h, X_ai=X_a)
    print(f"saved: {args.out_npz} (human={X_h.shape}, ai={X_a.shape})")


if __name__ == "__main__":
    main()
