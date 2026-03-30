# scripts/p3_extract_meta_features.py
# -*- coding: utf-8 -*-
"""
P3 Step 1: 从原始输入文本提取元特征（不需要重写）
输出: {domain}_meta_features.npz (X_meta: 4000×8, y: 4000)
"""
import argparse, os, json, re
import numpy as np
from pathlib import Path

DOMAIN_DIRS = {
    "ccnews": "CCNews",
    "squad": "SQuAD",
}

def meta_features(text):
    """从原始文本提取 8 维元特征"""
    words = text.split()
    chars = list(text)
    n_words = len(words) if words else 1
    n_chars = len(chars) if chars else 1

    # 句子切分（简单用 .!? 分割）
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    n_sents = len(sentences) if sentences else 1

    # 词汇多样性
    unique_words = set(w.lower() for w in words)
    ttr = len(unique_words) / n_words if n_words > 0 else 0

    return np.array([
        n_words / 1000.0,                                    # word_count (归一化)
        np.mean([len(w) for w in words]) if words else 0,    # avg_word_len
        n_sents,                                              # sentence_count
        n_words / n_sents,                                    # avg_sent_len
        sum(1 for c in chars if c in '.,;:!?-()[]{}"\'/') / n_chars,  # punct_ratio
        sum(1 for c in chars if c.isdigit()) / n_chars,       # digit_ratio
        sum(1 for c in chars if c.isupper()) / n_chars,       # upper_ratio
        ttr,                                                   # type_token_ratio
    ], dtype=np.float32)

def load_inputs(project_root, domain, text_type):
    """从任意一个 prompt 的重写 JSON 读取 input 字段（所有 prompt 的 input 相同）"""
    path = os.path.join(project_root, DOMAIN_DIRS[domain],
                        "rewrites_p1", f"{text_type}_P0_baseline.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [item["input"] for item in data]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--out_dir", default=r"outputs\p3_routing\meta_features")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for domain in ["ccnews", "squad"]:
        print(f"=== {domain} ===")

        human_inputs = load_inputs(args.project_root, domain, "human")
        ai_inputs = load_inputs(args.project_root, domain, "ai")

        print(f"  human: {len(human_inputs)}, ai: {len(ai_inputs)}")

        all_inputs = human_inputs + ai_inputs
        X_meta = np.array([meta_features(t) for t in all_inputs])
        y = np.concatenate([np.zeros(len(human_inputs)), np.ones(len(ai_inputs))])

        out_path = os.path.join(args.out_dir, f"{domain}_meta_features.npz")
        np.savez(out_path, X_meta=X_meta, y=y,
                 feature_names=json.dumps([
                     "word_count", "avg_word_len", "sentence_count", "avg_sent_len",
                     "punct_ratio", "digit_ratio", "upper_ratio", "type_token_ratio"
                 ]))
        print(f"  shape={X_meta.shape} -> {out_path}")

    print("\nDone.")

if __name__ == "__main__":
    main()