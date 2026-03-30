# scripts/p0_sample_corpus.py
# -*- coding: utf-8 -*-
"""
Step 1: 从 HuggingFace 下载 SQuAD / CC-News，抽样人类文本语料。
去重 + 长度过滤 + 格式过滤 + 固定 seed。

运行命令：
    # SQuAD
    python scripts\p0_sample_corpus.py --dataset squad --sample_size 2000 --out_json SQuAD\human_corpus.json

    # CC-News
    python scripts\p0_sample_corpus.py --dataset cc_news --sample_size 2000 --out_json CCNews\human_corpus.json
"""
import argparse, json, hashlib, re, os
from pathlib import Path

def word_count(text):
    return len(text.split())

def has_bad_format(text):
    """过滤包含列表符号、大量URL、代码块的段落"""
    if re.search(r'(^|\n)\s*[\-\*•]\s', text):
        return True
    if text.count('http') > 2:
        return True
    if '```' in text or '{' in text and '}' in text and ';' in text:
        return True
    return False

def text_id(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]

def sample_squad(sample_size, min_words, max_words, seed=42):
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad", split="train")

    seen = set()
    candidates = []
    for item in ds:
        ctx = item["context"].strip()
        tid = text_id(ctx)
        if tid in seen:
            continue
        seen.add(tid)
        wc = word_count(ctx)
        if wc < min_words or wc > max_words:
            continue
        if has_bad_format(ctx):
            continue
        candidates.append({
            "text_id": tid,
            "text": ctx,
            "word_count": wc,
            "title": item.get("title", ""),
            "source": "squad"
        })

    import random
    random.seed(seed)
    random.shuffle(candidates)
    sampled = candidates[:sample_size]
    print(f"[SQuAD] candidates={len(candidates)}, sampled={len(sampled)}")
    return sampled

def sample_ccnews(sample_size, min_words, max_words, seed=42):
    from datasets import load_dataset
    ds = load_dataset("cc_news", split="train", streaming=True)

    seen = set()
    candidates = []
    count = 0
    for item in ds:
        count += 1
        if count > 200000:  # 只扫描前 20 万条，够用
            break
        text_raw = item.get("text", "").strip()
        # 抽取中部段落（跳过首段和末段）
        paragraphs = [p.strip() for p in text_raw.split('\n\n') if p.strip()]
        if len(paragraphs) < 3:
            continue
        # 取中间段落
        for p in paragraphs[1:-1]:
            tid = text_id(p)
            if tid in seen:
                continue
            seen.add(tid)
            wc = word_count(p)
            if wc < min_words or wc > max_words:
                continue
            if has_bad_format(p):
                continue
            candidates.append({
                "text_id": tid,
                "text": p,
                "word_count": wc,
                "title": item.get("title", ""),
                "source": "cc_news"
            })

    import random
    random.seed(seed)
    random.shuffle(candidates)
    sampled = candidates[:sample_size]
    print(f"[CC-News] candidates={len(candidates)}, sampled={len(sampled)}")
    return sampled

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["squad", "cc_news"])
    ap.add_argument("--sample_size", type=int, default=2000)
    ap.add_argument("--min_words", type=int, default=50)
    ap.add_argument("--max_words", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    if args.dataset == "squad":
        data = sample_squad(args.sample_size, args.min_words, args.max_words, args.seed)
    else:
        data = sample_ccnews(args.sample_size, args.min_words, args.max_words, args.seed)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"saved: {args.out_json} ({len(data)} items)")

if __name__ == "__main__":
    main()
