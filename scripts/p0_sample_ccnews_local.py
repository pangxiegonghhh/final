#将本地已下载好的CCNEWS加载到当前项目文件夹，绕过huggingface的网络获取
# scripts/p0_sample_ccnews_local.py
import json, random, re, argparse
from pathlib import Path
import pandas as pd

def word_count(text):
    return len(re.findall(r'\w+', text))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet_dir", default=r"E:\hf_cache\hub\datasets--cc_news\snapshots\81eb2ce0d2a9dad6ad16b68ef750ec290880fa36\plain_text")
    ap.add_argument("--sample_size", type=int, default=2000)
    ap.add_argument("--min_words", type=int, default=150)
    ap.add_argument("--max_words", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    random.seed(args.seed)
    records = []
    for pf in sorted(Path(args.parquet_dir).glob("*.parquet")):
        print(f"读取 {pf.name} ...")
        df = pd.read_parquet(pf, columns=["text"])
        for text in df["text"]:
            if not isinstance(text, str):
                continue
            wc = word_count(text)
            if args.min_words <= wc <= args.max_words:
                records.append({"id": len(records), "text": text, "n_words": wc})
        if len(records) >= args.sample_size * 10:  # 够用了就停
            break

    print(f"候选文本 {len(records)} 条，随机采样 {args.sample_size} 条")
    sampled = random.sample(records, min(args.sample_size, len(records)))

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)
    print(f"已保存至 {args.out_json}")

if __name__ == "__main__":
    main()