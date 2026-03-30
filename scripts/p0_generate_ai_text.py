# scripts/p0_generate_ai_text.py
# -*- coding: utf-8 -*-
"""
Step 2: 用 DeepSeek 为每条人类文本生成同域同长度的 AI 对照文本。
采用"元信息驱动生成"（用 title 做 prompt），而非直接改写。

运行命令（api_key 自动从 configs/p0_config.json 读取）：
    # SQuAD
    python scripts\p0_generate_ai_text.py --human_json SQuAD\human_corpus.json --out_json SQuAD\ai_corpus.json

    # CC-News
    python scripts\p0_generate_ai_text.py --human_json CCNews\human_corpus.json --out_json CCNews\ai_corpus.json
"""
import argparse, json, os, time
from pathlib import Path
from openai import OpenAI

def word_count(text):
    return len(text.split())

def build_generation_prompt(title, source, target_words, tolerance=0.10):
    n_min = int(target_words * (1 - tolerance))
    n_max = int(target_words * (1 + tolerance))

    if source == "squad":
        style = "an informative encyclopedic paragraph"
    else:
        style = "a news article paragraph"

    return f"""Write {style} about "{title}".
Target length: between {n_min} and {n_max} words.
- Write naturally and fluently.
- Include 2-3 factual points.
- Do NOT use bullet points or lists.
- Output ONLY the paragraph, no titles or labels."""

def generate_one(client, model, prompt, temperature, max_tokens):
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful writer."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  API error: {e}")
        time.sleep(5)
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human_json", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--base_url", default="https://api.deepseek.com/v1")
    ap.add_argument("--config", default="configs/p0_config.json", help="配置文件路径，从中读取 api_key")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--tolerance", type=float, default=0.10)
    ap.add_argument("--max_retries", type=int, default=2)
    args = ap.parse_args()

    # 从配置文件读取 API key（优先），也兼容环境变量
    api_key = ""
    if os.path.exists(args.config):
        cfg = json.load(open(args.config, "r", encoding="utf-8"))
        api_key = cfg.get("api_key", "")
        # 也可以从配置文件覆盖 model / base_url
        if not args.model or args.model == "deepseek-chat":
            args.model = cfg.get("rewriter", {}).get("model", args.model)
        if not args.base_url or args.base_url == "https://api.deepseek.com/v1":
            args.base_url = cfg.get("rewriter", {}).get("base_url", args.base_url)
    if not api_key:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    assert api_key, "请在 configs/p0_config.json 的 api_key 字段填入密钥，或设置环境变量 DEEPSEEK_API_KEY"

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    human_data = json.load(open(args.human_json, "r", encoding="utf-8"))

    # 如果已有部分缓存，加载继续
    cache = {}
    if os.path.exists(args.out_json):
        existing = json.load(open(args.out_json, "r", encoding="utf-8"))
        cache = {item["text_id"]: item for item in existing}
        print(f"loaded cache: {len(cache)} items")

    results = list(cache.values())

    for i, item in enumerate(human_data):
        tid = item.get("text_id") or str(item.get("id", i))
        if tid in cache:
            continue

        source = item.get("source", "cc_news")
        title = item.get("title") or item.get("text", "")[:80].split("\n")[0].strip()
        target_wc = item.get("word_count") or item.get("n_words", 200)

        prompt = build_generation_prompt(title, source, target_wc, args.tolerance)

        generated = None
        for attempt in range(args.max_retries):
            text = generate_one(client, args.model, prompt, args.temperature, args.max_tokens)
            if text is None:
                continue
            wc = word_count(text)
            n_min = int(target_wc * (1 - args.tolerance))
            n_max = int(target_wc * (1 + args.tolerance))
            if n_min <= wc <= n_max:
                generated = text
                break
            elif attempt == args.max_retries - 1:
                generated = text  # 最后一次保留，标记 violation

        if generated is None:
            print(f"  [{i}/{len(human_data)}] FAILED: {tid}")
            continue
        else:
            print(f"  [{i+1}/{len(human_data)}] OK  wc={word_count(generated)}  id={str(tid)[:8]}")

        out_wc = word_count(generated)
        length_ok = abs(out_wc - target_wc) / max(target_wc, 1) <= args.tolerance

        entry = {
            "text_id": tid,
            "text": generated,
            "word_count": out_wc,
            "target_word_count": target_wc,
            "length_violation": not length_ok,
            "title": title,
            "source": source,
            "model": args.model,
            "temperature": args.temperature
        }
        results.append(entry)
        cache[tid] = entry

        if (i + 1) % 50 == 0:
            # 定期保存
            with open(args.out_json, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"  [{i+1}/{len(human_data)}] saved {len(results)} items")

        time.sleep(0.3)  # rate limit

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"done: {args.out_json} ({len(results)} items)")

if __name__ == "__main__":
    main()
