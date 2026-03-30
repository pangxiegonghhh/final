# scripts/p0_rewrite.py
# -*- coding: utf-8 -*-
"""
Step 3: 对 human 和 AI 文本各做一次 DeepSeek 重写。
使用区间约束（±10%），单固定 prompt (P0)。

运行命令（api_key 自动从 configs/p0_config.json 读取）：
    # SQuAD: 重写 human
    python scripts\p0_rewrite.py --input_json SQuAD\human_corpus.json --out_json SQuAD\rewrite_squad_human.json

    # SQuAD: 重写 AI
    python scripts\p0_rewrite.py --input_json SQuAD\ai_corpus.json --out_json SQuAD\rewrite_squad_ai.json

    # CC-News: 重写 human
    python scripts\p0_rewrite.py --input_json CCNews\human_corpus.json --out_json CCNews\rewrite_ccnews_human.json

    # CC-News: 重写 AI
    python scripts\p0_rewrite.py --input_json CCNews\ai_corpus.json --out_json CCNews\rewrite_ccnews_ai.json
"""
import argparse, json, os, time
from pathlib import Path
from openai import OpenAI

def word_count(text):
    return len(text.split())

def build_rewrite_prompt(text, tolerance=0.10):
    wc = word_count(text)
    n_min = int(wc * (1 - tolerance))
    n_max = int(wc * (1 + tolerance))
    return f"""Rewrite the following paragraph to improve fluency while preserving meaning.
Target length: between {n_min} and {n_max} words (close to the original length).
- Do NOT add new facts, names, numbers, or claims.
- Do NOT delete important information.
- Do NOT summarize.
- Keep named entities and numeric values unchanged whenever possible.
Output ONLY the rewritten paragraph.

Paragraph:
{text}"""

def rewrite_one(client, model, text, temperature, max_tokens, tolerance):
    prompt = build_rewrite_prompt(text, tolerance)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a careful editor."},
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
    ap.add_argument("--input_json", required=True, help="human_corpus.json 或 ai_corpus.json")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--base_url", default="https://api.deepseek.com/v1")
    ap.add_argument("--config", default="configs/p0_config.json", help="配置文件路径")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--tolerance", type=float, default=0.10)
    ap.add_argument("--prompt_id", default="P0_range_10pct")
    args = ap.parse_args()

    # 从配置文件读取 API key（优先），也兼容环境变量
    api_key = ""
    if os.path.exists(args.config):
        cfg = json.load(open(args.config, "r", encoding="utf-8"))
        api_key = cfg.get("api_key", "")
        if not args.model or args.model == "deepseek-chat":
            args.model = cfg.get("rewriter", {}).get("model", args.model)
        if not args.base_url or args.base_url == "https://api.deepseek.com/v1":
            args.base_url = cfg.get("rewriter", {}).get("base_url", args.base_url)
    if not api_key:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    assert api_key, "请在 configs/p0_config.json 的 api_key 字段填入密钥，或设置环境变量 DEEPSEEK_API_KEY"
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    data = json.load(open(args.input_json, "r", encoding="utf-8"))

    # 缓存机制
    cache = {}
    if os.path.exists(args.out_json):
        existing = json.load(open(args.out_json, "r", encoding="utf-8"))
        cache = {item["text_id"]: item for item in existing}
        print(f"loaded cache: {len(cache)} items")

    results = list(cache.values())

    for i, item in enumerate(data):
        tid = item.get("text_id") or str(item.get("id", i))
        if tid in cache:
            continue

        original = item["text"]
        rewritten = rewrite_one(client, args.model, original, args.temperature, args.max_tokens, args.tolerance)

        if rewritten is None:
            print(f"  [{i}] FAILED: {tid}")
            continue

        entry = {
            "text_id": tid,
            "input": original,
            "rewrite": rewritten,
            "input_wc": word_count(original),
            "rewrite_wc": word_count(rewritten),
            "prompt_id": args.prompt_id,
            "model": args.model,
            "temperature": args.temperature
        }
        results.append(entry)
        cache[tid] = entry

        if (i + 1) % 50 == 0:
            with open(args.out_json, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"  [{i+1}/{len(data)}] saved {len(results)} items")

        time.sleep(0.3)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"done: {args.out_json} ({len(results)} items)")

if __name__ == "__main__":
    main()
