# scripts/p1_rewrite_multi_prompt.py
# -*- coding: utf-8 -*-
"""
P1 Step 1: 对 human 和 AI 文本，用 prompt_pool 中的每个 prompt 各做一次重写。
支持断点续跑（缓存机制）。
"""
import argparse, json, os, time
from pathlib import Path
from openai import OpenAI


def word_count(text):
    return len(text.split())


def build_prompt(template, text, tolerance=0.10, length_control=True):
    wc = word_count(text)
    n_min = int(wc * (1 - tolerance)) if tolerance else wc
    n_max = int(wc * (1 + tolerance)) if tolerance else wc
    return template.replace("{text}", text).replace("{n_min}", str(n_min)).replace("{n_max}", str(n_max))


def rewrite_one(client, model, system_msg, user_msg, temperature, max_tokens):
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
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
    ap.add_argument("--prompt_pool_json", default="configs/p1_prompt_pool.json")
    ap.add_argument("--prompt_id", required=True, help="prompt_pool 中的 key，如 P1_fluency")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--config", default="configs/p0_config.json", help="读取 api_key")
    ap.add_argument("--max_retries", type=int, default=2)
    args = ap.parse_args()

    # 读 API key
    api_key = ""
    cfg = {}
    if os.path.exists(args.config):
        cfg = json.load(open(args.config, "r", encoding="utf-8"))
        api_key = cfg.get("api_key", "")
    if not api_key:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    assert api_key, "请在 configs/p0_config.json 填入 api_key"

    # 读 prompt pool
    pool = json.load(open(args.prompt_pool_json, "r", encoding="utf-8"))
    prompt_cfg = pool["prompt_pool"][args.prompt_id]
    model = pool.get("rewrite_model", "deepseek-chat")
    base_url = pool.get("base_url", "https://api.deepseek.com/v1")
    temperature = pool.get("temperature", 0.7)
    max_tokens = pool.get("max_tokens", 1024)

    client = OpenAI(api_key=api_key, base_url=base_url)
    data = json.load(open(args.input_json, "r", encoding="utf-8"))

    # 缓存
    cache = {}
    if os.path.exists(args.out_json):
        existing = json.load(open(args.out_json, "r", encoding="utf-8"))
        cache = {item["text_id"]: item for item in existing}
        print(f"loaded cache: {len(cache)} items")

    results = list(cache.values())

    for i, item in enumerate(data):
        tid = item.get("text_id") or item.get("id")
        if tid in cache:
            continue

        original = item["text"]
        tolerance = prompt_cfg.get("tolerance", 0.10)
        length_control = prompt_cfg.get("length_control", True)
        user_msg = build_prompt(prompt_cfg["user_template"], original, tolerance, length_control)
        system_msg = prompt_cfg.get("system", "You are a careful editor.")

        rewritten = None
        for attempt in range(args.max_retries):
            text = rewrite_one(client, model, system_msg, user_msg, temperature, max_tokens)
            if text is None:
                continue
            if length_control and tolerance:
                wc_in = word_count(original)
                wc_out = word_count(text)
                n_min = int(wc_in * (1 - tolerance))
                n_max = int(wc_in * (1 + tolerance))
                if n_min <= wc_out <= n_max:
                    rewritten = text
                    break
                elif attempt == args.max_retries - 1:
                    rewritten = text
            else:
                rewritten = text
                break

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
            "model": model,
            "temperature": temperature
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
