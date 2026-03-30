# main_yelp_gpt_rewrite_zjj_50.py
# -*- coding: utf-8 -*-

import os
import json
from pathlib import Path

import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential

# =========================
# 0) Config
# =========================
BASE_DIR = Path(__file__).resolve().parent  # Yelp/

MODEL = os.getenv("RAIDAR_REWRITE_MODEL", "gpt-3.5-turbo")  # 可改成 gpt-4o-mini
MAX_SAMPLES = 50  # 你这个脚本固定 50

HUMAN_IN = BASE_DIR / "yelp_human.json"
GPT_IN = BASE_DIR / "yelp_GPT_concise.json"

HUMAN_OUT = BASE_DIR / f"rewrite_yelp_human_inv_zjj_{MAX_SAMPLES}.json"
GPT_OUT = BASE_DIR / f"rewrite_yelp_GPT_inv_zjj_{MAX_SAMPLES}.json"

CACHE_FILE = BASE_DIR / f"rewrite_cache_zjj_{MAX_SAMPLES}.json"  # 断点续跑缓存

# =========================
# 1) Key (DO NOT hardcode)
# =========================
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set.\n"
        "PowerShell 里先执行：$env:OPENAI_API_KEY='你的key'\n"
    )
openai.api_key = api_key


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def openai_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def rewrite_once(text: str) -> str:
    # 跟 RAIDAR 的“改写但保留语义”一致
    resp = openai_backoff(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": "Rewrite the following text while preserving its meaning. "
                           "Do not add new information:\n\n" + text
            }
        ],
        temperature=0.7,
    )
    return resp["choices"][0]["message"]["content"].strip()


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def ensure_cache():
    if CACHE_FILE.exists():
        cache = load_json(CACHE_FILE)
        # 兼容旧格式
        if "human" not in cache:
            cache["human"] = {}
        if "gpt" not in cache:
            cache["gpt"] = {}
        return cache
    return {"human": {}, "gpt": {}}


def main():
    human_list = load_json(HUMAN_IN)[:MAX_SAMPLES]
    gpt_list = load_json(GPT_IN)[:MAX_SAMPLES]

    cache = ensure_cache()

    def process(split_name: str, items):
        """
        split_name: 'human' or 'gpt'
        items: list[str]
        return: list[dict] each dict: {"input": ..., "r1": ...}
        """
        out = []
        for i, src in enumerate(items):
            key = str(i)
            if key in cache[split_name]:
                r1 = cache[split_name][key]["r1"]
            else:
                r1 = rewrite_once(src)
                cache[split_name][key] = {"input": src, "r1": r1}
                save_json(CACHE_FILE, cache)  # 每条写一次，断点续跑安全

            out.append({"input": src, "r1": r1})

            if (i + 1) % 10 == 0 or i == 0:
                print(f"[{split_name}] {i+1}/{MAX_SAMPLES} done")

        return out

    human_out = process("human", human_list)
    gpt_out = process("gpt", gpt_list)

    save_json(HUMAN_OUT, human_out)
    save_json(GPT_OUT, gpt_out)

    print("saved:", HUMAN_OUT.name, GPT_OUT.name)
    print("cache:", CACHE_FILE.name)
    print("model:", MODEL, "max_samples:", MAX_SAMPLES)


if __name__ == "__main__":
    main()