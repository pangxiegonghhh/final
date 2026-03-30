# deepseek_rewrite_by_idx.py
# -*- coding: utf-8 -*-
"""
DeepSeek batch rewrite by index list (Repo-3 replacement experiment)

PowerShell usage:
  python .\deepseek_rewrite_by_idx.py --domain arxiv --api_key "YOUR_KEY" `
    --in_json .\Arxiv\rewrite_arxiv_human_inv.json --idx_json .\Arxiv\idx_all_350_0to349.json `
    --out_json .\Arxiv\rewrite_arxiv_human_inv_deepseek_350.json --target_ratio 1.0 --sleep 1 --resume
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Optional, Any, Dict, Set

import requests

# ===== Prompt sets (align with repo keys) =====
PROMPTS_YELP_ARXIV: List[str] = [
    "Revise this with your best effort",
    "Help me polish this",
    "Rewrite this for me",
    "Make this fluent while doing minimal change",
    "Refine this for me please",
    "Concise this for me and keep all the information",
    "Improve this in GPT way",
]

PROMPTS_CODE: List[str] = [
    "Revise the code with your best effort",
    "Help me polish this code",
    "Rewrite the code with GPT style",
    "Refine the code for me please",
    "Concise the code without change the functionality",
]


def word_count(s: str) -> int:
    return len([w for w in str(s).split() if w.strip()])


def build_user_prompt(instruction: str, text: str, target_words: Optional[int]) -> str:
    """DART-like length constraint: about target_words (±10%)."""
    if target_words is None:
        return f"{instruction}\n\nTEXT:\n{text}"
    return (
        f"{instruction}\n\n"
        f"Constraints:\n"
        f"- Keep the meaning.\n"
        f"- Target length: about {target_words} words (±10%).\n\n"
        f"TEXT:\n{text}"
    )


def _sleep_backoff(attempt: int, base: float = 1.0, cap: float = 30.0) -> None:
    """Exponential backoff with cap."""
    t = min(cap, base * (2 ** attempt))
    time.sleep(t)


def call_deepseek_chat(
    api_key: str,
    base_url: str,
    model: str,
    user_prompt: str,
    temperature: float = 0.2,
    timeout: int = 120,
    max_retries: int = 6,
) -> str:
    """
    DeepSeek OpenAI-compatible endpoint: POST {base_url}/v1/chat/completions

    Retries on 429/5xx/timeout with exponential backoff.
    """
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user_prompt}],
        "temperature": temperature,
    }

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                _sleep_backoff(attempt, base=1.0, cap=30.0)
                continue

            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()

        except (requests.Timeout, requests.ConnectionError) as e:
            last_err = e
            _sleep_backoff(attempt, base=1.0, cap=30.0)
            continue
        except Exception as e:
            raise

    raise RuntimeError(f"DeepSeek call failed after retries. Last error: {last_err}")


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def choose_prompts(domain: str) -> List[str]:
    if domain in ("yelp", "arxiv"):
        return PROMPTS_YELP_ARXIV
    return PROMPTS_CODE


def _collect_done_indices(existing: List[Any]) -> Set[int]:
    """
    Resume key: by _idx (preferred). Backward compatible:
    - if old file has no _idx, try to infer nothing (so it will re-run).
    """
    done: Set[int] = set()
    for item in existing:
        if isinstance(item, dict) and "_idx" in item:
            try:
                done.add(int(item["_idx"]))
            except Exception:
                pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["yelp", "arxiv", "code"], required=True)
    ap.add_argument("--in_json", required=True, help="source rewrite_*_inv.json (use input field only)")
    ap.add_argument("--idx_json", required=True, help="index list json, e.g. idx_all_350_0to349.json")
    ap.add_argument("--out_json", required=True, help="output deepseek rewrite json")
    ap.add_argument("--target_ratio", type=float, default=1.0, help="length ratio target, e.g. 0.8/1.0/1.2")
    ap.add_argument("--sleep", type=float, default=0.2, help="sleep seconds after each prompt call (set 0 to disable)")
    ap.add_argument("--temperature", type=float, default=0.2)

    ap.add_argument("--api_key", default=os.getenv("DEEPSEEK_API_KEY", ""))
    ap.add_argument("--base_url", default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    ap.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--max_retries", type=int, default=6)

    ap.add_argument("--resume", action="store_true", help="resume if out_json exists (skip finished indices)")
    args = ap.parse_args()

    if not args.api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is empty. Please pass --api_key or set env var.")

    prompts = choose_prompts(args.domain)

    src: List[Dict[str, Any]] = load_json(args.in_json)
    idx: List[int] = load_json(args.idx_json)

    # Optional resume: load existing outputs and skip completed indices
    existing: List[Any] = []
    done_idx: Set[int] = set()
    if args.resume and Path(args.out_json).exists():
        try:
            existing = load_json(args.out_json)
            done_idx = _collect_done_indices(existing)
            print(f"[resume] loaded {len(existing)} existing items, will skip by _idx. done_idx={len(done_idx)}")
        except Exception:
            print("[resume] failed to load existing out_json, starting fresh.")
            existing = []
            done_idx = set()

    out: List[Any] = list(existing)

    total = len(idx)
    for n, i in enumerate(idx, start=1):
        # bounds check
        if i < 0 or i >= len(src):
            print(f"[warn] idx {i} out of range for src size {len(src)}. skip {n}/{total}")
            continue

        if args.resume and i in done_idx:
            print(f"skip {n}/{total} (already done idx={i})")
            continue

        original = src[i].get("input", "")
        wc = word_count(original)
        target_words = max(5, int(round(wc * args.target_ratio)))

        item: Dict[str, Any] = {"_idx": int(i), "input": original}

        for inst in prompts:
            user_prompt = build_user_prompt(inst, original, target_words)
            rewritten = call_deepseek_chat(
                api_key=args.api_key,
                base_url=args.base_url,
                model=args.model,
                user_prompt=user_prompt,
                temperature=args.temperature,
                timeout=args.timeout,
                max_retries=args.max_retries,
            )
            item[inst] = rewritten

            if args.sleep and args.sleep > 0:
                time.sleep(args.sleep)

        out.append(item)
        done_idx.add(int(i))

        # periodic flush (safer for long runs)
        if n % 5 == 0:
            save_json(out, args.out_json)
            print(f"done {n}/{total} (autosaved)")
        else:
            print(f"done {n}/{total}")

    save_json(out, args.out_json)
    print("saved:", args.out_json)


if __name__ == "__main__":
    main()