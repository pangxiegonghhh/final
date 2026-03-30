# P0 Baseline 执行报告：SQuAD / CC-News 单次重写 Invariance 基线

## 总览

本报告为你的 P0 阶段提供**逐步可执行的代码与命令**，覆盖从数据下载到最终评估指标导出的完整流程。

**核心决策摘要：**

- **重写提示词**：使用区间约束（±10%），更稳健，避免模型为凑精确词数而扭曲语义
- **分类器**：LR（主）+ XGBoost（辅）+ MLP（对齐 RAIDAR 对照）
- **特征**：对齐 RAIDAR 定义——n-gram overlap（1~4-gram）+ Levenshtein ratio，共 5 维
- **评估**：AUROC、F1、TPR@FPR=1%/5%，复用原仓库评估逻辑
- **原仓库地址**：`https://github.com/cvlab-columbia/RaidarLLMDetect`

---

## Step 0：项目目录结构与环境准备

### 0.1 目录结构

```
E:\jnu\Final_thesis\实验\Raidar\RaidarLLMDetect\
├── SQuAD\                          # SQuAD 数据与重写缓存
│   ├── raw\                        # 原始数据
│   ├── human_corpus.json           # 抽样后的人类文本
│   ├── ai_corpus.json              # DeepSeek 生成的 AI 文本
│   ├── rewrite_squad_human.json    # 重写缓存（human）
│   └── rewrite_squad_ai.json       # 重写缓存（AI）
├── CCNews\                         # CC-News 同结构
├── scripts\
│   ├── p0_download_data.py
│   ├── p0_sample_corpus.py
│   ├── p0_generate_ai_text.py
│   ├── p0_rewrite.py
│   ├── p0_extract_features.py
│   ├── p0_train_eval.py
│   └── p0_export_results.py
├── outputs\
│   ├── p0_baseline\                # P0 产物
│   │   ├── features\
│   │   ├── models\
│   │   ├── metrics\
│   │   └── figures\
│   └── repro\
└── configs\
    └── p0_config.json
```

### 0.2 依赖安装

```powershell
cd "E:\jnu\Final_thesis\实验\Raidar\RaidarLLMDetect"
pip install datasets scikit-learn xgboost python-Levenshtein numpy pandas matplotlib openai scipy
```

### 0.3 实验配置文件

保存为 `configs\p0_config.json`：

```json
{
  "datasets": {
    "squad": {
      "hf_name": "rajpurkar/squad",
      "split": "train",
      "text_field": "context",
      "sample_size": 2000,
      "length_filter": {"min_words": 50, "max_words": 300}
    },
    "cc_news": {
      "hf_name": "cc_news",
      "split": "train",
      "text_field": "text",
      "sample_size": 2000,
      "length_filter": {"min_words": 50, "max_words": 300}
    }
  },
  "rewriter": {
    "model": "deepseek-chat",
    "base_url": "https://api.deepseek.com/v1",
    "temperature": 0.7,
    "max_tokens": 1024,
    "length_policy": "range",
    "length_tolerance": 0.10
  },
  "features": ["lev_ratio", "ngram_1", "ngram_2", "ngram_3", "ngram_4"],
  "classifiers": ["LR", "XGBoost", "MLP"],
  "eval": {
    "seed_start": 42,
    "num_seeds": 10,
    "test_ratio": 0.2,
    "fpr_thresholds": [0.01, 0.05]
  }
}
```

---

## Step 1：下载数据与抽样

### 1.1 下载并抽样人类文本

保存为 `scripts\p0_sample_corpus.py`：

```python
# scripts/p0_sample_corpus.py
# -*- coding: utf-8 -*-
"""
Step 1: 从 HuggingFace 下载 SQuAD / CC-News，抽样人类文本语料。
去重 + 长度过滤 + 格式过滤 + 固定 seed。
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
```

**运行命令：**

```powershell
cd "E:\jnu\Final_thesis\实验\Raidar\RaidarLLMDetect"

# SQuAD
python scripts\p0_sample_corpus.py --dataset squad --sample_size 2000 --out_json SQuAD\human_corpus.json

# CC-News
python scripts\p0_sample_corpus.py --dataset cc_news --sample_size 2000 --out_json CCNews\human_corpus.json
```

---

## Step 2：生成 AI 对照文本

保存为 `scripts\p0_generate_ai_text.py`：

```python
# scripts/p0_generate_ai_text.py
# -*- coding: utf-8 -*-
"""
Step 2: 用 DeepSeek 为每条人类文本生成同域同长度的 AI 对照文本。
采用"元信息驱动生成"（用 title 做 prompt），而非直接改写。
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
    ap.add_argument("--api_key_env", default="DEEPSEEK_API_KEY")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--tolerance", type=float, default=0.10)
    ap.add_argument("--max_retries", type=int, default=2)
    args = ap.parse_args()

    api_key = os.environ.get(args.api_key_env, "")
    assert api_key, f"请设置环境变量 {args.api_key_env}"

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
        tid = item["text_id"]
        if tid in cache:
            continue

        source = item.get("source", "squad")
        title = item.get("title", "General Topic")
        target_wc = item["word_count"]

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
            print(f"  [{i}] FAILED: {tid}")
            continue

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
```

**运行命令：**

```powershell
# 先设置 API Key
$env:DEEPSEEK_API_KEY = "你的key"

# SQuAD
python scripts\p0_generate_ai_text.py --human_json SQuAD\human_corpus.json --out_json SQuAD\ai_corpus.json

# CC-News
python scripts\p0_generate_ai_text.py --human_json CCNews\human_corpus.json --out_json CCNews\ai_corpus.json
```

---

## Step 3：DeepSeek 重写（单固定 prompt + 区间约束）

保存为 `scripts\p0_rewrite.py`：

```python
# scripts/p0_rewrite.py
# -*- coding: utf-8 -*-
"""
Step 3: 对 human 和 AI 文本各做一次 DeepSeek 重写。
使用区间约束（±10%），单固定 prompt (P0)。
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
    ap.add_argument("--api_key_env", default="DEEPSEEK_API_KEY")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--tolerance", type=float, default=0.10)
    ap.add_argument("--prompt_id", default="P0_range_10pct")
    args = ap.parse_args()

    api_key = os.environ.get(args.api_key_env, "")
    assert api_key, f"请设置环境变量 {args.api_key_env}"
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
        tid = item["text_id"]
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
```

**运行命令：**

```powershell
# SQuAD: 重写 human
python scripts\p0_rewrite.py --input_json SQuAD\human_corpus.json --out_json SQuAD\rewrite_squad_human.json

# SQuAD: 重写 AI
python scripts\p0_rewrite.py --input_json SQuAD\ai_corpus.json --out_json SQuAD\rewrite_squad_ai.json

# CC-News: 重写 human
python scripts\p0_rewrite.py --input_json CCNews\human_corpus.json --out_json CCNews\rewrite_ccnews_human.json

# CC-News: 重写 AI
python scripts\p0_rewrite.py --input_json CCNews\ai_corpus.json --out_json CCNews\rewrite_ccnews_ai.json
```

---

## Step 4：特征提取

**特征定义完全对齐 RAIDAR：**

1. **Levenshtein ratio**：`1 - lev(input, rewrite) / max(len(input), len(rewrite))`（字符级）
2. **n-gram overlap**（1~4-gram）：`|共有 n-gram 数| / |input 的 n-gram 数|`（词级）

共 5 维特征向量。

保存为 `scripts\p0_extract_features.py`：

```python
# scripts/p0_extract_features.py
# -*- coding: utf-8 -*-
"""
Step 4: 提取 Levenshtein ratio + n-gram overlap 特征。
特征定义对齐 RAIDAR 原文。
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
```

**运行命令：**

```powershell
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
```

---

## Step 5：训练分类器 + 评估（LR / XGBoost / MLP）

这是核心步骤。你的设计是：

- **LR**：主分类器，稳定可解释
- **XGBoost**：辅助，验证非线性
- **MLP**：对照，对齐 RAIDAR 原仓库

评估指标对齐 RAIDAR：**AUROC、F1、TPR@FPR=1%、TPR@FPR=5%**。

**关于复用原仓库评估代码的回答：** RAIDAR 原仓库的检测脚本（如 `detect_yelp_inv.py`）内部做的事情就是：加载特征 → 训练 LR/XGBoost → 计算 F1。其评估指标计算比较简单（主要用 sklearn 的 `f1_score`、`roc_auc_score`），你可以直接复用其指标计算逻辑。但因为你新增了数据集和 MLP，建议把评估函数单独抽出来（如下），保证三个分类器用同一套评估代码，避免不一致。原仓库的 `f1_score`、`roc_auc_score` 都来自 sklearn，你直接用同一个库同一个函数，就能保证指标实现完全一致。

保存为 `scripts\p0_train_eval.py`：

```python
# scripts/p0_train_eval.py
# -*- coding: utf-8 -*-
"""
Step 5: 训练 LR / XGBoost / MLP 分类器，评估 AUROC、F1、TPR@FPR。
10-seed 交叉评估，输出 JSON + 汇总表。
"""
import argparse, json, os
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score, confusion_matrix
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed, skipping XGBoost")

# ============================================================
# 评估函数（对齐 RAIDAR 原仓库的指标定义）
# ============================================================

def compute_tpr_at_fpr(y_true, y_prob, target_fpr):
    """
    计算 TPR@FPR=target_fpr。
    逻辑：遍历阈值，找到 FPR <= target_fpr 时的最大 TPR。
    这与 RAIDAR 的 TPR@FPR 定义一致。
    """
    from sklearn.metrics import roc_curve
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_prob)
    # 找 fpr <= target_fpr 的最大 tpr
    valid = fpr_arr <= target_fpr
    if not valid.any():
        return 0.0
    return float(tpr_arr[valid].max())

def evaluate(y_true, y_pred, y_prob):
    """返回一组指标的字典"""
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))
    auroc = float(roc_auc_score(y_true, y_prob))
    tpr_1 = compute_tpr_at_fpr(y_true, y_prob, 0.01)
    tpr_5 = compute_tpr_at_fpr(y_true, y_prob, 0.05)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": acc,
        "f1": f1,
        "auroc": auroc,
        "tpr_at_fpr_1pct": tpr_1,
        "tpr_at_fpr_5pct": tpr_5,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
    }

# ============================================================
# 分类器工厂
# ============================================================

def make_classifier(name, seed):
    if name == "LR":
        return LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            C=1.0,
            random_state=seed,
            solver="lbfgs"
        )
    elif name == "XGBoost":
        if not HAS_XGB:
            return None
        return XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=seed,
            use_label_encoder=False,
            eval_metric="logloss"
        )
    elif name == "MLP":
        return MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.15
        )
    else:
        raise ValueError(f"unknown classifier: {name}")

# ============================================================
# 单 seed 训练评估
# ============================================================

def train_eval_one_seed(X, y, clf_name, seed, test_ratio=0.2):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_ratio, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    clf = make_classifier(clf_name, seed)
    if clf is None:
        return None

    clf.fit(X_tr_s, y_tr)

    y_pred = clf.predict(X_te_s)

    # 获取概率
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_te_s)[:, 1]
    else:
        y_prob = clf.decision_function(X_te_s)

    metrics = evaluate(y_te, y_pred, y_prob)
    return metrics

# ============================================================
# 主函数
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="特征 npz 文件")
    ap.add_argument("--out_dir", required=True, help="输出目录")
    ap.add_argument("--dataset_name", required=True, help="squad / ccnews")
    ap.add_argument("--classifiers", default="LR,XGBoost,MLP")
    ap.add_argument("--seed_start", type=int, default=42)
    ap.add_argument("--num_seeds", type=int, default=10)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    args = ap.parse_args()

    # 加载特征
    npz = np.load(args.npz)
    X_human = npz["X_human"]  # label=0
    X_ai = npz["X_ai"]        # label=1

    X = np.concatenate([X_human, X_ai], axis=0)
    y = np.concatenate([
        np.zeros(X_human.shape[0], dtype=np.int64),
        np.ones(X_ai.shape[0], dtype=np.int64)
    ])

    print(f"Total samples: {len(y)} (human={X_human.shape[0]}, ai={X_ai.shape[0]})")
    print(f"Feature dim: {X.shape[1]}")

    clf_names = [c.strip() for c in args.classifiers.split(",")]
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    all_results = {}

    for clf_name in clf_names:
        print(f"\n=== {clf_name} ===")
        seed_results = []

        for seed in seeds:
            metrics = train_eval_one_seed(X, y, clf_name, seed, args.test_ratio)
            if metrics is None:
                print(f"  [seed={seed}] skipped (classifier unavailable)")
                continue
            seed_results.append(metrics)
            print(f"  [seed={seed}] acc={metrics['accuracy']:.4f} "
                  f"f1={metrics['f1']:.4f} auroc={metrics['auroc']:.4f} "
                  f"tpr@1%={metrics['tpr_at_fpr_1pct']:.4f} "
                  f"tpr@5%={metrics['tpr_at_fpr_5pct']:.4f}")

        if not seed_results:
            continue

        # 汇总
        summary = {
            "classifier": clf_name,
            "dataset": args.dataset_name,
            "num_seeds": len(seed_results),
            "seeds": seeds[:len(seed_results)],
        }

        for metric_key in ["accuracy", "f1", "auroc", "tpr_at_fpr_1pct", "tpr_at_fpr_5pct"]:
            vals = [r[metric_key] for r in seed_results]
            summary[f"{metric_key}_list"] = vals
            summary[f"{metric_key}_mean"] = float(np.mean(vals))
            summary[f"{metric_key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

        summary["per_seed"] = seed_results

        # 保存单分类器结果
        out_path = os.path.join(args.out_dir, f"{args.dataset_name}_{clf_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"  saved: {out_path}")

        all_results[clf_name] = summary

    # 保存总汇总
    summary_path = os.path.join(args.out_dir, f"{args.dataset_name}_all_classifiers.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nsaved summary: {summary_path}")

    # 打印总表
    print(f"\n{'='*80}")
    print(f"{'Classifier':<12} {'Acc':>12} {'F1':>12} {'AUROC':>12} {'TPR@1%':>12} {'TPR@5%':>12}")
    print(f"{'-'*80}")
    for clf_name, s in all_results.items():
        print(f"{clf_name:<12} "
              f"{s['accuracy_mean']:.4f}±{s['accuracy_std']:.4f} "
              f"{s['f1_mean']:.4f}±{s['f1_std']:.4f} "
              f"{s['auroc_mean']:.4f}±{s['auroc_std']:.4f} "
              f"{s['tpr_at_fpr_1pct_mean']:.4f}±{s['tpr_at_fpr_1pct_std']:.4f} "
              f"{s['tpr_at_fpr_5pct_mean']:.4f}±{s['tpr_at_fpr_5pct_std']:.4f}")

if __name__ == "__main__":
    main()
```

**运行命令：**

```powershell
# SQuAD
python scripts\p0_train_eval.py ^
  --npz outputs\p0_baseline\features\squad_features.npz ^
  --out_dir outputs\p0_baseline\metrics ^
  --dataset_name squad

# CC-News
python scripts\p0_train_eval.py ^
  --npz outputs\p0_baseline\features\ccnews_features.npz ^
  --out_dir outputs\p0_baseline\metrics ^
  --dataset_name ccnews
```

---

## Step 6：导出可复现结果（总表 + 图 + 复现信息）

保存为 `scripts\p0_export_results.py`：

```python
# scripts/p0_export_results.py
# -*- coding: utf-8 -*-
"""
Step 6: 汇总两域三分类器结果，生成论文级总表（CSV + MD）与柱状图。
"""
import argparse, json, os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_all_results(metrics_dir):
    rows = []
    for fp in sorted(glob.glob(os.path.join(metrics_dir, "*_all_classifiers.json"))):
        data = json.load(open(fp, "r", encoding="utf-8"))
        for clf_name, s in data.items():
            rows.append({
                "Dataset": s["dataset"],
                "Classifier": clf_name,
                "Accuracy": f"{s['accuracy_mean']:.4f}±{s['accuracy_std']:.4f}",
                "F1": f"{s['f1_mean']:.4f}±{s['f1_std']:.4f}",
                "AUROC": f"{s['auroc_mean']:.4f}±{s['auroc_std']:.4f}",
                "TPR@FPR=1%": f"{s['tpr_at_fpr_1pct_mean']:.4f}±{s['tpr_at_fpr_1pct_std']:.4f}",
                "TPR@FPR=5%": f"{s['tpr_at_fpr_5pct_mean']:.4f}±{s['tpr_at_fpr_5pct_std']:.4f}",
                # 数值版本用于画图
                "_acc_mean": s["accuracy_mean"],
                "_acc_std": s["accuracy_std"],
                "_f1_mean": s["f1_mean"],
                "_f1_std": s["f1_std"],
                "_auroc_mean": s["auroc_mean"],
                "_auroc_std": s["auroc_std"],
            })
    return pd.DataFrame(rows)

def save_table(df, out_dir):
    display_cols = ["Dataset", "Classifier", "Accuracy", "F1", "AUROC", "TPR@FPR=1%", "TPR@FPR=5%"]
    df_display = df[display_cols]

    csv_path = os.path.join(out_dir, "p0_baseline_results.csv")
    df_display.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"saved: {csv_path}")

    # Markdown
    md_lines = ["| " + " | ".join(display_cols) + " |"]
    md_lines.append("| " + " | ".join(["---"] * len(display_cols)) + " |")
    for _, row in df_display.iterrows():
        md_lines.append("| " + " | ".join(str(row[c]) for c in display_cols) + " |")
    md_path = os.path.join(out_dir, "p0_baseline_results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"saved: {md_path}")

def plot_metrics(df, out_dir):
    fig_dir = os.path.join(out_dir, "..", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    datasets = df["Dataset"].unique()
    classifiers = df["Classifier"].unique()

    for metric, label in [("_f1", "F1"), ("_auroc", "AUROC"), ("_acc", "Accuracy")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(datasets))
        width = 0.25

        for j, clf in enumerate(classifiers):
            means = []
            stds = []
            for ds in datasets:
                row = df[(df["Dataset"] == ds) & (df["Classifier"] == clf)]
                if len(row) == 0:
                    means.append(0); stds.append(0)
                else:
                    means.append(row[f"{metric}_mean"].values[0])
                    stds.append(row[f"{metric}_std"].values[0])
            offset = (j - (len(classifiers) - 1) / 2) * width
            ax.bar(x + offset, means, width, label=clf, yerr=stds, capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_ylabel(label)
        ax.set_title(f"P0 Baseline: {label} by Dataset & Classifier")
        ax.legend()
        ax.set_ylim(0, 1.05)
        plt.tight_layout()

        for ext in ["png", "svg"]:
            fp = os.path.join(fig_dir, f"p0_{label.lower()}.{ext}")
            plt.savefig(fp, dpi=300)
        plt.close()
        print(f"saved: {fig_dir}/p0_{label.lower()}.png/svg")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_dir", default="outputs/p0_baseline/metrics")
    args = ap.parse_args()

    df = load_all_results(args.metrics_dir)
    if df.empty:
        print("No results found!")
        return

    save_table(df, args.metrics_dir)
    plot_metrics(df, args.metrics_dir)

if __name__ == "__main__":
    main()
```

**运行命令：**

```powershell
python scripts\p0_export_results.py --metrics_dir outputs\p0_baseline\metrics
```

---

## Step 7：复现留痕

```powershell
cd "E:\jnu\Final_thesis\实验\Raidar\RaidarLLMDetect"

# 冻结依赖
New-Item -ItemType Directory -Force outputs\repro | Out-Null
python -m pip freeze > outputs\repro\requirements.txt

# 生成文件 SHA256
Get-ChildItem -Recurse outputs\p0_baseline -File | ForEach-Object {
    "{0}`t{1}" -f (Get-FileHash $_.FullName -Algorithm SHA256).Hash, $_.FullName
} | Set-Content -Encoding UTF8 outputs\repro\p0_manifest_sha256.tsv

# 记录配置
Copy-Item configs\p0_config.json outputs\repro\p0_config_snapshot.json
```

---

## 关于你三个具体问题的回答

### Q1：原仓库用 MLP，我用 LR / XGBoost，如何处理？

RAIDAR 原论文 Table 15 明确显示：绝大多数域用 **Logistic Regression**，只有 Student Essay 用 XGBoost。所以你用 LR 作为主分类器完全符合原文做法。上面的脚本同时跑 LR + XGBoost + MLP 三个，其中 MLP 用于对齐原仓库的对照。

### Q2：评估代码能否直接复用原仓库？

可以。原仓库（`detect_yelp_inv.py` 等）的评估本质上就是调用 sklearn 的 `f1_score`、`roc_auc_score`。上面脚本中的 `evaluate()` 函数使用的是**完全相同的 sklearn 函数**，保证指标实现一致。额外增加了 `TPR@FPR` 的计算（用 `roc_curve` 实现），这是原仓库没有单独封装但你开题报告要求的指标。

### Q3：区间约束 vs 精确词数？

上面的 `p0_rewrite.py` 默认使用**区间约束（±10%）**，即 prompt 中写 `between {n_min} and {n_max} words`。这比精确词数更稳健：模型不会为了凑字数而插入冗余内容或截断关键信息，语义保持更好。你的"固定到nword.md"和"SQuADCC-News 基线阶段"文档中也推荐了这种做法。

---

## 完整执行顺序速查表

| 步骤 | 命令 | 产物 | 耗时估计 |
|------|------|------|----------|
| 0 | `pip install ...` | 环境就绪 | 2 min |
| 1 | `p0_sample_corpus.py` ×2 | `human_corpus.json` ×2 | 5 min |
| 2 | `p0_generate_ai_text.py` ×2 | `ai_corpus.json` ×2 | 2-4 h（API） |
| 3 | `p0_rewrite.py` ×4 | `rewrite_*.json` ×4 | 4-8 h（API） |
| 4 | `p0_extract_features.py` ×2 | `*_features.npz` ×2 | 5 min |
| 5 | `p0_train_eval.py` ×2 | `*_all_classifiers.json` ×2 | 2 min |
| 6 | `p0_export_results.py` | 总表 CSV/MD + 柱状图 | 1 min |
| 7 | 复现留痕命令 | requirements + SHA256 | 1 min |
